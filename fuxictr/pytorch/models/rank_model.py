# Modified by the PriCoRec authors in 2026.
# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import torch.nn as nn
import numpy as np
import torch
import os, sys
import logging
from collections.abc import Mapping
from fuxictr.pytorch.layers import FeatureEmbeddingDict
from fuxictr.metrics import evaluate_metrics
from fuxictr.pytorch.torch_utils import get_device, get_optimizer, get_loss, get_regularizer
from fuxictr.utils import Monitor, not_in_whitelist
from tqdm import tqdm


def _load_tensor_state_dict(checkpoint):
    """Load only tensor-only checkpoints without enabling pickle object loading."""
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError as exc:
        # Torch releases before weights_only cannot safely deserialize arbitrary
        # checkpoints.  Failing closed is preferable to accepting executable data.
        raise RuntimeError(
            "Safe checkpoint loading requires a PyTorch version that supports "
            "torch.load(..., weights_only=True); refusing to load this checkpoint."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Refusing to load an unsafe or unsupported checkpoint. Checkpoints must be "
            "tensor-only state_dict artifacts created by save_weights()."
        ) from exc

    if not isinstance(state_dict, Mapping):
        raise RuntimeError("Checkpoint must contain a tensor state_dict mapping.")
    invalid_entries = [
        key for key, value in state_dict.items()
        if not isinstance(key, str) or not isinstance(value, torch.Tensor)
    ]
    if invalid_entries:
        raise RuntimeError(
            "Checkpoint contains non-tensor state and is not accepted by the safe loader."
        )
    return state_dict


class BaseModel(nn.Module):
    def __init__(self,
                 feature_map,
                 model_id="BaseModel",
                 task="binary_classification",
                 gpu=-1,
                 save_checkpoints=False,
                 monitor="AUC",
                 save_best_only=True,
                 monitor_mode="max",
                 early_stop_patience=2,
                 eval_steps=None,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 reduce_lr_on_plateau=True,
                 **kwargs):
        super(BaseModel, self).__init__()
        self.device = get_device(gpu)
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._early_stop_patience = early_stop_patience
        self._eval_steps = eval_steps # None default, that is evaluating every epoch
        self._save_best_only = save_best_only
        self._save_checkpoints = save_checkpoints
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._verbose = kwargs["verbose"]
        self.feature_map = feature_map
        self.output_activation = self.get_output_activation(task)
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + ".model"))
        self.validation_metrics = kwargs["metrics"]
        self.num_fields = feature_map.num_fields

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        self.loss_fn = get_loss(loss)

    def regularization_loss(self):
        reg_term = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            emb_params = set()
            for m_name, module in self.named_modules():
                if type(module) == FeatureEmbeddingDict:
                    for p_name, param in module.named_parameters():
                        if param.requires_grad:
                            emb_params.add(".".join([m_name, p_name]))
                            for emb_p, emb_lambda in emb_reg:
                                reg_term += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if name not in emb_params:
                        for net_p, net_lambda in net_reg:
                            reg_term += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_term

    def add_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        return loss

    def compute_loss(self, return_dict, y_true):
        loss = self.add_loss(return_dict, y_true) + self.regularization_loss()
        return loss

    def reset_parameters(self):
        def default_reset_params(m):
            # initialize nn.Linear/nn.Conv1d layers by default
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        def custom_reset_params(m):
            # initialize layers with customized init_weights()
            if hasattr(m, 'init_weights'):
                m.init_weights()
        self.apply(default_reset_params)
        self.apply(custom_reset_params)

    def get_inputs(self, inputs, feature_source=None):
        X_dict = dict()
        for feature in inputs.keys():
            if feature in self.feature_map.labels:
                continue
            if feature not in self.feature_map.features:
                continue
            spec = self.feature_map.features[feature]
            if spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(spec["source"], feature_source):
                continue
            X_dict[feature] = inputs[feature].to(self.device)
        return X_dict

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        y = inputs[labels[0]].to(self.device)
        return y.float().view(-1, 1)

    def get_group_id(self, inputs):
        return inputs[self.feature_map.group_id]

    def get_feature_group_id(self, inputs):
        return inputs[self.feature_map.feature_group_id]

    def model_to_device(self):
        self.to(device=self.device)

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            reduced_lr = max(param_group["lr"] * factor, min_lr)
            param_group["lr"] = reduced_lr
        return reduced_lr

    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self.train_epoch(data_generator)
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)
        if not self._save_checkpoints:
            logging.info("Remove checkpoints: {}".format(self.checkpoint))
            os.remove(self.checkpoint)

    def checkpoint_and_earlystop(self, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
           (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({})={:.6f} Best=({:.6f}) STOP!".format(
                self._monitor_mode, monitor_value, self._best_metric))
            if self._reduce_lr_on_plateau:
                current_lr = self.lr_decay()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({})={:.6f}"\
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps >= self._early_stop_patience:
            self._stop_training = True
            logging.info("********* Epoch={} early stop *********".format(self._epoch_index + 1))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)

    def eval_step(self):
        logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())
        self.checkpoint_and_earlystop(val_logs)
        self.train()

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0
        self.train()
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss = self.train_step(batch_data)
            train_loss += loss.item()
            if self._total_steps % self._eval_steps == 0:
                logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                train_loss = 0
                self.eval_step()
            if self._stop_training:
                break

    def evaluate(self, data_generator, metrics=None, save_predictions=False, save_dir="./predictions"):
        """
        评估模型性能

        Args:
            data_generator: 数据生成器
            metrics: 评估指标
            save_predictions: 是否保存预测结果，用于后续模型拼接
            save_dir: 保存预测结果的目录
        """
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            group_id = []
            feature_group_id = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
                if self.feature_map.feature_group_id is not None:
                    feature_group_id.extend(self.get_feature_group_id(batch_data).numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None
            feature_group_id = np.array(feature_group_id) if len(feature_group_id) > 0 else None
            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id, feature_group_id)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id, feature_group_id)
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))

            # 保存预测结果用于后续模型拼接
            if save_predictions and val_logs:
                self._save_prediction_results(y_pred, y_true, group_id, feature_group_id, val_logs, save_dir)

            return val_logs

    def _save_prediction_results(self, y_pred, y_true, group_id, feature_group_id, val_logs, save_dir):
        """
        保存预测结果到文件，便于后续模型拼接

        Args:
            y_pred: 预测值
            y_true: 真实值
            group_id: 分组ID
            feature_group_id: 特征分组ID
            val_logs: 评估结果字典
            save_dir: 保存目录
        """
        try:
            # 创建保存目录
            os.makedirs(save_dir, exist_ok=True)

            # 构建文件名
            filename = self._build_prediction_filename(val_logs)
            filepath = os.path.join(save_dir, filename + ".npz")

            # 准备保存的数据
            save_data = {
                'y_pred': y_pred,
                'y_true': y_true,
                'model_id': self.model_id
            }

            # 添加group_id（如果存在）
            if group_id is not None:
                save_data['group_id'] = group_id

            # 添加feature_group_id（如果存在）
            if feature_group_id is not None:
                save_data['feature_group_id'] = feature_group_id

            # 添加评估指标
            for key, value in val_logs.items():
                if isinstance(value, (int, float)):
                    save_data[f'metric_{key}'] = value

            # 保存数据
            np.savez_compressed(filepath, **save_data)
            logging.info(f"Prediction results saved to: {filepath}")

        except Exception as e:
            logging.warning(f"Failed to save prediction results: {str(e)}")

    def _build_prediction_filename(self, val_logs):
        """
        根据评估结果构建文件名
        格式: {model_id}_AUC_{auc_value}_groupAUC_{name}_{auc_value}_groupAUC_{name}_{auc_value}

        Args:
            val_logs: 评估结果字典

        Returns:
            str: 构建的文件名
        """
        filename_parts = [self.model_id]

        # 添加主要AUC指标
        if 'AUC' in val_logs:
            auc_value = f"{val_logs['AUC']:.4f}"
            filename_parts.extend(['AUC', auc_value])

        # 添加所有包含AUC的分组指标
        group_auc_metrics = []
        for key, value in val_logs.items():
            if 'AUC' in key and key != 'AUC' and isinstance(value, (int, float)):
                # 处理分组AUC指标
                group_name = key.replace('_AUC', '').replace('AUC_', '')
                group_value = f"{value:.4f}"
                group_auc_metrics.extend(['groupAUC', group_name, group_value])

        filename_parts.extend(group_auc_metrics)

        # 如果没有找到任何AUC指标，使用其他主要指标
        if len(filename_parts) == 1:  # 只有model_id
            for key, value in val_logs.items():
                if isinstance(value, (int, float)) and key in ['logloss', 'gAUC', 'avgAUC']:
                    metric_value = f"{value:.4f}"
                    filename_parts.extend([key, metric_value])
                    break

        # 确保文件名不会过长，限制在200字符内
        filename = '_'.join(filename_parts)
        if len(filename) > 200:
            # 截断过长的文件名，保留前180字符
            filename = filename[:180] + '_truncated'

        return filename

    def predict(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            return y_pred

    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None, feature_group_id=None):
        return evaluate_metrics(y_true, y_pred, metrics, group_id, feature_group_id)

    def save_weights(self, checkpoint):
        state_dict = self.state_dict()
        if getattr(self, '_save_fp16', False):
            # FP16 conversion: .half() creates NEW tensors, breaking shared data_ptr.
            # Detect shared tensors first, convert, then re-share to preserve dedup.
            ptr_to_key = {}
            shared_groups = {}  # master_key -> [alias_keys]
            for k, v in state_dict.items():
                ptr = v.data_ptr()
                if ptr in ptr_to_key:
                    master = ptr_to_key[ptr]
                    shared_groups.setdefault(master, []).append(k)
                else:
                    ptr_to_key[ptr] = k

            # Convert all to FP16
            state_dict = {k: v.half() if v.is_floating_point() else v
                          for k, v in state_dict.items()}

            # Re-share: point aliases to the same converted tensor
            for master, aliases in shared_groups.items():
                for alias in aliases:
                    state_dict[alias] = state_dict[master]

        torch.save(state_dict, checkpoint)


    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = _load_tensor_state_dict(checkpoint)
        # Use strict=False to handle missing keys (e.g., _diversity_emb_dict_layer
        # which is excluded from save to avoid duplicate weights)
        self.load_state_dict(state_dict, strict=False)

    def get_output_activation(self, task):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "regression":
            return nn.Identity()
        else:
            raise NotImplementedError("task={} is not supported.".format(task))

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters():
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))
