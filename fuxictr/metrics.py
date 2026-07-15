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


from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import numpy as np
import pandas as pd
import multiprocessing as mp
import re
from collections import OrderedDict


def evaluate_metrics(y_true, y_pred, metrics, group_id=None, feature_group_id=None):
    """
    Evaluate metrics for predictions.

    Args:
        y_true: True labels
        y_pred: Predicted values
        metrics: List of metrics to compute
        group_id: Group IDs for group-based metrics (gAUC, avgAUC, etc.)
        feature_group_id: Feature values for feature-based group metrics (e.g., is_personalization)

    Returns:
        OrderedDict containing computed metrics
    """
    return_dict = OrderedDict()
    group_metrics = []

    # 计算总体metrics
    for metric in metrics:
        if 'group' in metric:
            metric = metric.split('_')[0]  # 处理如group_AUC等
            feature_group_results = compute_feature_group_metrics(
                y_true, y_pred, [metric], feature_group_id
            )
            return_dict.update(feature_group_results)
        if metric in ['logloss', 'binary_crossentropy']:
            return_dict[metric] = log_loss(y_true, y_pred)
        elif metric == 'AUC':
            return_dict[metric] = roc_auc_score(y_true, y_pred)
        elif metric in ["gAUC", "avgAUC", "MRR"] or metric.startswith("NDCG"):
            return_dict[metric] = 0
            group_metrics.append(metric)
        else:
            raise ValueError("metric={} not supported.".format(metric))

    # 处理需要group_id的metrics（如gAUC）
    if len(group_metrics) > 0:
        assert group_id is not None, "group_index is required."
        metric_funcs = [_resolve_group_metric(metric) for metric in group_metrics]
        score_df = pd.DataFrame({"group_index": group_id,
                                 "y_true": y_true,
                                 "y_pred": y_pred})
        results = []
        pool = mp.Pool(processes=mp.cpu_count() // 2)
        for idx, df in score_df.groupby("group_index"):
            results.append(pool.apply_async(evaluate_block, args=(df, metric_funcs)))
        pool.close()
        pool.join()
        results = [res.get() for res in results]
        sum_results = np.array(results).sum(0)
        average_result = list(sum_results[:, 0] / sum_results[:, 1])
        return_dict.update(dict(zip(group_metrics, average_result)))

    # 处理按特征分组的metrics
    if feature_group_id is not None:
        feature_group_results = compute_feature_group_metrics(
            y_true, y_pred, metrics, feature_group_id
        )
        return_dict.update(feature_group_results)

    return return_dict


def compute_feature_group_metrics(y_true, y_pred, metrics, feature_group_id):
    """
    计算按特征分组的metrics

    Args:
        y_true: True labels
        y_pred: Predicted values
        metrics: List of metrics to compute
        feature_group_id: Feature values for grouping

    Returns:
        Dict containing group metrics and ratios
    """
    result_dict = {}

    # 创建DataFrame用于分组
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'feature_group': feature_group_id
    })

    # 计算每个组的样本数量和比例
    total_samples = len(df)
    group_counts = df['feature_group'].value_counts().sort_index()

    # 添加分组比例信息
    for group_value, count in group_counts.items():
        ratio = (count / total_samples) * 100
        result_dict[f'group_{group_value}_ratio'] = ratio
        result_dict[f'group_{group_value}_count'] = count

    # 按特征分组计算metrics
    for group_value, group_df in df.groupby('feature_group'):
        group_y_true = group_df['y_true'].values
        group_y_pred = group_df['y_pred'].values

        # 为每个组计算指定的metrics
        for metric in metrics:
            if metric in ['logloss', 'binary_crossentropy']:
                if len(group_y_true) > 0:  # 确保组不为空
                    group_metric_value = log_loss(group_y_true, group_y_pred)
                    result_dict[f'{metric}_group_{group_value}'] = group_metric_value
            elif metric == 'AUC':
                # 确保组中有正负样本
                if len(group_y_true) > 0 and len(np.unique(group_y_true)) > 1:
                    group_metric_value = roc_auc_score(group_y_true, group_y_pred)
                    result_dict[f'{metric}_group_{group_value}'] = group_metric_value
                else:
                    # 如果组中只有一种标签，设置为0
                    result_dict[f'{metric}_group_{group_value}'] = 0.0
            # 注意：对于group-based metrics如gAUC，我们不在这里处理，因为它们需要特殊的group_id参数

    return result_dict

def evaluate_block(df, metric_funcs):
    res_list = []
    for fn in metric_funcs:
        v = fn(df.y_true.values, df.y_pred.values)
        if type(v) == tuple:
            res_list.append(v)
        else: # add group weight
            res_list.append((v, 1))
    return res_list

def avgAUC(y_true, y_pred):
    """ avgAUC used in MIND news recommendation """
    if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
        auc = roc_auc_score(y_true, y_pred)
        return (auc, 1)
    else: # in case all negatives or all positives for a group
        return (0, 0)

def gAUC(y_true, y_pred):
    """ gAUC defined in DIN paper """
    if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
        auc = roc_auc_score(y_true, y_pred)
        n_samples = len(y_true)
        return (auc * n_samples, n_samples)
    else: # in case all negatives or all positives for a group
        return (0, 0)

def MRR(y_true, y_pred):
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    mrr = np.sum(rr_score) / (np.sum(y_true) + 1e-12)
    return mrr


class NDCG(object):
    """Normalized discounted cumulative gain metric."""
    def __init__(self, k=1):
        self.topk = k

    def dcg_score(self, y_true, y_pred):
        order = np.argsort(y_pred)[::-1]
        y_true = np.take(y_true, order[:self.topk])
        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)

    def __call__(self, y_true, y_pred):
        idcg = self.dcg_score(y_true, y_true)
        dcg = self.dcg_score(y_true, y_pred)
        return dcg / (idcg + 1e-12)


_NDCG_METRIC_PATTERN = re.compile(r"^NDCG(?:\((?P<paren_k>[1-9][0-9]*)\)|@(?P<at_k>[1-9][0-9]*))?$")
_GROUP_METRIC_REGISTRY = {
    "gAUC": gAUC,
    "avgAUC": avgAUC,
    "MRR": MRR,
}


def _resolve_group_metric(metric):
    """Resolve supported group metrics without evaluating configuration text."""
    if metric in _GROUP_METRIC_REGISTRY:
        return _GROUP_METRIC_REGISTRY[metric]
    match = _NDCG_METRIC_PATTERN.fullmatch(metric) if isinstance(metric, str) else None
    if match:
        k = match.group("paren_k") or match.group("at_k")
        return NDCG(k=int(k) if k else 1)
    raise NotImplementedError('metrics={} not implemented.'.format(metric))
