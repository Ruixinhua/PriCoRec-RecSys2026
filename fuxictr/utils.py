# Modified by the PriCoRec authors in 2026.
# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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

import os
import logging
import logging.config
import yaml
import glob
import json
from collections import OrderedDict
import fuxictr
import time
import psutil


def load_config(config_dir, experiment_id):
    params = load_model_config(config_dir, experiment_id)
    data_params = load_dataset_config(config_dir, params['dataset_id'])
    params.update(data_params)
    return params

def load_model_config(config_dir, experiment_id):
    model_configs = glob.glob(os.path.join(config_dir, "model_config.yaml"))
    if not model_configs:
        model_configs = glob.glob(os.path.join(config_dir, "model_config/*.yaml"))
    if not model_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(config_dir))
    found_params = dict()
    for config in model_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if 'Base' in config_dict:
                found_params['Base'] = config_dict['Base']
            if experiment_id in config_dict:
                found_params[experiment_id] = config_dict[experiment_id]
        if len(found_params) == 2:
            break
    # Update base and exp_id settings consectively to allow overwritting when conflicts exist
    params = found_params.get('Base', {})
    params.update(found_params.get(experiment_id, {}))
    assert "dataset_id" in params, f'expid={experiment_id} is not valid in config.'
    params["model_id"] = experiment_id
    return params

def load_dataset_config(config_dir, dataset_id):
    params = {"dataset_id": dataset_id}
    dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config.yaml"))
    if not dataset_configs:
        dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config/*.yaml"))
    for config in dataset_configs:
        with open(config, "r") as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                params.update(config_dict[dataset_id])
                return params
    raise RuntimeError(f'dataset_id={dataset_id} is not found in config.')

def set_logger(params):
    dataset_id = params['dataset_id']
    model_id = params.get('model_id', '')
    log_dir = os.path.join(params.get('model_root', './checkpoints'), dataset_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, model_id + '.log')

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])
    logging.info("FuxiCTR version: " + fuxictr.__version__)

def print_to_json(data, sort_keys=True):
    new_data = dict((k, str(v)) for k, v in data.items())
    if sort_keys:
        new_data = OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)

def print_to_list(data):
    return ' - '.join('{}: {:.6f}'.format(k, v) for k, v in data.items())

def save_results_to_csv(params, experiment_id, result_filename, valid_result, test_result):
    import csv
    tunner_params_key = params.get('tunner_params_key')
    tunner_params_key = tunner_params_key.split(',') if tunner_params_key is not None else []

    base_dataset_id = str(params['dataset_id'])

    group_ids = []
    if isinstance(valid_result, dict):
        for k in valid_result.keys():
            if k.startswith('group_') and k.endswith('_ratio'):
                gid = k[len('group_'):-len('_ratio')]
                group_ids.append(gid)
    group_ids = sorted(group_ids, key=lambda x: float(x)) if group_ids else []

    header = [
        'model_id', 'dataset_id', 'group_id', 'ratio', 'count',
        'val_auc', 'val_logloss', 'test_auc', 'test_logloss'
    ] + tunner_params_key

    file_exists = os.path.exists(result_filename)
    need_header = True
    if file_exists:
        try:
            need_header = os.path.getsize(result_filename) == 0
        except Exception:
            need_header = True

    with open(result_filename, 'a+', newline='') as fcsv:
        writer = csv.writer(fcsv, lineterminator='\n')
        if not file_exists or need_header:
            writer.writerow(header)

        def get_metric(result_dict, key, default=''):
            if isinstance(result_dict, dict):
                v = round(result_dict.get(key, 0), 6)
                if 0.5 < v < 1:  # e.g. auc
                    v  = '{:.2f}'.format(v*100)
                elif 0 <= v <= 0.5:  # e.g. logloss
                    v = '{:.4f}'.format(v)
                return v
            return default

        model_id = params.get('model_id', experiment_id)

        if group_ids:
            for gid in group_ids:
                ratio_key = f'group_{gid}_ratio'
                count_key = f'group_{gid}_count'
                val_auc_key = f'AUC_group_{gid}'
                val_logloss_key = f'logloss_group_{gid}'
                test_auc_key = f'AUC_group_{gid}'
                test_logloss_key = f'logloss_group_{gid}'

                row = [
                    model_id,
                    base_dataset_id,
                    gid,
                    get_metric(valid_result, ratio_key, ''),
                    get_metric(valid_result, count_key, ''),
                    get_metric(valid_result, val_auc_key, get_metric(valid_result, 'AUC', '')),
                    get_metric(valid_result, val_logloss_key, get_metric(valid_result, 'logloss', '')),
                    get_metric(test_result, test_auc_key, get_metric(test_result, 'AUC', '')),
                    get_metric(test_result, test_logloss_key, get_metric(test_result, 'logloss', '')),
                ]
                for k in tunner_params_key:
                    row.append(params.get(k, ''))
                writer.writerow(row)
        row = [
            model_id,
            base_dataset_id,
            'all',
            '',
            '',
            get_metric(valid_result, 'AUC', ''),
            get_metric(valid_result, 'logloss', ''),
            get_metric(test_result, 'AUC', ''),
            get_metric(test_result, 'logloss', '')
        ]
        for k in tunner_params_key:
            row.append(params.get(k, ''))
        writer.writerow(row)

def get_memory_usage_for_linux():
    """
    获取 Linux 系统的内存使用情况，并突出关键指标。
    """
    memory_info = psutil.virtual_memory()

    # 将字节转换为 GB
    total_gb = round(memory_info.total / (1024 ** 3), 2)
    available_gb = round(memory_info.available / (1024 ** 3), 2)
    used_gb = round(memory_info.used / (1024 ** 3), 2)

    # 内存使用率应基于 "available" 来计算，更能反映内存压力
    percent_pressure = round((total_gb - available_gb) / total_gb * 100, 1)

    return {
        "total": total_gb,
        "available": available_gb,  # ★★★ 真正可用的内存
        "percent": percent_pressure,
        "used": used_gb,  # 仅应用使用的内存
    }

def monitor_memory_in_realtime():
    mem = get_memory_usage_for_linux()
    os.system('cls' if os.name == 'nt' else 'clear')
    print("--- 系统内存实时监控 (Linux) ---")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"总内存    : {mem['total']} GB")
    print("-------------------------------------")
    print(f"✅ 可用内存 : {mem['available']} GB")
    print(f"   内存压力 : {mem['percent']}%")
    print("-------------------------------------")
    print(f"应用已用  : {mem['used']} GB")

class Monitor(object):
    def __init__(self, kv):
        if isinstance(kv, str):
            kv = {kv: 1}
        self.kv_pairs = kv

    def get_value(self, logs):
        value = 0
        for k, v in self.kv_pairs.items():
            value += logs.get(k, 0) * v
        return value

    def get_metrics(self):
        return list(self.kv_pairs.keys())


def not_in_whitelist(element, whitelist=[]):
    if not whitelist:
        return False
    elif type(whitelist) == list:
        return element not in whitelist
    else:
        return element != whitelist


# =========================================================================
# Training Efficiency Utilities
# =========================================================================

import torch

def count_model_parameters(model, log=True):
    """Count and optionally log total / trainable / frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    info = {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen,
    }
    if log:
        logging.info(f"Model parameters: total={total:,}, trainable={trainable:,}, frozen={frozen:,}")
    return info


def get_gpu_memory_usage(device=None):
    """Return current / peak GPU memory usage in MB. Returns None if CUDA unavailable."""
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "peak_allocated_mb": round(peak_allocated, 2),
    }


class TrainingTimer:
    """Wall-clock timer for training phases (fit, epoch, eval).

    Usage:
        timer = TrainingTimer()
        timer.start("fit")
        for epoch in range(n_epochs):
            timer.start("epoch")
            ...
            timer.stop("epoch")
        timer.stop("fit")
        timer.summary()  # logs all timings
    """
    def __init__(self):
        self._starts = {}
        self._totals = {}  # accumulated durations
        self._counts = {}  # number of stop calls per phase

    def start(self, phase="fit"):
        self._starts[phase] = time.time()

    def stop(self, phase="fit"):
        if phase not in self._starts:
            return 0.0
        elapsed = time.time() - self._starts.pop(phase)
        self._totals[phase] = self._totals.get(phase, 0.0) + elapsed
        self._counts[phase] = self._counts.get(phase, 0) + 1
        return elapsed

    def get(self, phase):
        return self._totals.get(phase, 0.0)

    def summary(self, log=True):
        """Return dict of phase → {total_sec, count, avg_sec} and optionally log it."""
        results = {}
        for phase in sorted(self._totals.keys()):
            total = self._totals[phase]
            count = self._counts[phase]
            avg = total / count if count > 0 else 0.0
            results[phase] = {"total_sec": round(total, 3),
                              "count": count,
                              "avg_sec": round(avg, 3)}
        if log:
            logging.info("=== Training Efficiency Report ===")
            for phase, v in results.items():
                logging.info(f"  {phase}: total={v['total_sec']:.3f}s, "
                             f"count={v['count']}, avg={v['avg_sec']:.3f}s")
            gpu_mem = get_gpu_memory_usage()
            if gpu_mem:
                logging.info(f"  GPU memory: allocated={gpu_mem['allocated_mb']:.1f}MB, "
                             f"peak={gpu_mem['peak_allocated_mb']:.1f}MB")
        return results


def log_training_efficiency(model, timer=None):
    """One-call convenience: log parameter counts + timing + GPU memory."""
    count_model_parameters(model)
    if timer is not None:
        timer.summary()
    else:
        gpu_mem = get_gpu_memory_usage()
        if gpu_mem:
            logging.info(f"GPU memory: allocated={gpu_mem['allocated_mb']:.1f}MB, "
                         f"peak={gpu_mem['peak_allocated_mb']:.1f}MB")
