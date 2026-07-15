#!/usr/bin/env python
# =========================================================================
# Hyperparameter Search Script for Cloud-Device Recommendation System
# =========================================================================
"""
Performs grid search over hyperparameters for pipeline stages.

Usage:
    # Dry run (show parameter combinations only)
    python run_hyperparam_search.py \
        --search_config ./config/search_example.yaml \
        --base_config ./config/TaobaoOpenMCC_full_DT_DIN_DIN.yaml \
        --dry_run

    # Actual search
    python run_hyperparam_search.py \
        --search_config ./config/search_example.yaml \
        --base_config ./config/TaobaoOpenMCC_full_DT_DIN_DIN.yaml \
        --output_dir ./outputs/hp_search \
        --gpu 0

    # Quick test with limited data
    python run_hyperparam_search.py \
        --search_config ./config/search_example.yaml \
        --base_config ./config/TaobaoOpenMCC_full_DT_DIN_DIN.yaml \
        --output_dir ./outputs/hp_search_test \
        --n_rows 1000 \
        --gpu 0
"""

import os
import sys
import argparse
import yaml
import json
import copy
import subprocess
import itertools
import hashlib
import re
import pandas as pd
from datetime import datetime
from collections import deque
from typing import Dict, List, Any, Tuple
import time


MAX_HYPERPARAM_EXPERIMENTS = 10_000
MAX_SAFE_IDENTIFIER_LENGTH = 128


def _safe_identifier_fragment(value: Any, fallback: str = 'value') -> str:
    """Convert a configured value into one safe path-identifier fragment."""
    fragment = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('._-')
    if not fragment:
        fragment = fallback
    if not fragment[0].isalnum():
        fragment = f'{fallback}_{fragment}'
    return fragment


def _bounded_identifier(readable: str, digest: str, prefix: str = '') -> str:
    """Bound an identifier while retaining a stable uniqueness digest."""
    suffix = f'__{digest}'
    available = MAX_SAFE_IDENTIFIER_LENGTH - len(prefix) - len(suffix)
    if available < 1:
        raise ValueError('Identifier prefix and digest exceed the safe length limit')
    base = _safe_identifier_fragment(readable, 'value')[:available].rstrip('._-') or 'value'
    return f'{prefix}{base}{suffix}'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter Search for Cloud-Device Recommendation Pipeline'
    )
    parser.add_argument('--search_config', type=str, required=True,
                        help='Path to search configuration YAML file')
    parser.add_argument('--base_config', type=str, required=True,
                        help='Path to base pipeline configuration YAML file')
    parser.add_argument('--config_dir', type=str, default='./config',
                        help='Configuration directory (for dataset_config.yaml)')
    parser.add_argument('--dataset_id', type=str, default=None,
                        help='Dataset ID to use (overrides base config)')
    parser.add_argument('--output_dir', type=str, default='./outputs/hp_search',
                        help='Output directory for experiment artifacts such as checkpoints and metrics')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Directory for search-level artifacts such as results.csv')
    parser.add_argument('--gpu', nargs='+', type=int, default=[-1],
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'retrieval', 'preranking', 'reranking'],
                        help='Pipeline execution mode')
    parser.add_argument('--n_rows', type=int, default=None,
                        help='Override debug n_rows for quick testing')
    parser.add_argument('--prev_output_path', type=str, default=None,
                        help='Path to previous stage outputs (for preranking/reranking mode)')
    parser.add_argument('--retrieval_output_path', type=str, default=None,
                        help='Path to retrieval stage outputs (for reranking mode)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only show parameter combinations without running')
    parser.add_argument('--max_experiments', type=int, default=MAX_HYPERPARAM_EXPERIMENTS,
                        help=f'Hard cap on generated combinations (max {MAX_HYPERPARAM_EXPERIMENTS})')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous search (skip completed experiments)')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--run_retrieval_test', type=int, default=0,
                        help='Whether to run retrieval test stage')
    parser.add_argument('--run_reranking_test', type=int, default=0,
                        help='Whether to run reranking test stage')
    parser.add_argument('--run_preranking_test', type=int, default=0,
                        help='Whether to run repranking test stage')
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    """Load a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str):
    """Save data to a YAML file."""
    with open(path, 'w') as f:
        yaml.dump(data, f, indent=2, allow_unicode=True)


def set_nested_value(d: dict, key_path: str, value: Any) -> dict:
    """
    Set a nested value in a dictionary using dot notation.

    Example:
        set_nested_value({}, 'model_params.embedding_dim', 64)
        -> {'model_params': {'embedding_dim': 64}}
    """
    keys = key_path.split('.')
    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return d


def get_nested_value(d: dict, key_path: str, default=None) -> Any:
    """
    Get a nested value from a dictionary using dot notation.
    """
    keys = key_path.split('.')
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def generate_param_combinations(
    search_space: Dict[str, Dict[str, List]],
    max_experiments: int = MAX_HYPERPARAM_EXPERIMENTS,
) -> List[Dict]:
    """
    Generate all parameter combinations from search space.

    Args:
        search_space: {stage_name: {param_path: [value1, value2, ...]}, ...}
                      May also contain a top-level 'seed' key:
                      {'seed': [2024, 42, 0], ...}

    Returns:
        List of (seed_override, param_dict) tuples.
        seed_override is None when no seed search is requested.
        param_dict format: {stage_name: {param_path: value, ...}, ...}
    """
    if max_experiments <= 0 or max_experiments > MAX_HYPERPARAM_EXPERIMENTS:
        raise ValueError(
            f"max_experiments must be between 1 and {MAX_HYPERPARAM_EXPERIMENTS}"
        )

    # Extract seed values (top-level, not a stage)
    seed_values = search_space.get('seed', None)
    if seed_values is not None and (not isinstance(seed_values, list) or not seed_values):
        raise ValueError("search_space.seed must be a non-empty list when provided")
    stage_search_space = {k: v for k, v in search_space.items() if k != 'seed'}

    # Flatten to list of (stage, param_path, values)
    param_specs = []
    for stage, params in stage_search_space.items():
        if not isinstance(params, dict):
            raise ValueError(f"search_space.{stage} must be a mapping")
        for param_path, values in params.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"search_space.{stage}.{param_path} must be a non-empty list")
            param_specs.append((stage, param_path, values))

    all_values = [spec[2] for spec in param_specs]
    per_seed_count = 1
    for values in all_values:
        per_seed_count *= len(values)
    seed_options = seed_values or [None]
    total_count = per_seed_count * len(seed_options)
    if total_count > max_experiments:
        raise ValueError(
            f"search space expands to {total_count} experiments, exceeding max_experiments={max_experiments}"
        )

    # The count is bounded above, so materializing the queue is safe.  Create
    # the product fresh for every seed rather than retaining an intermediate
    # Cartesian-product list.
    result = []
    for seed in seed_options:
        combos = itertools.product(*all_values) if all_values else [()]
        for combo in combos:
            param_dict = {}
            for index, value in enumerate(combo):
                stage, param_path, _ = param_specs[index]
                param_dict.setdefault(stage, {})[param_path] = value
            result.append((seed, param_dict))
    return result


def apply_params_to_config(base_config: dict, params: Dict[str, Dict[str, Any]],
                           fixed_overrides: Dict[str, Dict[str, Any]] = None) -> dict:
    """
    Apply parameter overrides to a base configuration.

    Args:
        base_config: Base pipeline configuration dict
        params: {stage_name: {param_path: value, ...}}
        fixed_overrides: Optional fixed overrides to apply (not part of search)

    Returns:
        Modified configuration dict

    Note:
        Stage names that are inside `stages:` are accessed via config['stages'][stage].
        Top-level sections like `joint_training` are accessed directly via config[stage].
    """
    config = copy.deepcopy(base_config)

    TOP_LEVEL_SECTIONS = {'joint_training', 'vocab_pruning'}

    def _get_target_dict(cfg, stage):
        """Return the dict to apply params to for the given stage name."""
        if stage in TOP_LEVEL_SECTIONS:
            if stage not in cfg:
                cfg[stage] = {}
            return cfg[stage]
        else:
            return cfg.get('stages', {}).get(stage)

    # Apply fixed overrides first
    if fixed_overrides:
        for stage, stage_params in fixed_overrides.items():
            target = _get_target_dict(config, stage)
            if target is None:
                continue
            for param_path, value in stage_params.items():
                set_nested_value(target, param_path, value)

    # Apply search params
    for stage, stage_params in params.items():
        target = _get_target_dict(config, stage)
        if target is None:
            print(f"Warning: Stage '{stage}' not found in config, skipping")
            continue
        for param_path, value in stage_params.items():
            set_nested_value(target, param_path, value)

    return config


def generate_experiment_id(params: Dict[str, Dict[str, Any]], seed_override: int = None) -> str:
    """
    Generate a unique experiment ID from parameters.

    The ID is human-readable with a short hash suffix for uniqueness.
    """
    # Build a descriptive string
    parts = []
    if seed_override is not None:
        parts.append(f"seed_{_safe_identifier_fragment(seed_override, 'seed')}")
    for stage, stage_params in sorted(params.items()):
        for param_path, value in sorted(stage_params.items()):
            # Get last part of param path for brevity
            short_name = _safe_identifier_fragment(param_path.split('.')[-1], 'param')
            # Format value
            if isinstance(value, list):
                val_str = '_'.join(str(v) for v in value)
            elif isinstance(value, float):
                val_str = f"{value:.0e}" if value < 0.01 else str(value)
            else:
                val_str = str(value)
            stage_prefix = _safe_identifier_fragment(stage, 'stage')[0]
            parts.append(
                f"{stage_prefix}{short_name}_{_safe_identifier_fragment(val_str, 'value')}"
            )

    # Create base name
    base_name = '__'.join(parts) if parts else 'baseline'

    # Add short hash for uniqueness
    hash_input = json.dumps({'seed': seed_override, 'params': params}, sort_keys=True)
    full_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]

    return _bounded_identifier(base_name, full_hash)


def build_temp_config_namespace(args: argparse.Namespace, result_dir: str, base_config: dict) -> str:
    """
    Build a stable namespace for generated temp configs.

    Concurrent searches can share the same experiment_id when they sweep the same
    hyperparameter grid across different datasets / scenarios. Include the search
    context so temp YAMLs do not overwrite each other.
    """
    dataset_id = args.dataset_id or base_config.get('dataset_id', 'unknown_dataset')
    result_tag = os.path.basename(os.path.normpath(result_dir)) or 'default'
    namespace_source = json.dumps({
        'base_config': os.path.abspath(args.base_config),
        'dataset_id': dataset_id,
        'mode': args.mode,
        'result_dir': os.path.abspath(result_dir),
        'search_config': os.path.abspath(args.search_config),
    }, sort_keys=True)
    namespace_hash = hashlib.md5(namespace_source.encode()).hexdigest()[:8]
    # Keep room for an experiment ID when rendering generated config filenames.
    safe_tag = _safe_identifier_fragment(result_tag, 'default')[:48].rstrip('._-') or 'default'
    return f'{safe_tag}__{namespace_hash}'


def build_temp_config_stem(namespace: str, experiment_id: str) -> str:
    """Build a path-safe, bounded config filename stem for one experiment."""
    digest_source = f'{namespace}\0{experiment_id}'
    digest = hashlib.sha256(digest_source.encode('utf-8')).hexdigest()[:10]
    return _bounded_identifier(
        f'{namespace}__{experiment_id}',
        digest,
        prefix='hp_temp_',
    )


def params_to_flat_dict(params: Dict[str, Dict[str, Any]], prefix: str = '') -> Dict[str, Any]:
    """
    Flatten nested params dict for CSV output.

    Example:
        {'retrieval': {'model_params.embedding_dim': 64}}
        -> {'retrieval.model_params.embedding_dim': 64}
    """
    result = {}
    for stage, stage_params in params.items():
        for param_path, value in stage_params.items():
            key = f"{prefix}{stage}.{param_path}" if prefix else f"{stage}.{param_path}"
            # Convert lists to strings for CSV
            if isinstance(value, list):
                result[key] = str(value)
            else:
                result[key] = value
    return result


def build_run_pipeline_cmd(
    config_name: str,
    config_dir: str,
    experiment_id: str,
    args: argparse.Namespace,
    script_dir: str,
    gpu_id: int,
    seed_override: int = None
) -> List[str]:
    """Build the run_pipeline.py command for an experiment."""
    run_pipeline_path = os.path.join(script_dir, 'run_pipeline.py')

    seed = seed_override if seed_override is not None else args.seed

    cmd = [
        sys.executable,
        run_pipeline_path,
        '--config', config_dir,
        '--pipeline_id', config_name,  # Just the name, not .yaml extension
        '--mode', args.mode,
        '--output_dir', args.output_dir,
        '--experiment_id', experiment_id,
        '--gpu', str(gpu_id),
        '--seed', str(seed),
        '--save_stage_outputs', '0',
    ]

    if args.dataset_id:
        cmd.extend(['--dataset_id', args.dataset_id])

    if args.n_rows:
        cmd.extend(['--n_rows', str(args.n_rows)])

    if args.run_retrieval_test:
        cmd.extend(['--run_retrieval_test', str(args.run_retrieval_test)])

    if args.prev_output_path:
        cmd.extend(['--prev_output_path', args.prev_output_path])

    if args.retrieval_output_path:
        cmd.extend(['--retrieval_output_path', args.retrieval_output_path])

    if args.run_reranking_test:
        cmd.extend(['--run_reranking_test', str(args.run_reranking_test)])

    if args.run_preranking_test:
        cmd.extend(['--run_preranking_test', str(args.run_preranking_test)])
    return cmd


def load_metrics(output_dir: str, experiment_id: str) -> Dict[str, Any]:
    """Load metrics.json if present."""
    metrics_path = os.path.join(output_dir, experiment_id, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    print(f"Warning: metrics.json not found at {metrics_path}")
    return {}


def save_results(
    results: List[Dict[str, Any]],
    output_path: str,
    all_param_keys: List[str] = None
):
    """
    Save results to CSV file.

    Args:
        results: List of result dicts with 'params', 'metrics', 'experiment_id', 'status'
        output_path: Path to output CSV
        all_param_keys: All parameter keys for consistent column ordering
    """
    rows = []
    for r in results:
        row = {
            'experiment_id': r['experiment_id'],
            'seed': r.get('seed', ''),
            'status': r['status'],
            'timestamp': r.get('timestamp', ''),
            'run_dir': r.get('run_dir', ''),
            'stage_outputs_dir': r.get('stage_outputs_dir', ''),
        }
        # Add params
        flat_params = params_to_flat_dict(r['params'])
        row.update(flat_params)
        # Add metrics
        row.update(r.get('metrics', {}))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns: experiment_id, status, timestamp, params..., metrics...
    if len(df) > 0:
        meta_cols = ['experiment_id', 'seed', 'status', 'timestamp', 'run_dir', 'stage_outputs_dir']
        param_cols = sorted([c for c in df.columns if c.startswith(('retrieval.', 'preranking.', 'reranking.', 'dtcn.', 'cloud_teacher.'))])
        metric_cols = sorted([c for c in df.columns if c not in meta_cols + param_cols])
        df = df[meta_cols + param_cols + metric_cols]

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def main():
    args = parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    result_dir = os.path.abspath(args.result_dir or args.output_dir)

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load configurations
    print(f"Loading search config: {args.search_config}")
    search_config = load_yaml(args.search_config)

    print(f"Loading base config: {args.base_config}")
    base_config = load_yaml(args.base_config)

    # Extract search space and fixed overrides
    search_space = search_config.get('search_space', {})
    fixed_overrides = search_config.get('fixed_overrides', {})

    # Apply global section overrides from search config to base config
    # Anything that is not 'search_space' or 'fixed_overrides' is treated as a config section override
    global_keys = [k for k in search_config.keys() if k not in ('search_space', 'fixed_overrides')]
    for k in global_keys:
        base_config[k] = search_config[k]
        print(f"Applied global override for section: {k}")

    # Generate parameter combinations (each element is (seed_override, params))
    try:
        param_combinations = generate_param_combinations(
            search_space,
            max_experiments=args.max_experiments,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Determine searched stages (exclude 'seed')
    searched_keys = [k for k in search_space.keys() if k != 'seed']
    seed_values = search_space.get('seed', None)

    print(f"\n{'='*60}")
    print("Hyperparameter Search Configuration")
    print(f"{'='*60}")
    print(f"Total combinations: {len(param_combinations)}")
    print(f"Stages being searched: {searched_keys}")
    if seed_values:
        print(f"Seeds being searched: {seed_values}")
    print(f"Model output directory: {args.output_dir}")
    print(f"Search result directory: {result_dir}")
    print(f"Mode: {args.mode}")
    print(f"GPU: {args.gpu}")
    if args.n_rows:
        print(f"Debug n_rows: {args.n_rows}")
    print(f"{'='*60}\n")

    # Display all combinations
    print("Parameter combinations:")
    for i, (seed_override, params) in enumerate(param_combinations):
        exp_id = generate_experiment_id(params, seed_override)
        print(f"\n  [{i+1}/{len(param_combinations)}] {exp_id}")
        if seed_override is not None:
            print(f"      seed = {seed_override}")
        for stage, stage_params in params.items():
            for param_path, value in stage_params.items():
                print(f"      {stage}.{param_path} = {value}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running experiments.")
        return

    # Create artifact directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Temp configs are saved under config/generated/<YYYY-MM>/ to keep the
    # config root clean while still letting run_pipeline load them via pipeline_id.
    temp_config_bucket = datetime.now().strftime('%Y-%m')
    temp_config_subdir = os.path.join('generated', temp_config_bucket)
    temp_config_dir = os.path.join(args.config_dir, temp_config_subdir)
    os.makedirs(temp_config_dir, exist_ok=True)
    temp_config_namespace = build_temp_config_namespace(args, result_dir, base_config)

    # Check for existing results (for resume)
    results_path = os.path.join(result_dir, 'results.csv')
    completed_experiments = set()
    existing_results = []

    if args.resume and os.path.exists(results_path):
        existing_df = pd.read_csv(results_path)
        completed_experiments = set(existing_df[existing_df['status'] == 'success']['experiment_id'].tolist())
        existing_results = existing_df.to_dict('records')
        print(f"\nResuming: Found {len(completed_experiments)} completed experiments")

    # Run experiments concurrently across provided GPUs
    results = existing_results.copy()
    free_gpus = list(args.gpu)

    pending = deque()
    for i, (seed_override, params) in enumerate(param_combinations):
        experiment_id = generate_experiment_id(params, seed_override)
        if experiment_id in completed_experiments:
            print(f"\n[{i+1}/{len(param_combinations)}] Skipping {experiment_id} (already completed)")
            continue
        pending.append((i, seed_override, params, experiment_id))

    running = {}

    def launch_experiment(gpu_id: int):
        i, seed_override, params, experiment_id = pending.popleft()
        seed_msg = f" (seed={seed_override})" if seed_override is not None else ""
        print(f"\n[{i+1}/{len(param_combinations)}] Starting {experiment_id} on GPU {gpu_id}{seed_msg}")

        modified_config = apply_params_to_config(base_config, params, fixed_overrides)

        config_stem = build_temp_config_stem(temp_config_namespace, experiment_id)
        config_name = os.path.join(temp_config_subdir, config_stem).replace(os.sep, '/')
        temp_config_path = os.path.join(args.config_dir, f'{config_name}.yaml')
        save_yaml(modified_config, temp_config_path)

        cmd = build_run_pipeline_cmd(
            config_name=config_name,
            config_dir=args.config_dir,
            experiment_id=experiment_id,
            args=args,
            script_dir=script_dir,
            gpu_id=gpu_id,
            seed_override=seed_override
        )

        print(f"Command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            cwd=script_dir,
            text=True
        )

        running[process.pid] = {
            'process': process,
            'gpu_id': gpu_id,
            'params': params,
            'seed_override': seed_override,
            'experiment_id': experiment_id,
            'temp_config_path': temp_config_path,
        }

    while pending or running:
        while pending and free_gpus:
            launch_experiment(free_gpus.pop(0))

        finished_pids = []
        for pid, info in list(running.items()):
            if info['process'].poll() is not None:
                finished_pids.append(pid)

        if not finished_pids:
            time.sleep(2)
            continue

        for pid in finished_pids:
            info = running.pop(pid)
            gpu_id = info['gpu_id']
            process = info['process']
            experiment_id = info['experiment_id']
            params = info['params']
            temp_config_path = info['temp_config_path']

            return_code = process.returncode
            metrics = load_metrics(args.output_dir, experiment_id)
            success = (return_code == 0) and bool(metrics)

            try:
                os.remove(temp_config_path)
            except OSError:
                pass

            result = {
                'experiment_id': experiment_id,
                'seed': info['seed_override'],
                'params': params,
                'metrics': metrics,
                'status': 'success' if success else 'failed',
                'timestamp': datetime.now().isoformat(),
                'run_dir': os.path.join(args.output_dir, experiment_id),
                'stage_outputs_dir': os.path.join(args.output_dir, experiment_id, 'stage_outputs'),
            }
            results.append(result)

            save_results(results, results_path)

            free_gpus.append(gpu_id)

    # Final summary
    print(f"\n{'='*60}")
    print("Search Complete!")
    print(f"{'='*60}")
    print(f"Total experiments: {len(param_combinations)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Results saved to: {results_path}")

    # Show best results if available
    if results:
        successful = [r for r in results if r['status'] == 'success' and r.get('metrics')]
        if successful:
            # Try to find a good metric to sort by
            sample_metrics = successful[0]['metrics']
            sort_metrics = [
                'reranking_valid_gAUC',
                'reranking_valid_Recall@1',
                'preranking_valid_gAUC',
                'preranking_valid_Recall@100',
                'retrieval_valid_Recall@1000',
                'retrieval_valid_nDCG@1000',
                'reranking_test_Recall@5',
                'preranking_test_Recall@100',
                'retrieval_test_Recall@1000',
                'test_auc',
            ]
            sort_by = None
            for m in sort_metrics:
                if m in sample_metrics:
                    sort_by = m
                    break

            if sort_by:
                best = max(successful, key=lambda x: x['metrics'].get(sort_by, 0))
                print(f"\nBest result (by {sort_by}):")
                print(f"  Experiment: {best['experiment_id']}")
                print(f"  {sort_by}: {best['metrics'].get(sort_by)}")


if __name__ == '__main__':
    main()
