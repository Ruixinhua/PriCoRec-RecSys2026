# =============================================================================
# Configuration Parser for Cloud-Device Recommendation System
# =============================================================================

"""
Parses and expands compact configuration format.

Handles:
- Expanding grouped feature names into individual feature definitions
- Resolving file paths with default patterns
- Validating configuration structure
"""

import os
import yaml
import logging
from typing import Dict, List, Any
from pathlib import Path


class ConfigParser:
    """
    Parses pipeline and dataset configurations.
    Expands compact feature definitions and resolves paths.
    """

    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_pipeline_config(self, path: str = None) -> Dict[str, Any]:
        """Load pipeline configuration"""
        if path is None:
            legacy_default = self.config_dir / "pipeline_config.yaml"
            smoke_default = self.config_dir / "pipeline_config" / "smoke_pipeline.yaml"
            # Keep compatibility for callers that provide the legacy flat
            # file, while making the repository's tracked smoke pipeline a
            # usable default when it is absent.
            if legacy_default.is_file() or not smoke_default.is_file():
                path = legacy_default
            else:
                path = smoke_default

        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def load_dataset_config(self,
                            path: str = None,
                            dataset_id: str = None) -> Dict[str, Any]:
        """
        Load dataset configuration.

        Args:
            path: Path to dataset config file
            dataset_id: Dataset ID to load from config

        Returns:
            Dataset configuration dict
        """
        if path is None:
            path = self.config_dir / "dataset_config.yaml"

        with open(path, 'r') as f:
            all_configs = yaml.safe_load(f)

        if dataset_id:
            if dataset_id not in all_configs:
                raise ValueError(f"Dataset '{dataset_id}' not found in config")
            config = all_configs[dataset_id]
        else:
            # Use first dataset
            dataset_id = list(all_configs.keys())[0]
            config = all_configs[dataset_id]

        config['dataset_id'] = dataset_id
        return config

    def expand_feature_cols(self,
                            feature_cols: List[Dict]) -> List[Dict]:
        """
        Expand compact feature definitions into individual features.

        Example:
            Input:  {name: [a, b, c], type: categorical, dtype: int, feature_group: FG1}
            Output: [{name: a, type: categorical, dtype: int, feature_group: FG1},
                     {name: b, type: categorical, dtype: int, feature_group: FG1},
                     {name: c, type: categorical, dtype: int, feature_group: FG1}]
        """
        expanded = []

        for feat_def in feature_cols:
            names = feat_def.get('name', [])

            # Handle single name (string) vs multiple names (list)
            if isinstance(names, str):
                expanded.append(feat_def.copy())
            elif isinstance(names, list):
                # Expand each name
                share_map = feat_def.get('share_embedding_map', {})

                for name in names:
                    new_feat = feat_def.copy()
                    new_feat['name'] = name

                    # Remove the grouped fields
                    new_feat.pop('share_embedding_map', None)

                    # Add share_embedding if in map
                    if name in share_map:
                        new_feat['share_embedding'] = share_map[name]

                    expanded.append(new_feat)
            else:
                self.logger.warning(f"Invalid name type: {type(names)}")

        self.logger.info(f"Expanded {len(feature_cols)} feature groups "
                        f"to {len(expanded)} individual features")

        return expanded

    def resolve_raw_paths(self, config: Dict) -> Dict[str, str]:
        """
        Resolve raw data file paths.

        Returns:
            Dict mapping file key to full path
        """
        root = config.get('raw_data_root', '.')
        fmt = config.get('raw_data_format', 'csv')
        files = config.get('raw_files', ['train', 'valid', 'test'])

        paths = {}
        for f in files:
            filename = f"{f}.{fmt}"
            paths[f] = os.path.join(root, filename)

        return paths

    def resolve_processed_paths(self, config: Dict) -> Dict[str, str]:
        """
        Resolve processed data file paths.

        Returns:
            Dict mapping file key to full path
        """
        root = config.get('processed_data_root', './data/processed')
        fmt = config.get('processed_data_format', 'parquet')

        # Default output files
        paths = {
            'train': os.path.join(root, f"train.{fmt}"),
            'valid': os.path.join(root, f"valid.{fmt}"),
            'test': os.path.join(root, f"test.{fmt}"),
            'feature_map': os.path.join(root, "feature_map.json"),
            'feature_vocab': os.path.join(root, "feature_vocab.json"),
        }

        return paths

    def get_features_by_group(self,
                              expanded_cols: List[Dict],
                              groups: List[str]) -> List[str]:
        """Get feature names for specified feature groups"""
        return [
            f['name'] for f in expanded_cols
            if f.get('feature_group') in groups
        ]

    def get_full_config(self,
                        pipeline_path: str = None,
                        dataset_path: str = None,
                        dataset_id: str = None) -> Dict[str, Any]:
        """
        Load and merge pipeline + dataset configs.

        Args:
            pipeline_path: Optional path to pipeline config
            dataset_path: Optional path to dataset config
            dataset_id: Optional dataset ID to override pipeline config

        Returns:
            Merged configuration with expanded features
        """
        # Load pipeline config
        pipeline_config = self.load_pipeline_config(pipeline_path)

        # Use provided dataset_id or fall back to pipeline config
        if dataset_id is None:
            dataset_id = pipeline_config.get('dataset_id')
        dataset_config = self.load_dataset_config(dataset_path, dataset_id)

        # Expand feature columns
        if 'feature_cols' in dataset_config:
            dataset_config['feature_cols_expanded'] = self.expand_feature_cols(
                dataset_config['feature_cols']
            )

        # Resolve paths
        dataset_config['raw_paths'] = self.resolve_raw_paths(dataset_config)
        dataset_config['processed_paths'] = self.resolve_processed_paths(dataset_config)

        # Merge
        full_config = {
            'pipeline': pipeline_config,
            'dataset': dataset_config
        }

        return full_config

    def validate_config(self, config: Dict) -> bool:
        """Validate configuration structure"""
        errors = []

        dataset = config.get('dataset', {})

        # Check required fields
        if 'feature_cols' not in dataset and 'feature_cols_expanded' not in dataset:
            errors.append("Missing feature_cols in dataset config")

        if 'label_col' not in dataset:
            errors.append("Missing label_col in dataset config")

        # Check raw paths exist
        for key, path in dataset.get('raw_paths', {}).items():
            if not os.path.exists(path):
                self.logger.warning(f"Raw data not found: {path}")

        if errors:
            for err in errors:
                self.logger.error(err)
            return False

        return True


def load_config(config_dir: str = "./config") -> Dict[str, Any]:
    """Convenience function to load full configuration"""
    parser = ConfigParser(config_dir)
    return parser.get_full_config()


# Test function
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    parser = ConfigParser("./cloud_device_recsys/config")
    config = parser.get_full_config()

    print("=== Dataset Config ===")
    print(f"Dataset ID: {config['dataset'].get('dataset_id')}")
    print(f"Raw paths: {json.dumps(config['dataset'].get('raw_paths', {}), indent=2)}")
    print(f"Processed paths: {json.dumps(config['dataset'].get('processed_paths', {}), indent=2)}")

    print("\n=== Expanded Features ===")
    for feat in config['dataset'].get('feature_cols_expanded', [])[:5]:
        print(f"  {feat['name']}: {feat['type']} ({feat.get('feature_group')})")
    print(f"  ... total {len(config['dataset'].get('feature_cols_expanded', []))} features")

    print("\n=== Pipeline Stages ===")
    for stage, stage_config in config['pipeline'].get('stages', {}).items():
        print(f"  {stage}: {stage_config.get('model')} -> top_k={stage_config.get('top_k')}")
