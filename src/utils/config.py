import yaml
import json
from typing import Dict, Any


class Config:
    """Configuration manager for the system."""

    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize configuration.

        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict or self._default_config()

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'model': {
                'name': 'openai/clip-vit-base-patch32',
                'device': 'cpu',
                'embedding_dim': 512,
            },
            'index': {
                'type': 'flat',  # flat, ivf, pq, hnsw
                'metric': 'l2',
                'nlist': 100,  # for IVF
                'hnsw_m': 32,  # for HNSW
            },
            'video': {
                'sample_fps': 1,
                'max_frames': 100,
                'pooling': 'mean',
            },
            'audio': {
                'sample_rate': 16000,
                'n_mels': 128,
                'duration': 10.0,
            },
            'search': {
                'default_k': 10,
                'rerank_strategy': 'distance',
            },
            'storage': {
                'base_path': './data/db',
            }
        }

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    @classmethod
    def from_json(cls, json_path: str):
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)

    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def save_json(self, json_path: str):
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def get(self, key: str, default=None):
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value by key (supports dot notation)."""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def __getitem__(self, key: str):
        """Allow dict-like access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """Allow dict-like setting."""
        self.set(key, value)
