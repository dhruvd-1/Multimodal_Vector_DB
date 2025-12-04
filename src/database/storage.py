import os
import json
import shutil
from typing import Dict, Any, Optional


class StorageManager:
    """Manages persistent storage for the vector database."""

    def __init__(self, base_path: str):
        """
        Initialize storage manager.

        Args:
            base_path: Base directory for storage
        """
        self.base_path = base_path
        self.index_path = os.path.join(base_path, 'indices')
        self.data_path = os.path.join(base_path, 'data')
        self.config_path = os.path.join(base_path, 'config.json')

        # Create directories
        os.makedirs(self.index_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)

    def save_config(self, config: Dict[str, Any]):
        """Save database configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def load_config(self) -> Optional[Dict[str, Any]]:
        """Load database configuration."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return None

    def get_index_path(self, name: str = 'main') -> str:
        """Get path for named index."""
        return os.path.join(self.index_path, name)

    def get_data_path(self, filename: str) -> str:
        """Get path for data file."""
        return os.path.join(self.data_path, filename)

    def exists(self) -> bool:
        """Check if storage exists."""
        return os.path.exists(self.config_path)

    def clear(self):
        """Clear all storage."""
        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)
            os.makedirs(self.index_path, exist_ok=True)
            os.makedirs(self.data_path, exist_ok=True)

    def get_size(self) -> int:
        """Get total storage size in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.base_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size
