"""Configuration management using YAML files."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration class for managing project settings."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters
        """
        self._config = config_dict
        self._update_attributes()

    def _update_attributes(self) -> None:
        """Convert nested dictionaries to Config objects."""
        for key, value in self._config.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config object back to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        result = {}
        for key, value in self._config.items():
            if isinstance(value, dict):
                result[key] = Config(value).to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key is not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return hasattr(self, key) or key in self._config


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.
                    If None, looks for config.yaml in project root.

    Returns:
        Config object with loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if config_path is None:
        # Default to config.yaml in project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Determine project root (parent of config file directory)
    project_root = config_path.parent

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        config_dict = {}

    # Expand environment variables and resolve relative paths
    config_dict = _expand_env_vars(config_dict)
    config_dict = _resolve_paths(config_dict, project_root)

    return Config(config_dict)


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand environment variables in configuration.

    Args:
        obj: Configuration object (dict, list, or str)

    Returns:
        Object with environment variables expanded
    """
    if isinstance(obj, dict):
        return {key: _expand_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        return os.path.expandvars(obj)
    else:
        return obj


def _resolve_paths(obj: Any, project_root: Path) -> Any:
    """Recursively resolve relative paths in configuration.

    Args:
        obj: Configuration object (dict, list, or str)
        project_root: Project root directory

    Returns:
        Object with paths resolved
    """
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if key == "paths" and isinstance(value, dict):
                # Resolve all paths in the paths section
                result[key] = {
                    k: str(project_root / v) if isinstance(v, str) and not Path(v).is_absolute() else v
                    for k, v in value.items()
                }
            else:
                result[key] = _resolve_paths(value, project_root)
        return result
    elif isinstance(obj, list):
        return [_resolve_paths(item, project_root) for item in obj]
    else:
        return obj
