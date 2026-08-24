from pathlib import Path
from typing import Any
from functools import lru_cache

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

load_dotenv(PROJECT_ROOT / ".env")

def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH
) -> dict[str, Any]:
    """Load project configuration from a YAML file."""

    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid configuration format: {config}"
        )

    return config

@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Return cache application configuration."""
    return load_config(DEFAULT_CONFIG_PATH)

def resolve_path(path: str | Path) -> Path:
    """Convert a configured relative path to an absolute path."""

    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path
