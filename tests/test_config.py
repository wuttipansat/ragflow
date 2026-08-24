import pytest
from ragflow.config.config import get_config

def test_get_config() -> None:
    config = get_config()

    assert isinstance(config, dict)
    assert "project" in config
    assert "paths" in config
    assert "chunking" in config

