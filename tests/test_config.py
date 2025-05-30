from pathlib import Path

import pytest

from tinylm.config import TinyLMConfig, config_override, get_config


@pytest.fixture(scope="module")
def config():
    path = Path(__file__).parent / "config.yaml"
    return TinyLMConfig.from_yaml(path)


def test_config(config: TinyLMConfig):
    assert config.model_path == "gghfez/gemma-3-4b-novision"


def test_config_override(config: TinyLMConfig):
    with config_override(config, model_path="google/gemma-3-1b-it"):
        config = get_config()
        assert config.model_path == "google/gemma-3-1b-it"
