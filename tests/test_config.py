from tinylm.config import TinyLMConfig, config_override, get_config


def test_config(config: TinyLMConfig):
    assert config.model_path == "gghfez/gemma-3-4b-novision"


def test_config_override(config: TinyLMConfig):
    with config_override(config, model_path="google/gemma-3-1b-it"):
        config = get_config()
        assert config.model_path == "google/gemma-3-1b-it"
