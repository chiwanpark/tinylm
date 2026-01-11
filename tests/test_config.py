from tinylm.config import TinyLMTrainConfig, config_override, get_config


def test_train_config(train_config: TinyLMTrainConfig):
    assert train_config.model_name == "chiwanpark/TinyLM-Ko-0.5B"


def test_train_config_override(train_config: TinyLMTrainConfig):
    with config_override(train_config, model_name="google/gemma-3-1b-it"):
        config = get_config()
        assert config.model_name == "google/gemma-3-1b-it"
