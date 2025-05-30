from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional, Self

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import YamlConfigSettingsSource

_config: Optional["TinyLMConfig"] = None


class TinyLMConfig(BaseSettings):
    model_config = SettingsConfigDict(
        extra="forbid",
    )

    model_path: str = Field()
    """The path to the language model."""

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        return cls(**YamlConfigSettingsSource(cls, path)())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)


@contextmanager
def config_override(
    config: Optional[TinyLMConfig] = None, **kwargs: Any
) -> Generator[None, None, None]:
    if not config:
        config = get_config()
    data = config.model_dump()
    data.update(kwargs)

    global _config
    config_old = _config
    _config = TinyLMConfig.from_dict(data)
    try:
        yield
    finally:
        _config = config_old


def get_config(path: Optional[Path] = None) -> TinyLMConfig:
    global _config
    if path is not None:
        _config = TinyLMConfig.from_yaml(path)
    if not _config:
        raise ValueError("Configuration not set. Please provide a path to the configuration.")
    return _config
