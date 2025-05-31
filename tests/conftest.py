import os
from pathlib import Path
from typing import Generator

import pytest

from tinylm.config import TinyLMConfig, get_config


@pytest.fixture(scope="session")
def is_nvidia() -> Generator[bool, None, None]:
    flag = os.environ.get("TEST_NVIDIA", "0").lower() in ("1", "true")
    if not flag:
        pytest.skip("TEST_NVIDIA is not set to true; thus skip this test!")
    yield True


@pytest.fixture(scope="session")
def config() -> Generator[TinyLMConfig, None, None]:
    path = Path(__file__).parent / "config.yaml"
    yield get_config(path)
