import os
import shutil
import subprocess
from pathlib import Path
from typing import Generator

import pytest

from tinylm.config import TinyLMConfig, get_config


@pytest.fixture(scope="session")
def is_nvidia() -> bool:
    if skip := os.environ.get("SKIP_NVIDIA_TESTS"):
        if skip.lower() in ("1", "true", "yes"):
            return False
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                check=True,
                text=True,
                capture_output=True,
            )
            gpu_count = int(result.stdout.strip())
            return gpu_count > 0
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return False
    return False


@pytest.fixture(scope="session")
def config() -> Generator[TinyLMConfig, None, None]:
    path = Path(__file__).parent / "config.yaml"
    yield get_config(path)
