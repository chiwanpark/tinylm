from pathlib import Path

from modal import App, Image

ROOT_PATH = Path(__file__).parent.parent
REMOTE_ROOT_PATH = "/root/tinylm"
CUDA_IMAGE_TAG = "12.8.1-cudnn-devel-ubuntu24.04"
PYTHON_VERSION = "3.12"

app = App(
    "ci_nvidia",
    image=Image.from_registry(f"nvidia/cuda:{CUDA_IMAGE_TAG}", add_python=PYTHON_VERSION)
    .apt_install("python3-dev", "build-essential", "git", "libexpat1-dev")
    .pip_install("uv")
    .add_local_dir(
        ROOT_PATH,
        remote_path=REMOTE_ROOT_PATH,
        ignore=lambda p: p.is_relative_to(".venv"),  # ignore virtualenv
    ),
)


@app.function(gpu="L4", timeout=3600)
def do_test():
    import subprocess

    kwargs = {
        "check": True,
        "shell": True,
        "cwd": REMOTE_ROOT_PATH,
    }
    subprocess.run("uv sync --group dev --compile-bytecode", **kwargs)
    subprocess.run("TEST_NVIDIA=1 uv run --group dev poe test", **kwargs)
