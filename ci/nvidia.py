from pathlib import Path

from modal import App, Image, Mount

ROOT_PATH = Path(__file__).parent.parent
REMOTE_ROOT_PATH = "/root/tinylm"
CUDA_IMAGE_TAG = "12.8.1-cudnn-devel-ubuntu24.04"
PYTHON_VERSION = open(ROOT_PATH / ".python-version", "r").read().strip()

app = App(
    "ci_nvidia",
    image=Image.from_registry(f"nvidia/cuda:{CUDA_IMAGE_TAG}", add_python=PYTHON_VERSION)
    .pip_install("uv")
    .add_local_dir(ROOT_PATH, remote_path=REMOTE_ROOT_PATH),
)


@app.function(gpu="L4", timeout=3600)
def do_test():
    import subprocess

    subprocess.run("uv sync --group dev", check=True, shell=True, cwd=REMOTE_ROOT_PATH)
    subprocess.run("TEST_NVIDIA=1 uv run poe test", check=True, shell=True, cwd=REMOTE_ROOT_PATH)
