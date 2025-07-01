import pytest
import torch

from tinylm.config import TinyLMConfig, config_override
from tinylm.layers.layernorm import RMSNorm
from tinylm.testutil import allclose


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rms_norm_correctness(config: TinyLMConfig, is_nvidia: bool, device: str) -> None:
    if not is_nvidia and device == "cuda":
        pytest.skip("Skipping CUDA test on non-NVIDIA hardware")

    device_ = torch.device(device)
    with config_override(config, use_flashinfer=False):
        rms_norm = RMSNorm(dim=3).to(device_)

    x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=torch.bfloat16, device=device_)

    output = rms_norm(x)
    expected = torch.tensor(
        [[0.4629, 0.9258, 1.3906], [-0.4629, -0.9258, -1.3906]],
        dtype=torch.bfloat16,
        device=device_,
    )

    assert isinstance(output, torch.Tensor)
    assert allclose(expected, output)


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("dim", [16, 32])
def test_rms_norm_flashinfer(
    config: TinyLMConfig, is_nvidia: bool, batch_size: int, dim: int
) -> None:
    if not is_nvidia:
        pytest.skip("Skipping FlashInfer test on non-NVIDIA hardware")

    device = torch.device("cuda")
    with config_override(config, use_flashinfer=True):
        rms_norm = RMSNorm(dim=dim).to(device=device, dtype=torch.bfloat16)

    x = torch.randn(batch_size, dim, dtype=torch.bfloat16, device=device)
    expected = rms_norm.forward_torch(x.clone())
    output = rms_norm(x.clone())

    assert isinstance(output, torch.Tensor)
    assert isinstance(expected, torch.Tensor)
    assert allclose(expected, output)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_add_rms_norm_correctness(config: TinyLMConfig, is_nvidia: bool, device: str) -> None:
    if not is_nvidia and device == "cuda":
        pytest.skip("Skipping CUDA test on non-NVIDIA hardware")

    device_ = torch.device(device)
    with config_override(config, use_flashinfer=False):
        rms_norm = RMSNorm(dim=3).to(device_)

    x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=torch.bfloat16, device=device_)
    residual = torch.tensor(
        [[0.5, 1.5, 2.5], [-0.5, -1.5, -2.5]], dtype=torch.bfloat16, device=device_
    )
    expected_x = torch.tensor(
        [[0.3887, 0.9062, 1.4219], [-0.3887, -0.9062, -1.4219]],
        dtype=torch.bfloat16,
        device=device_,
    )
    expected_residual = torch.tensor(
        [[1.50, 3.50, 5.50], [-1.50, -3.50, -5.50]], dtype=torch.bfloat16, device=device_
    )

    output = rms_norm(x, residual)

    assert isinstance(output, tuple)
    assert allclose(expected_x, output[0])
    assert allclose(expected_residual, output[1])


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("dim", [16, 32])
def test_add_rms_norm_flashinfer(
    config: TinyLMConfig, is_nvidia: bool, batch_size: int, dim: int
) -> None:
    if not is_nvidia:
        pytest.skip("Skipping FlashInfer test on non-NVIDIA hardware")

    device = torch.device("cuda")
    with config_override(config, use_flashinfer=True):
        rms_norm = RMSNorm(dim=dim).to(device=device, dtype=torch.bfloat16)

    x = torch.randn(batch_size, dim, dtype=torch.bfloat16, device=device)
    residual = torch.randn_like(x)
    expected = rms_norm.forward_torch(x.clone(), residual.clone())
    output = rms_norm(x.clone(), residual.clone())

    assert isinstance(output, tuple)
    assert allclose(expected[0], output[0])
    assert allclose(expected[1], output[1])
