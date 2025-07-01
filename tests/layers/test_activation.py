import pytest
import torch

from tinylm.config import TinyLMConfig, config_override
from tinylm.layers.activation import GeluAndMul, SiluAndMul
from tinylm.testutil import allclose


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gelu_and_mul_correctness(config: TinyLMConfig, device: str, is_nvidia: bool) -> None:
    if device == "cuda" and not is_nvidia:
        pytest.skip("Skipping test on non-NVIDIA environment")

    device_ = torch.device(device)
    x = torch.tensor([[-1, 2, -3, 4], [5, -6, 7, -8]], dtype=torch.bfloat16, device=device_)
    with config_override(config, use_flashinfer=False):
        layer = GeluAndMul(approximate="none")
    expected = torch.tensor([[0.4766, 7.8125], [35, 0]], dtype=torch.bfloat16, device=device_)
    output = layer(x)

    assert allclose(expected, output)


@pytest.mark.parametrize("dim", [16, 64])
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("seq_len", [1, 4])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gelu_and_mul_flashinfer(
    dim: int,
    batch_size: int,
    seq_len: int,
    approximate: str,
    dtype: torch.dtype,
    is_nvidia: bool,
    config: TinyLMConfig,
) -> None:
    if not is_nvidia:
        pytest.skip("Skipping test on non-NVIDIA environment")

    x = torch.randn(batch_size, seq_len, 2 * dim).cuda().to(dtype)
    with config_override(config, use_flashinfer=True):
        layer = GeluAndMul(approximate=approximate)
    expected_output = layer.forward_torch(x)
    output = layer(x)

    assert allclose(expected_output, output)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_silu_and_mul_correctness(config: TinyLMConfig, device: str, is_nvidia: bool) -> None:
    if device == "cuda" and not is_nvidia:
        pytest.skip("Skipping test on non-NVIDIA environment")

    device_ = torch.device(device)
    x = torch.tensor([[-1, 2, -3, 4], [5, -6, 7, -8]], dtype=torch.bfloat16, device=device_)
    with config_override(config, use_flashinfer=False):
        layer = SiluAndMul()
    expected = torch.tensor(
        [[0.8086, 7.0312], [34.7500, 0.1187]], dtype=torch.bfloat16, device=device_
    )
    output = layer(x)

    assert allclose(expected, output)


@pytest.mark.parametrize("dim", [16, 64])
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("seq_len", [1, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_silu_and_mul_flashinfer(
    dim: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    is_nvidia: bool,
    config: TinyLMConfig,
) -> None:
    if not is_nvidia:
        pytest.skip("Skipping test on non-NVIDIA environment")

    x = torch.randn(batch_size, seq_len, 2 * dim).cuda().to(dtype)
    with config_override(config, use_flashinfer=True):
        layer = SiluAndMul()
    expected_output = layer.forward_torch(x)
    output = layer(x)

    assert allclose(expected_output, output)
