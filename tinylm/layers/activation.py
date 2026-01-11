import jax.numpy as jnp
from flax import nnx


class GeluAndMul(nnx.Module):
    def __init__(self, approximate: bool = True) -> None:
        super().__init__()
        self.approximate = approximate

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[-1] // 2
        a, b = x[..., :d], x[..., d:]
        return nnx.gelu(a, approximate=self.approximate) * b


class SiluAndMul(nnx.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[-1] // 2
        a, b = x[..., :d], x[..., d:]
        return nnx.silu(a) * b
