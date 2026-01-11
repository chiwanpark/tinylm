from functools import lru_cache

from flax import nnx
from jax import numpy as jnp


class CosSinCache(nnx.Variable):
    pass


class RotaryEmbedding(nnx.Module):
    def __init__(self, head_size: int, rotary_dim: int, max_position_embeddings: int, base: float) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        self.cos_sin_cache = CosSinCache(self._compute_cos_sin())

    def _compute_cos_sin(self) -> jnp.ndarray:
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.rotary_dim, 2) / self.rotary_dim))
        t = jnp.arange(self.max_position_embeddings)
        freqs = jnp.outer(t, inv_freq)
        return jnp.concatenate((jnp.cos(freqs), jnp.sin(freqs)), axis=-1)

    def __call__(self, positions: jnp.ndarray, query: jnp.ndarray, key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        cos_sin = self.cos_sin_cache[positions]

        cos, sin = jnp.split(cos_sin, 2, axis=-1)

        q_rot = self._apply_rotary_embedding(query, cos, sin)
        k_rot = self._apply_rotary_embedding(key, cos, sin)

        return q_rot, k_rot

    def _apply_rotary_embedding(self, x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
        dtype = x.dtype
        x = x.astype(jnp.float32)
        x1, x2 = jnp.split(x, 2, axis=-1)

        cos = jnp.expand_dims(cos, axis=-2)
        sin = jnp.expand_dims(sin, axis=-2)

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        return jnp.concatenate((y1, y2), axis=-1).astype(dtype)


@lru_cache(maxsize=1)
def get_rotary_embedding(
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
) -> RotaryEmbedding:
    return RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
    )
