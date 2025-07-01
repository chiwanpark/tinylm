from functools import lru_cache

import torch
from flashinfer.rope import apply_rope_with_cos_sin_cache

from tinylm.layers.base import AcceleratedModule


class RotaryEmbedding(AcceleratedModule[tuple[torch.Tensor, torch.Tensor]]):
    cos_sin_cache: torch.Tensor

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size

        # build the cosine and sine cache
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _apply_rotary_embedding(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        # Note that this conversion is different from other frameworks such as vLLM or SGLang.
        # We added float32 conversion because float32 is required for inputs with long context.
        x1, x2 = torch.chunk(x.float(), 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat((y1, y2), dim=-1).to(x.dtype)

    @torch.compile
    def forward_torch(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        q_rotated = self._apply_rotary_embedding(query, cos, sin)
        k_rotated = self._apply_rotary_embedding(key, cos, sin)

        return q_rotated, k_rotated

    def forward_flashinfer(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rope_with_cos_sin_cache(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self.cos_sin_cache,
        )


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
