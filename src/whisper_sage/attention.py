"""SageAttention as regular Hugging Face attention backends.

This registers new implementations with transformers' ``AttentionInterface``. Nothing global is
patched: ``torch.nn.functional.scaled_dot_product_attention`` stays untouched for every other model,
and only models loaded with ``attn_implementation="sage"`` (SageAttention2, INT8 QK / FP8 PV) or
``"sage3"`` (SageAttention3, FP4, Blackwell only) use these kernels.

Calls a kernel can't serve exactly (explicit attention masks, dropout, fp32, unsupported head sizes,
very short queries such as single-token decoding steps) go to PyTorch SDPA, the same path
``attn_implementation="sdpa"`` would take, so outputs stay correct.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

import torch
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask

try:
    from sageattention import sageattn as _sage2
except ImportError:  # pragma: no cover
    _sage2 = None

try:  # SageAttention3 (FP4) — gated release, Blackwell GPUs only
    from sageattn import sageattn_blackwell as _sage3
except ImportError:  # pragma: no cover
    _sage3 = None

# Below this query length the kernel launch/quantization overhead outweighs the gain (e.g.
# autoregressive decoding with a KV cache, where each step has one query token). Those calls use SDPA.
MIN_QUERY_LEN = 128

# How many calls went through each path; handy to confirm the kernel is really doing the work.
stats: Counter[str] = Counter()


def _run_sage2(query, key, value, is_causal, scaling):
    return _sage2(query, key, value, tensor_layout="HND", is_causal=is_causal, sm_scale=scaling)


def _run_sage3(query, key, value, is_causal, scaling):
    # sageattn_blackwell centres `key` in place, so hand it a copy. It has no scale argument and
    # always applies 1/sqrt(head_dim); fold any other scale into the query (Whisper, for example,
    # pre-scales the query and passes scaling=1.0).
    head_dim = query.shape[-1]
    if scaling is not None and abs(scaling - head_dim**-0.5) > 1e-6:
        query = query * (scaling * head_dim**0.5)
    return _sage3(query, key.clone(), value, is_causal=is_causal)


_BACKENDS: dict[str, tuple[Callable, tuple[int, ...]]] = {
    "sage": (_run_sage2, (64, 96, 128)),
    "sage3": (_run_sage3, (64, 128)),
}


def _make_forward(name: str):
    run, head_dims = _BACKENDS[name]

    def forward(
        module: torch.nn.Module,
        query: torch.Tensor,  # (batch, heads, q_len, head_dim)
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        **kwargs,
    ):
        q_len, kv_len = query.shape[2], key.shape[2]
        head_dim = query.shape[-1]
        usable = (
            attention_mask is None
            and dropout == 0.0
            and query.is_cuda
            and query.dtype in (torch.float16, torch.bfloat16)
            and head_dim in head_dims
            and q_len >= MIN_QUERY_LEN
            and query.shape[1] == key.shape[1]  # no grouped-query attention
        )
        if not usable:
            stats["sdpa"] += 1
            return sdpa_attention_forward(
                module, query, key, value, attention_mask,
                dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
            )

        if is_causal is None:
            is_causal = getattr(module, "is_causal", False)
        # the kernels align the causal mask to the top-left; only use it when that matches
        is_causal = bool(is_causal) and q_len == kv_len

        stats[name] += 1
        out = run(query, key, value, is_causal, scaling)
        # transformers expects (batch, q_len, heads, head_dim)
        return out.transpose(1, 2).contiguous(), None

    return forward


def available() -> list[str]:
    """Backends that can run on this machine."""
    if not torch.cuda.is_available():
        return []
    names = []
    if _sage2 is not None:
        names.append("sage")
    if _sage3 is not None and torch.cuda.get_device_capability()[0] >= 10:
        names.append("sage3")
    return names


def register(names: list[str] | None = None) -> list[str]:
    """Make ``attn_implementation="sage"`` (and ``"sage3"`` when present) available to transformers."""
    names = available() if names is None else names
    if not names:
        raise RuntimeError(
            "SageAttention needs a CUDA GPU and the `sageattention` package "
            "(see the README for how to build it)."
        )
    for name in names:
        AttentionInterface.register(name, _make_forward(name))
        # masks are built exactly like for SDPA (None whenever a plain causal / no mask is enough)
        AttentionMaskInterface.register(name, sdpa_mask)
    return names
