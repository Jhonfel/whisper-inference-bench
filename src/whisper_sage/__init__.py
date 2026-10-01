"""Fast Whisper speech-to-text with transformers, with SageAttention as an optional attention backend."""

from __future__ import annotations

import torch

from . import attention

__all__ = ["attention", "load_pipeline"]

DEFAULT_MODEL = "openai/whisper-large-v3-turbo"


def load_pipeline(model: str = DEFAULT_MODEL, attn: str = "sdpa", device: str | None = None):
    """A transformers ASR pipeline set up the fast way: fp16 on GPU and the chosen attention backend.

    ``attn`` is any transformers implementation (``"sdpa"``, ``"eager"``, ``"flash_attention_2"``) or one
    registered here: ``"sage"`` (SageAttention2) and ``"sage3"`` (SageAttention3, Blackwell GPUs).
    """
    from transformers import pipeline

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    on_gpu = device.startswith("cuda")
    if attn in ("sage", "sage3"):
        attention.register([attn])
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        dtype=torch.float16 if on_gpu else torch.float32,
        device=device,
        model_kwargs={"attn_implementation": attn},
    )
