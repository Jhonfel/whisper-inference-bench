"""Compare Whisper with PyTorch SDPA vs SageAttention on the same GPU, model and settings.

Measures, for each attention backend:
  - accuracy: word error rate on LibriSpeech test-clean utterances
  - end-to-end speed: long-form transcription throughput (audio seconds per wall second)
  - encoder speed: forward pass over a batch of 30 s windows (where SageAttention actually runs)
  - peak GPU memory

Usage:
  python bench/benchmark.py --librispeech data/LibriSpeech/test-clean
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import time

import jiwer
import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

from whisper_sage import attention


def load_librispeech(root: str, n: int, max_seconds: float | None = None):
    items = []
    for trans in sorted(glob.glob(os.path.join(root, "*", "*", "*.trans.txt"))):
        folder = os.path.dirname(trans)
        for line in open(trans):
            uid, text = line.strip().split(" ", 1)
            path = os.path.join(folder, uid + ".flac")
            if max_seconds and sf.info(path).duration > max_seconds:
                continue  # keep the accuracy set short-form (one 30 s window per utterance)
            items.append((path, text))
            if len(items) == n:
                return items
    return items


def read(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    assert sr == 16000
    return audio


def timed(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def run(impl: str, args, utterances, long_audio) -> dict:
    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        dtype=torch.float16,
        device="cuda:0",
        model_kwargs={"attn_implementation": impl},
    )
    gen = {"language": "en", "task": "transcribe"}
    normalize = pipe.tokenizer.normalize

    # accuracy
    audios = [{"raw": read(p), "sampling_rate": 16000} for p, _ in utterances]
    outs = pipe(audios, batch_size=args.batch_size, generate_kwargs=gen)
    hyp = [normalize(o["text"]) for o in outs]
    ref = [normalize(t) for _, t in utterances]
    wer = jiwer.wer(ref, hyp)

    # end-to-end long-form throughput (chunked, batched)
    def transcribe_long():
        pipe({"raw": long_audio, "sampling_rate": 16000}, chunk_length_s=30, batch_size=args.batch_size, generate_kwargs=gen)

    transcribe_long()  # warm-up
    torch.cuda.reset_peak_memory_stats()
    e2e = timed(transcribe_long, args.repeats)
    peak_e2e = torch.cuda.max_memory_allocated() / 2**20

    # encoder alone on a full batch of 30 s windows
    model = pipe.model
    feats = pipe.feature_extractor(
        [long_audio[i * 480000:(i + 1) * 480000] for i in range(args.batch_size)],
        sampling_rate=16000, return_tensors="pt",
    ).input_features.to("cuda:0", torch.float16)
    encoder = model.get_encoder()

    @torch.inference_mode()
    def encode():
        encoder(feats)

    encode()  # warm-up
    torch.cuda.reset_peak_memory_stats()
    enc = timed(encode, args.repeats * 5)
    peak_enc = torch.cuda.max_memory_allocated() / 2**20

    attention.stats.clear()
    encode()
    calls = dict(attention.stats)

    seconds = len(long_audio) / 16000
    del pipe, model, encoder
    torch.cuda.empty_cache()
    return {
        "attention": impl,
        "wer": wer,
        "e2e_seconds": e2e,
        "e2e_realtime_factor": seconds / e2e,
        "e2e_peak_mib": peak_e2e,
        "encoder_ms": enc * 1000,
        "encoder_peak_mib": peak_enc,
        "encoder_attention_calls": calls,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--utterances", type=int, default=500)
    ap.add_argument("--long-minutes", type=float, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--impls", default="sdpa,sage")
    ap.add_argument("--out", default="bench/results.json")
    args = ap.parse_args()

    attention.register()
    utterances = load_librispeech(args.librispeech, args.utterances, max_seconds=30)
    # long-form input: consecutive utterances glued together
    pieces, total = [], 0
    for p, _ in load_librispeech(args.librispeech, 10_000):
        a = read(p)
        pieces.append(a)
        total += len(a)
        if total >= args.long_minutes * 60 * 16000:
            break
    long_audio = np.concatenate(pieces)

    results = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "model": args.model,
        "batch_size": args.batch_size,
        "utterances": len(utterances),
        "long_audio_minutes": round(len(long_audio) / 16000 / 60, 1),
        "runs": [run(impl, args, utterances, long_audio) for impl in args.impls.split(",")],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
