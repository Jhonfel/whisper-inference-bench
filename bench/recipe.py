"""OpenAI's reference Whisper vs the transformers recipe (fp16 + chunked batching + efficient attention).

For each setup: WER on LibriSpeech test-clean, long-form throughput, peak GPU memory and the latency of
a single short phrase (what a voice interface sees).

Usage:
  python bench/recipe.py --librispeech data/LibriSpeech/test-clean --out bench/results-recipe.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import jiwer
import numpy as np
import torch
from transformers import WhisperTokenizer, pipeline

sys.path.insert(0, "bench")
from benchmark import load_librispeech, read  # noqa: E402

from whisper_sage import attention  # noqa: E402


def sync_time(fn) -> float:
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t


def measure(name, transcribe_one, transcribe_many, transcribe_long, utterances, phrases, long_audio, normalize):
    audios = [read(p) for p, _ in utterances]
    hyp = transcribe_many(audios)
    wer = jiwer.wer([normalize(t) for _, t in utterances], [normalize(h) for h in hyp])

    for a in phrases[:3]:
        transcribe_one(a)  # warm-up
    latency = [sync_time(lambda a=a: transcribe_one(a)) for a in phrases]

    transcribe_long(long_audio[: 16000 * 60])  # warm-up
    torch.cuda.reset_peak_memory_stats()
    long_s = statistics.median(sync_time(lambda: transcribe_long(long_audio)) for _ in range(3))
    peak = torch.cuda.max_memory_allocated() / 2**20

    r = {
        "setup": name,
        "wer": wer,
        "long_seconds": long_s,
        "x_realtime": len(long_audio) / 16000 / long_s,
        "peak_gpu_mib": peak,
        "phrase_latency_ms_median": statistics.median(latency) * 1000,
    }
    print(json.dumps(r), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", required=True)
    ap.add_argument("--utterances", type=int, default=500)
    ap.add_argument("--long-minutes", type=float, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="bench/results-recipe.json")
    ap.add_argument("--setup", type=int, default=None, help="run only this setup (index), appending to --out")
    args = ap.parse_args()

    utterances = load_librispeech(args.librispeech, args.utterances, max_seconds=30)
    phrases = [a for a in (read(p) for p, _ in load_librispeech(args.librispeech, 400)) if 4 <= len(a) / 16000 <= 8][:30]
    pieces, total = [], 0
    for p, _ in load_librispeech(args.librispeech, 10_000):
        a = read(p)
        pieces.append(a)
        total += len(a)
        if total >= args.long_minutes * 60 * 16000:
            break
    long_audio = np.concatenate(pieces)
    normalize = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo").normalize
    attention.register(["sage"])
    setups = [("openai", "large-v3"), ("openai", "turbo"),
              ("openai/whisper-large-v3", "sdpa"), ("openai/whisper-large-v3-turbo", "eager"),
              ("openai/whisper-large-v3-turbo", "sdpa"), ("openai/whisper-large-v3-turbo", "sage")]
    if args.setup is not None:
        setups = [setups[args.setup]]
    results = []

    import whisper  # OpenAI's reference implementation

    for name in [n for kind, n in setups if kind == "openai"]:
        m = whisper.load_model(name, device="cuda")
        run = lambda a, m=m: m.transcribe(a, language="en", fp16=True)["text"]  # noqa: E731
        results.append(measure(
            f"openai-whisper {name}", run, lambda auds, run=run: [run(a) for a in auds], run,
            utterances, phrases, long_audio, normalize,
        ))
        del m
        torch.cuda.empty_cache()

    gen = {"language": "en", "task": "transcribe"}
    for model, impl in [s for s in setups if s[0] != "openai"]:
        p = pipeline("automatic-speech-recognition", model=model, dtype=torch.float16, device="cuda:0",
                     model_kwargs={"attn_implementation": impl})
        bs = args.batch_size
        one = lambda a, p=p: p({"raw": a, "sampling_rate": 16000}, generate_kwargs=gen)["text"]  # noqa: E731
        many = lambda auds, p=p: [o["text"] for o in p(  # noqa: E731
            [{"raw": a, "sampling_rate": 16000} for a in auds], batch_size=bs, generate_kwargs=gen)]
        long = lambda a, p=p: p({"raw": a, "sampling_rate": 16000}, chunk_length_s=30,  # noqa: E731
                                batch_size=bs, generate_kwargs=gen)
        results.append(measure(
            f"transformers {model.split('/')[1].replace('whisper-', '')} {impl}",
            one, many, long, utterances, phrases, long_audio, normalize,
        ))
        del p
        torch.cuda.empty_cache()

    import os
    data = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "batch_size": args.batch_size,
            "utterances": len(utterances), "long_audio_minutes": round(len(long_audio) / 16000 / 60, 1),
            "results": []}
    if args.setup is not None and os.path.exists(args.out):
        data = json.load(open(args.out))
    data["results"] += results
    json.dump(data, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
