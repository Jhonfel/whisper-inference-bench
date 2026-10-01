"""Batch-size sweep: long-form throughput and peak GPU memory for each attention backend.

Each (backend, batch size) point runs in a fresh process so memory numbers don't leak between runs.

Usage:
  python bench/sweep.py --librispeech data/LibriSpeech/test-clean --impls sdpa,sage --out bench/results-sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys


def point(args):
    import numpy as np
    import torch

    sys.path.insert(0, "bench")
    from benchmark import load_librispeech, read, timed
    from whisper_sage import load_pipeline

    pieces, total = [], 0
    for p, _ in load_librispeech(args.librispeech, 10_000):
        a = read(p)
        pieces.append(a)
        total += len(a)
        if total >= args.long_minutes * 60 * 16000:
            break
    long_audio = np.concatenate(pieces)
    pipe = load_pipeline(args.model, args.impl)
    gen = {"language": "en", "task": "transcribe"}

    def run():
        pipe({"raw": long_audio, "sampling_rate": 16000}, chunk_length_s=30, batch_size=args.batch,
             generate_kwargs=gen)

    run()  # warm-up
    torch.cuda.reset_peak_memory_stats()
    seconds = timed(run, 3)
    print(json.dumps({
        "attention": args.impl, "batch_size": args.batch, "seconds": seconds,
        "x_realtime": len(long_audio) / 16000 / seconds,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
    }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--impls", default="sdpa,sage")
    ap.add_argument("--batches", default="1,2,4,8,16,24,32")
    ap.add_argument("--long-minutes", type=float, default=20)
    ap.add_argument("--out", default="bench/results-sweep.json")
    ap.add_argument("--impl")
    ap.add_argument("--batch", type=int)
    args = ap.parse_args()

    if args.impl:  # child process: measure one point
        return point(args)

    results = []
    for impl in args.impls.split(","):
        for b in map(int, args.batches.split(",")):
            out = subprocess.run(
                [sys.executable, __file__, "--librispeech", args.librispeech, "--model", args.model,
                 "--long-minutes", str(args.long_minutes), "--impl", impl, "--batch", str(b)],
                capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"},
            )
            line = [l for l in out.stdout.splitlines() if l.startswith("{")]
            if not line:
                print(out.stderr[-2000:], file=sys.stderr)
                continue
            r = json.loads(line[-1])
            print(json.dumps(r), flush=True)
            results.append(r)
    import torch

    json.dump({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "model": args.model,
               "long_audio_minutes": args.long_minutes, "results": results}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
