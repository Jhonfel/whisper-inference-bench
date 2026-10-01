"""Where does Whisper's encoder spend its GPU time? Kernel time grouped into attention, matrix
multiplications and everything else, for each attention backend.

Usage:
  python bench/profile_encoder.py --impls sdpa,sage,sage3 --out bench/results-profile.json
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import WhisperForConditionalGeneration

from whisper_sage import attention

ATTENTION_KERNELS = ("flash", "fmha", "attention", "attn", "qattn", "qk_int8", "sageattn", "fp4attn",
                     "quant", "mean")
MATMUL_KERNELS = ("gemm", "cutlass", "sm90_xmma", "sm80_xmma", "cublas", "nvjet", "ampere", "hopper")


def category(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ATTENTION_KERNELS):
        return "attention"
    if any(k in n for k in MATMUL_KERNELS):
        return "matmul"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--impls", default="sdpa,sage,sage3")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="bench/results-profile.json")
    args = ap.parse_args()

    attention.register([i for i in args.impls.split(",") if i.startswith("sage")])
    feats = torch.randn(args.batch, 128, 3000, device="cuda", dtype=torch.float16)
    results = []
    for impl in args.impls.split(","):
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.float16, attn_implementation=impl).cuda().eval()
        encoder = model.get_encoder()
        with torch.inference_mode():
            for _ in range(3):
                encoder(feats)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    encoder(feats)
                torch.cuda.synchronize()
        totals = {"attention": 0.0, "matmul": 0.0, "other": 0.0}
        kernels = {}
        for e in prof.key_averages():
            t = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
            if t <= 0 or e.key.startswith("ProfilerStep"):
                continue
            totals[category(e.key)] += t / 1000 / 5  # ms per encoder pass
            kernels[e.key] = t / 1000 / 5
        top = sorted(kernels.items(), key=lambda kv: -kv[1])[:8]
        r = {"attention_impl": impl, **{f"{k}_ms": round(v, 2) for k, v in totals.items()},
             "total_ms": round(sum(totals.values()), 2),
             "top_kernels": [{"kernel": k[:90], "ms": round(v, 2), "category": category(k)} for k, v in top]}
        print(json.dumps({k: v for k, v in r.items() if k != "top_kernels"}), flush=True)
        results.append(r)
        del model, encoder
        torch.cuda.empty_cache()

    json.dump({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "model": args.model,
               "batch": args.batch, "results": results}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
