"""Attention kernels alone: PyTorch SDPA vs SageAttention2 vs SageAttention3 across sequence lengths.

Whisper's encoder always attends over 1500 positions (30 s of audio); the sweep shows where each
kernel starts to pay off. Shapes use Whisper large-v3's 20 heads of size 64.

Usage:
  python bench/kernels.py --out bench/results-kernels.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F


def bench(fn, n: int) -> float:
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="256,512,1024,1500,3000,6000,12000,24000,32768")
    ap.add_argument("--heads", type=int, default=20)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=24000, help="keep batch*length roughly constant")
    ap.add_argument("--out", default="bench/results-kernels.json")
    args = ap.parse_args()

    kernels = {"sdpa": lambda q, k, v: F.scaled_dot_product_attention(q, k, v)}
    try:
        from sageattention import sageattn

        kernels["sage2"] = lambda q, k, v: sageattn(q, k, v, tensor_layout="HND")
    except ImportError:
        pass
    try:
        from sageattn import sageattn_blackwell

        kernels["sage3"] = lambda q, k, v: sageattn_blackwell(q, k.clone(), v)
    except ImportError:
        pass

    results = []
    for L in map(int, args.lengths.split(",")):
        B = max(1, args.tokens // L)
        q, k, v = (torch.randn(B, args.heads, L, args.head_dim, device="cuda", dtype=torch.float16) for _ in range(3))
        ref = F.scaled_dot_product_attention(q, k, v)
        n = max(10, int(2000 / (B * L * L / 1e7 + 1)))
        row = {"seq_len": L, "batch": B}
        for name, fn in kernels.items():
            row[f"{name}_ms"] = bench(lambda: fn(q, k, v), n)
            out = fn(q, k, v)
            row[f"{name}_cosine"] = F.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0).item()
        print(json.dumps(row), flush=True)
        results.append(row)
        del q, k, v, ref
        torch.cuda.empty_cache()

    json.dump({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "heads": args.heads,
               "head_dim": args.head_dim, "results": results}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
