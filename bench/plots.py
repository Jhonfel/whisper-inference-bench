"""Charts for the README, rendered for GitHub's light and dark themes.

Usage: python bench/plots.py   (reads bench/results-*.json, writes docs/img/*-light.svg / *-dark.svg)
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"

# Categorical slots validated for colour-blind separation on each surface (see README "Charts").
THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e1", neutral="#a3a29d",
                  s1="#2a78d6", s2="#eb6834", s3="#1baf7a"),
    "dark": dict(surface="#0d1117", ink="#f0f6fc", ink2="#c3c2b7", grid="#30363d", neutral="#6e6d68",
                 s1="#3987e5", s2="#d95926", s3="#199e70"),
}


def load(name):
    p = ROOT / "bench" / f"results-{name}.json"
    return json.load(open(p)) if p.exists() else None


def style(ax, t, xgrid=True):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    ax.grid(axis="x" if xgrid else "y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def figure(t, w, h, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=t["surface"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": t["ink"], "axes.labelcolor": t["ink2"]})
    return fig, axes


def save(fig, name, theme):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{theme}.svg", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def recipe(t, theme):
    data = load("recipe")
    if not data:
        return
    rows = data["results"]
    labels = [r["setup"].replace("transformers ", "transformers · ").replace("openai-whisper ", "OpenAI whisper · ")
              for r in rows]
    colors = [t["neutral"] if r["setup"].startswith("openai") else t["s1"] for r in rows]
    fig, (a, b) = figure(t, 10, 3.6, 2)
    for ax, key, title, fmt in ((a, "x_realtime", "Speed (× real time, higher is better)", "{:.0f}×"),
                                (b, "peak_gpu_mib", "Peak GPU memory (MiB, lower is better)", "{:,.0f}")):
        vals = [r[key] for r in rows]
        y = range(len(rows))
        ax.barh(y, vals, color=colors, height=0.55, edgecolor=t["surface"], linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v, yi, "  " + fmt.format(v), va="center", fontsize=9, color=t["ink"])
        ax.set_yticks(list(y), labels if ax is a else [""] * len(rows))
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=10.5, color=t["ink"])
        ax.set_xlim(0, max(vals) * 1.25)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        style(ax, t)
    fig.suptitle(f"20 min of LibriSpeech audio, {data['gpu']}, batch {data['batch_size']}",
                 x=0.01, ha="left", fontsize=9, color=t["ink2"], y=1.02)
    save(fig, "recipe", theme)


def kernels(t, theme):
    data = load("kernels")
    if not data:
        return
    rows = data["results"]
    L = [r["seq_len"] for r in rows]
    fig, ax = figure(t, 7.5, 3.6)
    ax.axhline(1, color=t["ink2"], linewidth=1, linestyle=(0, (4, 3)))
    ax.text(L[-1], 1.02, "PyTorch SDPA ", ha="right", va="bottom", fontsize=8.5, color=t["ink2"])
    for key, name, c in (("sage2", "SageAttention2 (INT8/FP8)", t["s2"]), ("sage3", "SageAttention3 (FP4)", t["s3"])):
        if f"{key}_ms" not in rows[0]:
            continue
        sp = [r["sdpa_ms"] / r[f"{key}_ms"] for r in rows]
        ax.plot(L, sp, color=c, linewidth=2, marker="o", markersize=5, markeredgecolor=t["surface"], label=name)
        ax.text(L[-1], sp[-1], f"  {sp[-1]:.2f}×", va="center", fontsize=9, color=t["ink"])
    ax.axvline(1500, color=t["neutral"], linewidth=1)
    ax.text(1500, ax.get_ylim()[1], " Whisper encoder\n (30 s → 1500 positions)", va="top", fontsize=8.5, color=t["ink2"])
    ax.set_xscale("log")
    shown = [x for x in L if x != 24000]  # 24,000 and 32,768 collide on a log axis
    ax.set_xticks(shown, [f"{x:,}" for x in shown], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("Sequence length (positions)", fontsize=9)
    ax.set_ylabel("Speedup over SDPA", fontsize=9)
    ax.set_title("Attention kernel alone: where quantized attention pays off", loc="left", fontsize=10.5, color=t["ink"])
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 0.88))
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    style(ax, t, xgrid=False)
    save(fig, "kernels", theme)


def profile(t, theme):
    data = load("profile")
    if not data:
        return
    rows = data["results"]
    names = {"sdpa": "PyTorch SDPA", "sage": "SageAttention2", "sage3": "SageAttention3"}
    parts = (("attention_ms", "attention", t["s2"]), ("matmul_ms", "matrix multiplications", t["s1"]),
             ("other_ms", "everything else", t["neutral"]))
    fig, ax = figure(t, 8, 2.6)
    y = range(len(rows))
    left = [0] * len(rows)
    for key, label, c in parts:
        vals = [r[key] for r in rows]
        ax.barh(y, vals, left=left, color=c, height=0.5, edgecolor=t["surface"], linewidth=2, label=label)
        left = [l + v for l, v in zip(left, vals)]
    for yi, r in zip(y, rows):
        share = r["attention_ms"] / r["total_ms"] * 100
        ax.text(r["total_ms"], yi, f"  {r['total_ms']:.0f} ms · attention {share:.0f}%", va="center", fontsize=9,
                color=t["ink"])
    ax.set_yticks(list(y), [names.get(r["attention_impl"], r["attention_impl"]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.45)
    ax.set_xlabel("GPU time per encoder pass (ms)", fontsize=9)
    ax.set_title(f"Where Whisper's encoder spends its time (batch {data['batch']} × 30 s)", loc="left",
                 fontsize=10.5, color=t["ink"])
    leg = ax.legend(frameon=False, fontsize=8.5, ncols=3, loc="lower left", bbox_to_anchor=(0, -0.42))
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    style(ax, t)
    save(fig, "profile", theme)


def sweep(t, theme):
    data = load("sweep")
    if not data:
        return
    rows = data["results"]
    names = {"sdpa": ("PyTorch SDPA", t["s1"]), "sage": ("SageAttention2", t["s2"])}
    fig, (a, b) = figure(t, 10, 3.4, 2)
    for ax, key, title in ((a, "x_realtime", "Speed (× real time)"), (b, "peak_gpu_mib", "Peak GPU memory (MiB)")):
        for impl, (label, c) in names.items():
            pts = sorted((r["batch_size"], r[key]) for r in rows if r["attention"] == impl)
            if not pts:
                continue
            ax.plot(*zip(*pts), color=c, linewidth=2, marker="o", markersize=5, markeredgecolor=t["surface"],
                    label=label)
        ax.set_xscale("log", base=2)
        xs = sorted({r["batch_size"] for r in rows})
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.set_xlabel("Batch size (30 s chunks)", fontsize=9)
        ax.set_title(title, loc="left", fontsize=10.5, color=t["ink"])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        if key == "peak_gpu_mib":
            ax.text(0.03, 0.92, "identical for both backends", transform=ax.transAxes, fontsize=9, color=t["ink2"])
        style(ax, t, xgrid=False)
    leg = a.legend(frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    save(fig, "sweep", theme)


if __name__ == "__main__":
    for theme, t in THEMES.items():
        for fn in (recipe, kernels, profile, sweep):
            fn(t, theme)
    print("charts in", OUT, sorted(p.name for p in OUT.glob("*.svg")))
