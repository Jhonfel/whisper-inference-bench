"""Transcribe audio files from the command line.

    whisper-sage recording.wav meeting.flac --language es --attn sage
"""

from __future__ import annotations

import argparse
import pathlib
import time

from . import DEFAULT_MODEL, audio, load_pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=pathlib.Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--attn", default="sdpa", help="sdpa (default), sage, sage3, eager, flash_attention_2")
    ap.add_argument("--language", default=None, help="e.g. es, en (default: auto-detect)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--channel", type=int, default=0, help="channel to keep from multichannel audio")
    args = ap.parse_args()

    pipe = load_pipeline(args.model, args.attn)
    gen = {"task": "transcribe", **({"language": args.language} if args.language else {})}
    for f in args.files:
        samples = audio.load(f.read_bytes(), args.channel)
        t = time.perf_counter()
        out = pipe({"raw": samples, "sampling_rate": audio.SAMPLE_RATE}, chunk_length_s=30,
                   batch_size=args.batch_size, generate_kwargs=gen)
        dt = time.perf_counter() - t
        print(f"# {f.name} ({len(samples) / audio.SAMPLE_RATE:.1f} s of audio in {dt:.2f} s)")
        print(out["text"].strip())


if __name__ == "__main__":
    main()
