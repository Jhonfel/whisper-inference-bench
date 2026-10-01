"""Small HTTP speech-to-text server: POST audio, get text back.

    whisper-sage-server --attn sage --language es --port 5200

    curl -F file=@phrase.wav -F channel=0 http://localhost:5200/stt
"""

from __future__ import annotations

import argparse
import time

import torch
from flask import Flask, jsonify, request

from . import DEFAULT_MODEL, attention, audio, load_pipeline


def create_app(model: str = DEFAULT_MODEL, attn: str = "sdpa", language: str | None = None,
               batch_size: int = 16) -> Flask:
    pipe = load_pipeline(model, attn)
    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language
    app = Flask(__name__)

    @app.post("/stt")
    def stt():
        if "file" not in request.files:
            return jsonify(error="send the audio as form field 'file'"), 400
        channel = request.form.get("channel", 0, type=int)
        try:
            samples = audio.load(request.files["file"].read(), channel)
        except Exception as e:  # unreadable / unsupported file
            return jsonify(error=f"could not read audio: {e}"), 400
        t = time.perf_counter()
        out = pipe({"raw": samples, "sampling_rate": audio.SAMPLE_RATE}, chunk_length_s=30,
                   batch_size=batch_size, generate_kwargs=generate_kwargs)
        return jsonify(
            text=out["text"].strip(),
            seconds_of_audio=round(len(samples) / audio.SAMPLE_RATE, 2),
            processing_ms=round((time.perf_counter() - t) * 1000, 1),
        )

    @app.get("/health")
    def health():
        info = {"status": "ok", "model": model, "attention": attn, "device": str(pipe.device)}
        if attn.startswith("sage"):
            info["attention_calls"] = dict(attention.stats)  # how many calls each kernel really served
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_allocated_mib"] = round(torch.cuda.memory_allocated() / 2**20)
        return jsonify(info)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--attn", default="sdpa", help="sdpa (default), sage, sage3, eager, flash_attention_2")
    ap.add_argument("--language", default=None, help="e.g. es, en (default: auto-detect)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5200)
    args = ap.parse_args()
    create_app(args.model, args.attn, args.language, args.batch_size).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
