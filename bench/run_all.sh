#!/usr/bin/env bash
# Runs every measurement in sequence (one GPU job at a time). Two environments:
#   .venv        torch 2.8 / CUDA 12.8 + SageAttention 2.2 (the stack the server uses)
#   .venv-cu130  torch 2.14 / CUDA 13   + SageAttention 2.2 and 3 built from source
set -u
cd "$(dirname "$0")/.."
LS=data/LibriSpeech/test-clean
export PYTHONPATH=src
log() { echo "[$(date +%T)] $*"; }

log "recipe (OpenAI reference vs transformers setups)"
rm -f bench/results-recipe.json
for i in 0 1 2 3 4 5; do .venv/bin/python bench/recipe.py --librispeech $LS --setup $i 2>/dev/null | grep '^{'; done

log "batch-size sweep"
.venv/bin/python bench/sweep.py --librispeech $LS --impls sdpa,sage --out bench/results-sweep.json 2>/dev/null

log "attention kernels vs sequence length"
.venv-cu130/bin/python bench/kernels.py --out bench/results-kernels.json 2>/dev/null

log "encoder profile"
.venv-cu130/bin/python bench/profile_encoder.py --impls sdpa,sage,sage3 --out bench/results-profile.json 2>/dev/null

log "done"
