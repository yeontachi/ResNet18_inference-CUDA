#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/fp32/step8_e2e"
MANI="$ROOT/exports/resnet18/fp32"
IMDIR="$ROOT/data/imagenet_val/ILSVRC2012_img_val"   
PY="$ROOT/tools/bench_fp32_vs_torch.py"

if [[ ! -x "$BIN" ]]; then
  echo "Error: $BIN not found. Build first: bash scripts/build_fp32.sh"
  exit 1
fi

source "$ROOT/venv/bin/activate" 2>/dev/null || true

python "$PY" \
  --manifest "$MANI" \
  --bin "$BIN" \
  --imagenet_dir "$IMDIR" \
  --limit 500 \
  --warmup 5 \
  --iters 20 \
  --shuffle \
  --save_log
