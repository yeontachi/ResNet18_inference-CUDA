#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yeonjjhh/바탕화면/repo/DLQ/CUDA/resnet18-kernel-lab"
BIN="$ROOT/build/fp32/step8_e2e"
MANI="$ROOT/exports/resnet18/fp32"
IMDIR="$ROOT/data/imagenet_val/ILSVRC2012_img_val"

# 먼저 5장으로 스모크(권장)
python "$ROOT/tools/bench_fp32_vs_torch_e2e.py" \
  --manifest "$MANI" \
  --bin "$BIN" \
  --imagenet_dir "$IMDIR" \
  --limit 5 --shuffle

# 본 런 (500장)
python "$ROOT/tools/bench_fp32_vs_torch_e2e.py" \
  --manifest "$MANI" \
  --bin "$BIN" \
  --imagenet_dir "$IMDIR" \
  --limit 500 --shuffle
