#!/usr/bin/env bash
set -euo pipefail

BUILD=build/fp32/step2_conv1_bn1_relu
MANI=exports/resnet18/fp32

mkdir -p logs
LOG=logs/step2_$(date +'%Y%m%d_%H%M%S').log

if [[ ! -x "$BUILD" ]]; then
  echo "Error: $BUILD not found or not executable. Did you build?"
  echo "Try: bash scripts/build_fp32.sh"
  exit 1
fi

# 실행 + 로그 저장
{
  echo "[Run] $(date)"
  echo "[Cmd] $BUILD --manifest $MANI --input $MANI/fixtures/input.bin --expect $MANI/fixtures/expected.bin"
  "$BUILD" --manifest "$MANI" --input "$MANI/fixtures/input.bin" --expect "$MANI/fixtures/expected.bin"
} | tee "$LOG"

echo "Saved log -> $LOG"
