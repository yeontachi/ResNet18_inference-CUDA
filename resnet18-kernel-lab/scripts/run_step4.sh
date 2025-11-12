#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/fp32/step4_layer2"
MANI="$ROOT/exports/resnet18/fp32"

# 픽스처 확인/생성
if [[ ! -f "$MANI/fixtures_step4/layer2_block0_out.bin" ]]; then
  echo "Generating Step4 fixtures..."
  python "$ROOT/tools/make_step4_fixture.py" --manifest "$MANI"
fi

if [[ ! -x "$BIN" ]]; then
  echo "Error: $BIN not found. Build first: bash scripts/build_fp32.sh"
  exit 1
fi

LOGDIR="$ROOT/logs"; mkdir -p "$LOGDIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/step4_${TS}.log"

{
  echo "[Run] $(date '+%Y. %m. %d. (%a) %H:%M:%S %Z')"
  echo "[Cmd] $BIN --manifest $MANI"
  "$BIN" --manifest "$MANI"
} | tee "$LOG"

echo "Saved log -> $LOG"
