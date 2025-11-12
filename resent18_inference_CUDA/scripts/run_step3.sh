#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/fp32/step3_layer1"
MANI="$ROOT/exports/resnet18/fp32"

LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/step3_${TS}.log"

if [[ ! -x "$BIN" ]]; then
  echo "Error: $BIN not found or not executable. Build first: bash scripts/build_fp32.sh"
  exit 1
fi

# (선택) 픽스처 확인
FIXDIR="$MANI/fixtures_step3"
if [[ ! -d "$FIXDIR" ]]; then
  echo "Fixtures for step3 not found. Generating..."
  python "$ROOT/tools/make_step3_fixture.py" --manifest "$MANI"
fi

{
  echo "[Run] $(date '+%Y. %m. %d. (%a) %H:%M:%S %Z')"
  echo "[Cmd] $BIN --manifest $MANI"
  "$BIN" --manifest "$MANI"
} | tee "$LOG"

echo "Saved log -> $LOG"
