#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/fp32/step8_e2e"
MANI="$ROOT/exports/resnet18/fp32"
LOG="$ROOT/logs/step8_$(date +%Y%m%d_%H%M%S).log"

# (추가) Step8용 피처 생성
python "$ROOT/tools/make_step8_fixture.py" --manifest "$MANI"

if [[ ! -x "$BIN" ]]; then
  echo "Error: $BIN not found. Build first: bash scripts/build_fp32.sh"
  exit 1
fi

{
  echo "[Run] $(date)"
  echo "[Cmd] $BIN --manifest $MANI"
  "$BIN" --manifest "$MANI"
} | tee "$LOG"

echo "Saved log -> $LOG"
