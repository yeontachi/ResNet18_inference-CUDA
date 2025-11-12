#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/fp32/step6_layer4"
MANI="$ROOT/exports/resnet18/fp32"
FIXDIR="$MANI/fixtures_step6"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# 1) 피처 없으면 생성
if [[ ! -f "$FIXDIR/layer3_out.bin" ]] || \
   [[ ! -f "$FIXDIR/layer4_block0_out.bin" ]] || \
   [[ ! -f "$FIXDIR/layer4_block1_out.bin" ]]; then
  echo "Generating Step6 fixtures..."
  python "$ROOT/tools/make_step6_fixture.py" --manifest "$MANI"
fi

# 2) 바이너리 확인
if [[ ! -x "$BIN" ]]; then
  echo "Error: $BIN not found. Build first: bash scripts/build_fp32.sh"
  exit 1
fi

# 3) 실행 + 로그 저장
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/step6_${TS}.log"

{
  echo "[Run] $(date)"
  echo "[Cmd] $BIN --manifest $MANI"
  "$BIN" --manifest "$MANI"
} | tee "$LOG"

echo "Saved log -> $LOG"
