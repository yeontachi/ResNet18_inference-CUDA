#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/fp32/step7_head"
MANI="$ROOT/exports/resnet18/fp32"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# 1) 픽스처 생성
python "$ROOT/tools/make_step7_fixture.py" --manifest "$MANI"

# 2) 빌드
cmake -S "$ROOT/cpp" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$ROOT/build" -j >/dev/null

if [[ ! -x "$BIN" ]]; then
  echo "Error: $BIN not found. Build first."
  exit 1
fi

# 3) 실행 + 로그 저장
ts="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/step7_${ts}.log"
{
  echo "[Run] $(date '+%Y. %m. %d. (%a) %H:%M:%S %Z')"
  echo "[Cmd] $BIN --manifest $MANI"
  "$BIN" --manifest "$MANI"
} | tee "$LOG"

echo "Saved log -> $LOG"
