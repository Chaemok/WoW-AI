#!/usr/bin/env bash
set -euo pipefail

# PID 파일과 프로세스 상태를 함께 확인한다.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-logs}"
PID_FILE="$LOG_DIR/chatbot.pid"
OUT_LOG="$LOG_DIR/chatbot.out.log"
ERR_LOG="$LOG_DIR/chatbot.err.log"

if [ ! -f "$PID_FILE" ]; then
  echo "[INFO] 실행 중인 서버가 없습니다."
  exit 0
fi

SERVER_PID="$(cat "$PID_FILE")"
if kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "[INFO] 실행 중입니다. PID=$SERVER_PID"
  echo "[INFO] OUT_LOG=$OUT_LOG"
  echo "[INFO] ERR_LOG=$ERR_LOG"
  exit 0
fi

echo "[INFO] PID 파일은 있지만 프로세스는 종료되어 있습니다. PID=$SERVER_PID"
exit 1
