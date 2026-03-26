#!/usr/bin/env bash
set -euo pipefail

# 저장된 PID를 기준으로 백그라운드 AI 서버를 종료한다.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-logs}"
PID_FILE="$LOG_DIR/chatbot.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "[INFO] PID 파일이 없어 종료할 서버가 없습니다."
  exit 0
fi

SERVER_PID="$(cat "$PID_FILE")"
if kill -0 "$SERVER_PID" 2>/dev/null; then
  kill "$SERVER_PID"
  echo "[INFO] 서버 종료 요청 완료. PID=$SERVER_PID"
else
  echo "[INFO] 이미 종료된 PID입니다. PID=$SERVER_PID"
fi

rm -f "$PID_FILE"
