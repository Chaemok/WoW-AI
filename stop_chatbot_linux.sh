#!/usr/bin/env bash
set -euo pipefail

# 종료 스크립트도 같은 로그 위치를 보도록 루트를 맞춘다.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${ENV_FILE:-.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

LOG_DIR="${LOG_DIR:-logs}"
PID_FILE="$LOG_DIR/chatbot.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "[INFO] PID 파일이 없어 종료할 서버가 없습니다."
  exit 0
fi

SERVER_PID="$(cat "$PID_FILE")"
if kill -0 "$SERVER_PID" 2>/dev/null; then
  kill "$SERVER_PID"
  for _ in 1 2 3 4 5; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done

  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi

  echo "[INFO] 서버 종료 완료. PID=$SERVER_PID"
else
  echo "[INFO] 이미 종료된 PID입니다. PID=$SERVER_PID"
fi

rm -f "$PID_FILE"
