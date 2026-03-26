#!/usr/bin/env bash
set -euo pipefail

# 상태 확인도 프로젝트 루트를 기준으로 동작하도록 고정한다.
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
OUT_LOG="$LOG_DIR/chatbot.out.log"
ERR_LOG="$LOG_DIR/chatbot.err.log"
PORT="${PORT:-8000}"
HEALTH_HOST="${HEALTH_HOST:-127.0.0.1}"
HEALTH_URL="${HEALTH_URL:-http://$HEALTH_HOST:$PORT/health}"

if [ ! -f "$PID_FILE" ]; then
  echo "[INFO] 실행 중인 서버가 없습니다."
  exit 0
fi

SERVER_PID="$(cat "$PID_FILE")"
if kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "[INFO] 실행 중입니다. PID=$SERVER_PID"
  echo "[INFO] OUT_LOG=$OUT_LOG"
  echo "[INFO] ERR_LOG=$ERR_LOG"
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
      echo "[INFO] health 응답 확인됨: $HEALTH_URL"
    else
      echo "[INFO] 프로세스는 살아 있지만 health 응답 전입니다. 모델 로딩 중일 수 있습니다."
    fi
  fi
  exit 0
fi

echo "[INFO] PID 파일은 있지만 프로세스는 종료되어 있습니다. PID=$SERVER_PID"
exit 1
