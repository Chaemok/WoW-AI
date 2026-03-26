#!/usr/bin/env bash
set -euo pipefail

# 스크립트 위치를 기준으로 프로젝트 루트로 이동한다.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-logs}"
PID_FILE="$LOG_DIR/chatbot.pid"
OUT_LOG="$LOG_DIR/chatbot.out.log"
ERR_LOG="$LOG_DIR/chatbot.err.log"

export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
export LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
export ANALYZE_MAX_NEW_TOKENS="${ANALYZE_MAX_NEW_TOKENS:-256}"
export CHAT_MAX_NEW_TOKENS="${CHAT_MAX_NEW_TOKENS:-160}"

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ]; then
  EXISTING_PID="$(cat "$PID_FILE")"
  if kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "[INFO] 이미 실행 중입니다. PID=$EXISTING_PID"
    echo "[INFO] 로그: $OUT_LOG"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "[INFO] 가상환경이 없어 새로 생성합니다."
  python3 -m venv "$VENV_DIR"
fi

# 가상환경을 활성화하고 필요한 패키지를 설치한다.
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[INFO] 백그라운드로 AI 서버를 시작합니다."
echo "[INFO] HOST=$HOST PORT=$PORT"
echo "[INFO] MODEL_ID=$MODEL_ID"
echo "[INFO] OUT_LOG=$OUT_LOG"
echo "[INFO] ERR_LOG=$ERR_LOG"

nohup python chatbot.py --host "$HOST" --port "$PORT" >"$OUT_LOG" 2>"$ERR_LOG" &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

echo "[INFO] 시작 완료. PID=$SERVER_PID"
