#!/usr/bin/env bash
set -euo pipefail

# 스크립트 위치를 기준으로 프로젝트 루트로 이동한다.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
export LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
export ANALYZE_MAX_NEW_TOKENS="${ANALYZE_MAX_NEW_TOKENS:-256}"
export CHAT_MAX_NEW_TOKENS="${CHAT_MAX_NEW_TOKENS:-160}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "[INFO] 가상환경이 없어 새로 생성합니다."
  python3 -m venv "$VENV_DIR"
fi

# 가상환경을 활성화하고 필요한 패키지를 설치한다.
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[INFO] AI 서버를 시작합니다."
echo "[INFO] HOST=$HOST PORT=$PORT"
echo "[INFO] MODEL_ID=$MODEL_ID"
echo "[INFO] LOAD_IN_4BIT=$LOAD_IN_4BIT"

exec python chatbot.py --host "$HOST" --port "$PORT"
