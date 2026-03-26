#!/usr/bin/env bash
set -euo pipefail

# 현재 스크립트 위치를 기준으로 프로젝트 루트를 고정한다.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${ENV_FILE:-.env}"

CALLER_VENV_DIR="${VENV_DIR-}"
CALLER_HOST="${HOST-}"
CALLER_PORT="${PORT-}"
CALLER_LOG_DIR="${LOG_DIR-}"
CALLER_REQUIREMENTS_FILE="${REQUIREMENTS_FILE-}"
CALLER_APP_CACHE_DIR="${APP_CACHE_DIR-}"
CALLER_MODEL_ID="${MODEL_ID-}"
CALLER_LOAD_IN_4BIT="${LOAD_IN_4BIT-}"
CALLER_ANALYZE_MAX_NEW_TOKENS="${ANALYZE_MAX_NEW_TOKENS-}"
CALLER_CHAT_MAX_NEW_TOKENS="${CHAT_MAX_NEW_TOKENS-}"
CALLER_HF_HOME="${HF_HOME-}"
CALLER_TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE-}"
CALLER_XDG_CACHE_HOME="${XDG_CACHE_HOME-}"
CALLER_PIP_CACHE_DIR="${PIP_CACHE_DIR-}"
CALLER_PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR-}"
CALLER_TMPDIR="${TMPDIR-}"
CALLER_TMP="${TMP-}"
CALLER_TEMP="${TEMP-}"
CALLER_HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET-}"
CALLER_HF_HUB_OFFLINE="${HF_HUB_OFFLINE-}"
CALLER_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE-}"
CALLER_USE_SYSTEM_SITE_PACKAGES="${USE_SYSTEM_SITE_PACKAGES-}"
CALLER_USE_LOCAL_SNAPSHOT="${USE_LOCAL_SNAPSHOT-}"
CALLER_PYTHONUNBUFFERED="${PYTHONUNBUFFERED-}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

VENV_DIR="${CALLER_VENV_DIR:-${VENV_DIR:-.venv}}"
HOST="${CALLER_HOST:-${HOST:-0.0.0.0}}"
PORT="${CALLER_PORT:-${PORT:-8000}}"
LOG_DIR="${CALLER_LOG_DIR:-${LOG_DIR:-logs}}"
REQUIREMENTS_FILE="${CALLER_REQUIREMENTS_FILE:-${REQUIREMENTS_FILE:-requirements-server.txt}}"
APP_CACHE_DIR="${CALLER_APP_CACHE_DIR:-${APP_CACHE_DIR:-$ROOT_DIR/.cache}}"
MODEL_ID="${CALLER_MODEL_ID:-${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}}"
LOAD_IN_4BIT="${CALLER_LOAD_IN_4BIT:-${LOAD_IN_4BIT:-1}}"
ANALYZE_MAX_NEW_TOKENS="${CALLER_ANALYZE_MAX_NEW_TOKENS:-${ANALYZE_MAX_NEW_TOKENS:-256}}"
CHAT_MAX_NEW_TOKENS="${CALLER_CHAT_MAX_NEW_TOKENS:-${CHAT_MAX_NEW_TOKENS:-160}}"
HF_HOME="${CALLER_HF_HOME:-${HF_HOME:-$APP_CACHE_DIR/huggingface}}"
TRANSFORMERS_CACHE="${CALLER_TRANSFORMERS_CACHE:-${TRANSFORMERS_CACHE:-$APP_CACHE_DIR/huggingface}}"
XDG_CACHE_HOME="${CALLER_XDG_CACHE_HOME:-${XDG_CACHE_HOME:-$APP_CACHE_DIR}}"
PIP_CACHE_DIR="${CALLER_PIP_CACHE_DIR:-${PIP_CACHE_DIR:-$APP_CACHE_DIR/pip}}"
PIP_NO_CACHE_DIR="${CALLER_PIP_NO_CACHE_DIR:-${PIP_NO_CACHE_DIR:-1}}"
TMPDIR="${CALLER_TMPDIR:-${TMPDIR:-$APP_CACHE_DIR/tmp}}"
TMP="${CALLER_TMP:-${TMP:-$APP_CACHE_DIR/tmp}}"
TEMP="${CALLER_TEMP:-${TEMP:-$APP_CACHE_DIR/tmp}}"
HF_HUB_DISABLE_XET="${CALLER_HF_HUB_DISABLE_XET:-${HF_HUB_DISABLE_XET:-1}}"
HF_HUB_OFFLINE="${CALLER_HF_HUB_OFFLINE:-${HF_HUB_OFFLINE:-0}}"
TRANSFORMERS_OFFLINE="${CALLER_TRANSFORMERS_OFFLINE:-${TRANSFORMERS_OFFLINE:-0}}"
USE_SYSTEM_SITE_PACKAGES="${CALLER_USE_SYSTEM_SITE_PACKAGES:-${USE_SYSTEM_SITE_PACKAGES:-}}"
USE_LOCAL_SNAPSHOT="${CALLER_USE_LOCAL_SNAPSHOT:-${USE_LOCAL_SNAPSHOT:-1}}"
PYTHONUNBUFFERED="${CALLER_PYTHONUNBUFFERED:-${PYTHONUNBUFFERED:-1}}"

export HOST PORT MODEL_ID LOAD_IN_4BIT ANALYZE_MAX_NEW_TOKENS CHAT_MAX_NEW_TOKENS
export HF_HOME TRANSFORMERS_CACHE XDG_CACHE_HOME PIP_CACHE_DIR PIP_NO_CACHE_DIR
export TMPDIR TMP TEMP HF_HUB_DISABLE_XET HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export USE_LOCAL_SNAPSHOT PYTHONUNBUFFERED

PID_FILE="$LOG_DIR/chatbot.pid"
OUT_LOG="$LOG_DIR/chatbot.out.log"
ERR_LOG="$LOG_DIR/chatbot.err.log"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  REQUIREMENTS_FILE="requirements.txt"
fi

if [ -z "$USE_SYSTEM_SITE_PACKAGES" ] && [ "$REQUIREMENTS_FILE" = "requirements-server.txt" ]; then
  USE_SYSTEM_SITE_PACKAGES=1
fi

mkdir -p "$LOG_DIR" "$APP_CACHE_DIR" "$TMPDIR"

resolve_snapshot_dir() {
  local repo_id="$1"
  local repo_cache_dir="$HF_HOME/models--${repo_id//\//--}"
  local snapshots_dir="$repo_cache_dir/snapshots"
  local latest_snapshot=""

  if [ ! -d "$snapshots_dir" ]; then
    return 1
  fi

  latest_snapshot="$(
    find "$snapshots_dir" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )"

  if [ -z "$latest_snapshot" ] || [ ! -d "$latest_snapshot" ]; then
    return 1
  fi

  printf '%s\n' "$latest_snapshot"
}

# Runpod에서는 미리 받은 snapshot을 바로 쓰는 편이 더 안정적이다.
if [ "$USE_LOCAL_SNAPSHOT" = "1" ] && [ ! -d "$MODEL_ID" ] && [[ "$MODEL_ID" == */* ]]; then
  if SNAPSHOT_DIR="$(resolve_snapshot_dir "$MODEL_ID")"; then
    MODEL_ID="$SNAPSHOT_DIR"
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    export MODEL_ID HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  fi
fi

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
  if [ "$USE_SYSTEM_SITE_PACKAGES" = "1" ]; then
    python3 -m venv "$VENV_DIR" --system-site-packages
  else
    python3 -m venv "$VENV_DIR"
  fi
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$REQUIREMENTS_FILE"

echo "[INFO] 백그라운드로 AI 서버를 시작합니다."
echo "[INFO] HOST=$HOST PORT=$PORT"
echo "[INFO] MODEL_ID=$MODEL_ID"
echo "[INFO] LOAD_IN_4BIT=$LOAD_IN_4BIT"
echo "[INFO] REQUIREMENTS_FILE=$REQUIREMENTS_FILE"
echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] TMPDIR=$TMPDIR"
echo "[INFO] HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "[INFO] TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "[INFO] USE_LOCAL_SNAPSHOT=$USE_LOCAL_SNAPSHOT"
echo "[INFO] OUT_LOG=$OUT_LOG"
echo "[INFO] ERR_LOG=$ERR_LOG"

nohup python -u chatbot.py --host "$HOST" --port "$PORT" >"$OUT_LOG" 2>"$ERR_LOG" &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

sleep 2
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "[ERROR] 백그라운드 프로세스가 바로 종료되었습니다."
  echo "[ERROR] 최근 OUT_LOG:"
  tail -n 40 "$OUT_LOG" 2>/dev/null || true
  echo "[ERROR] 최근 ERR_LOG:"
  tail -n 40 "$ERR_LOG" 2>/dev/null || true
  rm -f "$PID_FILE"
  exit 1
fi

echo "[INFO] 시작 완료. PID=$SERVER_PID"
echo "[INFO] 14B는 모델 로딩까지 1~2분 정도 걸릴 수 있습니다."
echo "[INFO] 준비 상태 확인: ./status_chatbot_linux.sh"
