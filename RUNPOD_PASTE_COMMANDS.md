# Runpod 복붙 명령어

아래 순서는 `2026-03-27` 기준 `Runpod A40 / Container 20GB / Volume 60GB` fresh Pod에서 실제로 재현 성공한 흐름이다.

## 1. 최초 세팅

```bash
cd /workspace
git clone https://github.com/Chaemok/WoW-AI.git
cd WoW-AI
git checkout chaemok
cp .env.example .env
sed -i 's#^MODEL_ID=.*#MODEL_ID=Qwen/Qwen2.5-14B-Instruct#' .env
sed -i 's#^LOAD_IN_4BIT=.*#LOAD_IN_4BIT=0#' .env
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
chmod +x start_chatbot_background_linux.sh stop_chatbot_linux.sh status_chatbot_linux.sh
mkdir -p /workspace/WoW-AI/.cache/tmp
export APP_CACHE_DIR=/workspace/WoW-AI/.cache
export XDG_CACHE_HOME=$APP_CACHE_DIR
export HF_HOME=$APP_CACHE_DIR/huggingface
export TRANSFORMERS_CACHE=$APP_CACHE_DIR/huggingface
export TMPDIR=$APP_CACHE_DIR/tmp
export TMP=$APP_CACHE_DIR/tmp
export TEMP=$APP_CACHE_DIR/tmp
export HF_HUB_DISABLE_XET=1
hf download Qwen/Qwen2.5-14B-Instruct --cache-dir /workspace/WoW-AI/.cache/huggingface
./start_chatbot_background_linux.sh
./status_chatbot_linux.sh
sleep 100
./status_chatbot_linux.sh
curl http://127.0.0.1:8000/health
python smoke_test_api.py --server-url http://127.0.0.1:8000
```

메모:

- background 스크립트는 local snapshot이 있으면 자동으로 그 경로를 사용한다.
- 시작 직후에는 `health`가 바로 안 열릴 수 있다.
- 실제 검증에서는 `sleep 100` 뒤 `status -> /health -> smoke_test_api.py` 순서가 안정적이었다.

## 2. 로그 보기

```bash
tail -f logs/chatbot.out.log
```

```bash
tail -f logs/chatbot.err.log
```

## 3. 상태 확인

```bash
./status_chatbot_linux.sh
curl http://127.0.0.1:8000/health
python smoke_test_api.py --server-url http://127.0.0.1:8000
```

## 4. 코드 수정 후 재배포

```bash
cd /workspace/WoW-AI
git pull origin chaemok
source .venv/bin/activate
./stop_chatbot_linux.sh
./start_chatbot_background_linux.sh
./status_chatbot_linux.sh
sleep 100
./status_chatbot_linux.sh
curl http://127.0.0.1:8000/health
```

## 5. 서버 중지

```bash
./stop_chatbot_linux.sh
```
