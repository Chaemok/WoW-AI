# Runpod 복붙용 명령어

아래 명령어는 `Runpod A40 Pod`에 접속한 직후 바로 붙여넣기 위한 최소 세트다.

## 1. 최초 실행

```bash
cd /workspace
git clone https://github.com/Chaemok/WoW-AI.git
cd WoW-AI
git checkout chaemok
cp .env.example .env
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
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
sed -i 's#^MODEL_ID=.*#MODEL_ID=Qwen/Qwen2.5-14B-Instruct#' .env
sed -i 's#^LOAD_IN_4BIT=.*#LOAD_IN_4BIT=0#' .env
./start_chatbot_background_linux.sh
./status_chatbot_linux.sh
curl http://127.0.0.1:8000/health
```

참고:

- background 스크립트는 local snapshot이 있으면 자동으로 snapshot 경로를 사용한다
- health가 바로 안 열리면 모델 로딩 중일 수 있으니 `1~2분` 정도 기다린 뒤 다시 확인한다

## 2. 로그 확인

```bash
tail -f logs/chatbot.out.log
```

오류 로그:

```bash
tail -f logs/chatbot.err.log
```

## 3. 서버 상태 확인

```bash
curl http://127.0.0.1:8000/health
python smoke_test_api.py --server-url http://127.0.0.1:8000
```

## 4. 코드 수정 반영

```bash
cd /workspace/WoW-AI
git pull origin chaemok
source .venv/bin/activate
./stop_chatbot_linux.sh
./start_chatbot_background_linux.sh
./status_chatbot_linux.sh
```

## 5. 서버 중지

```bash
./stop_chatbot_linux.sh
```
