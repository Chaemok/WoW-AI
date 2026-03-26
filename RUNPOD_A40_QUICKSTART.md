# Runpod A40 Quickstart

`chatbot.py`를 Runpod A40에서 올리고 `:8000` 내부 AI API를 확인하는 가장 짧은 가이드다.  
아래 내용은 `2026-03-27` 기준 fresh Pod에서 재현 성공한 결과를 반영했다.

## 권장 기준

- GPU: `A40 48GB`
- 스토리지: `Container 20GB / Volume 60GB`
- 포트: `8000`
- 모델: `Qwen/Qwen2.5-14B-Instruct`
- 설정: `LOAD_IN_4BIT=0`
- 실행 방식: `/workspace` 캐시 + local snapshot + background script

## 1. Pod 준비

Runpod에서 아래 기준으로 Pod를 만든다.

- GPU: `A40 48GB`
- OS: `Ubuntu`
- Container Disk: `20GB`
- Volume Disk: `60GB`
- Open Port: `8000`

## 2. 레포 clone

```bash
cd /workspace
git clone https://github.com/Chaemok/WoW-AI.git
cd WoW-AI
git checkout chaemok
```

## 3. 서버 세팅

```bash
cp .env.example .env
sed -i 's#^MODEL_ID=.*#MODEL_ID=Qwen/Qwen2.5-14B-Instruct#' .env
sed -i 's#^LOAD_IN_4BIT=.*#LOAD_IN_4BIT=0#' .env
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
chmod +x start_chatbot_background_linux.sh stop_chatbot_linux.sh status_chatbot_linux.sh
```

중요:

- fresh Pod 재현 테스트에서 `requirements-server.txt`는 `transformers 4.57.6`으로 정상 설치됐다.
- Runpod에서는 `.venv --system-site-packages`를 쓰는 쪽이 안정적이었다.

## 4. 캐시 경로 고정과 모델 다운로드

```bash
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
```

핵심:

- Hugging Face 캐시와 임시 디렉터리를 모두 `/workspace` 아래로 고정한다.
- background 스크립트는 local snapshot이 있으면 자동으로 snapshot 경로를 `MODEL_ID`로 선택한다.

## 5. 백그라운드 실행

```bash
./start_chatbot_background_linux.sh
./status_chatbot_linux.sh
sleep 100
./status_chatbot_linux.sh
tail -n 50 logs/chatbot.out.log
tail -n 50 logs/chatbot.err.log
curl http://127.0.0.1:8000/health
python smoke_test_api.py --server-url http://127.0.0.1:8000
```

메모:

- 14B는 콜드 스타트가 길어서 시작 직후 `health`가 실패할 수 있다.
- 실제 fresh Pod 재현에서는 `sleep 100` 뒤 `status`, `/health`, `smoke_test_api.py` 확인이 안정적으로 성공했다.
- 최종적으로 background 스크립트, `/health`, `smoke_test_api.py`까지 모두 성공했다.

## 6. 코드 수정 후 반영

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

## 7. 확인 포인트

정상이라면 아래가 확인된다.

- `./status_chatbot_linux.sh`가 `health 응답 확인됨` 출력
- `curl http://127.0.0.1:8000/health`가 `status: ok` 반환
- `python smoke_test_api.py --server-url http://127.0.0.1:8000`가 `[OK]` 반환

## 8. 한 줄 요약

- Runpod A40 fresh Pod `20GB / 60GB`에서 14B 재현 성공
- 권장 흐름은 `hf download -> background start -> sleep 100 -> health/smoke`
