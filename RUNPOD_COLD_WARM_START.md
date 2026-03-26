# Runpod 콜드 스타트와 웜 스타트

`Runpod A40 + Qwen2.5-14B-Instruct` 기준으로 서버 기동 시간과 실제 응답 시간을 구분해서 정리한 문서다.

## 1. 개념

### 콜드 스타트

- 서버 프로세스를 새로 띄우는 순간부터 모델 로딩이 끝날 때까지의 시간
- 이 시간 동안은 `GET /health`가 실패할 수 있다
- 모델 shard 읽기, `safetensors` 로딩, GPU 메모리 적재가 모두 포함된다

### 웜 스타트

- 서버와 모델이 이미 메모리에 올라온 상태
- 이 상태의 `GET /health`는 거의 즉시 응답한다

## 2. 이번 검증 기준

- GPU: `NVIDIA A40 48GB`
- 모델: `Qwen/Qwen2.5-14B-Instruct`
- Pod 스토리지: `Container 20GB / Volume 60GB`
- 실행 방식:
  - `/workspace/WoW-AI/.cache/huggingface`에 `hf download`
  - local snapshot 자동 선택
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
  - background script 사용

## 3. 실제 확인한 시간

- fresh Pod 기준 콜드 스타트는 대략 `1분 30초 ~ 2분`
- 시작 직후 `curl http://127.0.0.1:8000/health`는 실패할 수 있다
- 실제 재현에서는 `sleep 100` 뒤 health와 smoke test가 안정적으로 성공했다

즉, 처음 1~2분은 서버가 느린 것이 아니라 모델이 아직 올라오는 중이라고 보면 된다.

## 4. 실전 대기 순서

background 실행 직후에는 아래 순서를 권장한다.

```bash
./status_chatbot_linux.sh
sleep 100
./status_chatbot_linux.sh
curl http://127.0.0.1:8000/health
python smoke_test_api.py --server-url http://127.0.0.1:8000
```

정상이라면:

- `logs/chatbot.out.log`에 `모델 로딩 완료`
- `logs/chatbot.err.log`에 Flask bind 로그
- `./status_chatbot_linux.sh`에 `health 응답 확인됨`

## 5. 응답속도 줄이는 방법

### 1. 서버를 계속 띄워둔다

- 가장 큰 최적화는 재시작을 줄이는 것이다
- 콜드 스타트는 길지만, 웜 상태에서는 health가 즉시 응답한다

### 2. 모델을 미리 받아두고 local snapshot으로 실행한다

- `hf download`로 먼저 `/workspace`에 모델을 받는다
- 실행 시 Hugging Face repo id 대신 local snapshot 경로를 사용하거나, background 스크립트의 자동 선택 기능을 쓴다
- offline 모드로 실행하면 재다운로드 리스크를 줄일 수 있다

### 3. 캐시와 TMPDIR를 `/workspace`로 고정한다

```bash
export APP_CACHE_DIR=/workspace/WoW-AI/.cache
export XDG_CACHE_HOME=$APP_CACHE_DIR
export HF_HOME=$APP_CACHE_DIR/huggingface
export TRANSFORMERS_CACHE=$APP_CACHE_DIR/huggingface
export TMPDIR=$APP_CACHE_DIR/tmp
export TMP=$APP_CACHE_DIR/tmp
export TEMP=$APP_CACHE_DIR/tmp
```

- 이렇게 해야 루트 디스크 부족 문제를 피하기 쉽다

### 4. 재시작 직후 웜업 확인을 한다

- 배포 직후 바로 사용자 요청을 받기보다
- `sleep 100 -> /health -> smoke_test_api.py`
- 순서로 먼저 준비 상태를 확인하는 게 안전하다

### 5. SLA가 아주 빡빡하면 더 작은 모델도 고려한다

- 14B는 품질은 좋지만 무겁다
- 응답속도와 운영 단순성이 더 중요하면 7B나 양자화 옵션을 다시 검토할 수 있다

## 6. 확인 명령

```bash
./status_chatbot_linux.sh
curl http://127.0.0.1:8000/health
python smoke_test_api.py --server-url http://127.0.0.1:8000
nvidia-smi
```

## 7. 한 줄 요약

- `14B`에서 `1분 30초 ~ 2분` 정도의 콜드 스타트는 정상 범위다
- fresh Pod `20GB / 60GB`에서도 `sleep 100` 대기 후 health와 smoke test까지 성공했다
- 가장 큰 최적화는 `재시작을 줄이고`, `local snapshot + /workspace 캐시`로 운영하는 것이다
