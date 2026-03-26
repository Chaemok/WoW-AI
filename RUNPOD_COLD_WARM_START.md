# Runpod 콜드 스타트 / 웜 스타트 정리

이 문서는 `Runpod A40 + Qwen2.5-14B-Instruct` 기준으로 실제 실행하면서 확인한 체감 시간과, 응답 속도를 줄이기 위한 운영 팁을 정리한 문서다.

## 1. 먼저 구분할 것

### 콜드 스타트

- Pod를 새로 올렸거나
- Python 프로세스를 다시 시작했거나
- 모델을 GPU 메모리에 다시 올려야 하는 상태

즉, `chatbot.py`가 처음 뜨면서 모델을 읽고 GPU에 적재하는 구간이다.

### 웜 스타트

- Python 프로세스가 이미 살아 있고
- 모델 로딩도 끝났고
- `GET /health`가 바로 응답하는 상태

즉, 서버와 모델이 이미 준비된 상태다.

## 2. 이번 세션에서 실제로 확인한 시간

기준 환경:

- GPU: `NVIDIA A40 48GB`
- 모델: `Qwen2.5-14B-Instruct`
- 실행 방식:
  - Hugging Face 모델을 `/workspace/WoW-AI/.cache/huggingface`에 미리 다운로드
  - `MODEL_ID`에 Hugging Face repo id 대신 local snapshot 경로 사용
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
  - `LOAD_IN_4BIT=0`

### 콜드 스타트 시간

- 실제 체감 기준으로 약 `1분 30초 ~ 2분`
- 이 시간 동안은 `curl http://127.0.0.1:8000/health`가 실패할 수 있다
- 원인:
  - 14B 가중치 파일 읽기
  - `safetensors` 로드
  - GPU 메모리 적재
  - Flask 서버 bind

### 웜 스타트 시간

- 모델 로딩이 끝난 뒤 `GET /health`는 거의 즉시 응답
- 이번 세션에서도 모델 로딩 완료 후에는 바로 `status: ok`를 반환했다

### 주의할 점

- 방금 체감한 `2분 대기`는 보통 `요청 응답시간`이 아니라 `서버 기동 시간`이다
- 실제 운영에서 중요한 것은:
  - 프로세스를 자주 재시작하지 않는 것
  - 모델을 다시 받거나 다시 올리는 상황을 피하는 것

## 3. 왜 콜드 스타트가 길어지나

14B 모델에서는 아래가 누적된다.

- 대용량 가중치 파일을 디스크에서 읽음
- 여러 shard를 병합해 모델 구조에 적재
- GPU 메모리로 이동
- 첫 실행 시 PyTorch / CUDA 관련 초기화가 일어남

즉, `14B`는 한 번 뜰 때 무겁고, 뜬 뒤에는 훨씬 안정적으로 응답하는 구조라고 보면 된다.

## 4. 응답 속도 줄이는 방법

### 1. 서버를 가능한 한 계속 띄워둔다

가장 효과가 크다.

- 프로세스를 죽이지 않으면 콜드 스타트를 다시 기다리지 않아도 된다
- 운영 시에는 백그라운드로 유지하는 것이 좋다

### 2. 모델은 미리 받아두고 local snapshot 경로로 실행한다

이번 세션에서 가장 안정적이었던 방식이다.

- `hf download`로 먼저 모델 다운로드
- `MODEL_ID=Qwen/...` 대신 local snapshot 경로 사용
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

이렇게 하면:

- 기동 시 허브 재다운로드를 피할 수 있고
- root overlay 디스크가 다시 차는 문제도 줄일 수 있다

### 3. 캐시와 임시 디렉터리를 모두 `/workspace`로 고정한다

아래 둘 다 중요하다.

- 모델 캐시
- 임시 다운로드 디렉터리

추천 환경변수:

```bash
export APP_CACHE_DIR=/workspace/WoW-AI/.cache
export XDG_CACHE_HOME=$APP_CACHE_DIR
export HF_HOME=$APP_CACHE_DIR/huggingface
export TRANSFORMERS_CACHE=$APP_CACHE_DIR/huggingface
export TMPDIR=$APP_CACHE_DIR/tmp
export TMP=$APP_CACHE_DIR/tmp
export TEMP=$APP_CACHE_DIR/tmp
```

이렇게 해야 `/root/.cache`나 `/tmp` 때문에 root 디스크가 차는 문제를 줄일 수 있다.

### 4. Pod를 살려둔 상태에서 코드만 pull 하고 재시작한다

운영 흐름은 아래가 낫다.

1. 로컬에서 수정 후 push
2. Runpod에서 `git pull`
3. AI 서버만 재시작

Pod를 새로 만들거나 캐시를 날리는 것보다 훨씬 빠르다.

### 5. 배포 직후 웜업 요청을 한 번 보낸다

백엔드 연결 전에 아래를 먼저 해두면 좋다.

- `GET /health`
- 간단한 `POST /api/analyze`

이렇게 하면 첫 사용자 요청이 콜드 상태를 직접 맞는 일을 줄일 수 있다.

### 6. 응답시간 SLA가 빡빡하면 7B도 고려한다

14B는 품질은 좋지만 무겁다.

- 기동 시간
- 메모리 사용량
- 운영 복잡도

가 모두 늘어난다.

만약 아주 짧은 응답시간이 더 중요하면:

- `7B`
- 또는 호환성 검증이 끝난 뒤 `4bit`

를 고려할 수 있다.

### 7. 이번 세션 기준으로 4bit는 바로 운영에 쓰지 않는다

이번 세션에서는 `bitsandbytes 4bit` 경로에서 호환성 문제가 있었다.

- `transformers 5.x`
- Runpod 기본 `torch 2.4.x`
- `bitsandbytes`

조합에서 `set_submodule` 에러가 발생했다.

즉, 속도나 메모리 최적화를 위해 4bit를 다시 시도하려면:

- `transformers 4.x`
- Runpod 템플릿 torch 버전
- bitsandbytes

조합을 다시 검증한 뒤 적용하는 편이 안전하다.

## 5. 운영 추천 흐름

### 최초 1회

1. `/workspace/WoW-AI`에 clone
2. `.venv --system-site-packages`
3. `requirements-server.txt` 설치
4. 모델을 `hf download`로 `/workspace`에 미리 받기
5. local snapshot 경로로 실행

### 이후 일반 운영

1. 서버는 가능하면 계속 켜둠
2. 코드 수정 후 `git pull`
3. 필요할 때만 재시작
4. 재시작 후 `health`와 간단한 analyze로 웜업

## 6. 확인 명령

### 서버 준비 여부

```bash
curl http://127.0.0.1:8000/health
```

### 간단한 기능 확인

```bash
python smoke_test_api.py --server-url http://127.0.0.1:8000
```

### GPU 메모리 확인

```bash
nvidia-smi
```

## 7. 한 줄 요약

- `14B`에서 `1분 30초 ~ 2분` 정도의 콜드 스타트는 꽤 정상 범위다
- 실제 요청 응답시간과 서버 기동 시간을 구분해서 봐야 한다
- 가장 큰 최적화는 `재시작을 줄이고`, `모델을 /workspace local snapshot으로 미리 받아두고`, `서버를 계속 띄워두는 것`이다
