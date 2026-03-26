# Runpod A40 빠른 실행 가이드

이 문서는 `Runpod A40`에서 이 AI 레포를 빠르게 실행하기 위한 최소 절차를 정리한 문서다.  
목표는 `chatbot.py`를 `:8000`에서 띄우고, 백엔드가 내부적으로 `POST /api/analyze`를 호출할 수 있게 만드는 것이다.

## 권장 기준

- GPU: `A40 48GB`
- 운영 방식: `Pod`
- 권장 포트:
  - `8000`: 내부 AI API
  - `5000`: 선택, HTML 데모
- 기본 모델:
  - `Qwen/Qwen2.5-7B-Instruct`
- Runpod 실사용 권장:
  - `Qwen/Qwen2.5-14B-Instruct`
  - `LOAD_IN_4BIT=0`
- 14B 모델은 다운로드와 캐시를 포함해 대략 `30GB` 안팎 공간을 차지할 수 있다.
- Runpod에서는 반드시 `/workspace` 아래에서 작업한다.

## 1. Pod 준비

Runpod에서 아래 기준으로 Pod를 생성한다.

- GPU: `A40 48GB`
- OS: `Ubuntu` 계열 템플릿
- 외부 포트 오픈:
  - `8000`
  - 필요 시 `5000`
- 디스크는 모델 다운로드까지 고려해서 여유 있게 잡는다.

## 2. 레포 준비

Pod 터미널에 접속한 뒤 레포를 clone 한다.

```bash
cd /workspace
git clone https://github.com/Chaemok/WoW-AI.git
cd WoW-AI
git checkout chaemok
```

메인 브랜치를 쓸 경우:

```bash
git checkout main
```

## 3. 서버 실행

가장 안정적인 방법은 Runpod 기본 PyTorch를 재사용하는 가상환경을 만든 뒤, 14B를 먼저 로컬 캐시에 받아 두고 실행하는 것이다.

필요하면 먼저 `.env.example`을 복사해 `.env`를 만든다.

```bash
cp .env.example .env
```

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
```

캐시와 임시 디렉터리는 모두 `/workspace` 아래로 고정한다.

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
```

14B 모델을 먼저 캐시에 다운로드한다.

```bash
hf download Qwen/Qwen2.5-14B-Instruct --cache-dir /workspace/WoW-AI/.cache/huggingface
```

다운로드가 끝나면 snapshot 경로를 찾아 로컬 모델 경로로 실행한다.

```bash
SNAPSHOT_DIR="$(find /workspace/WoW-AI/.cache/huggingface/models--Qwen--Qwen2.5-14B-Instruct/snapshots -mindepth 1 -maxdepth 1 -type d | head -n 1)"
export MODEL_ID="$SNAPSHOT_DIR"
export LOAD_IN_4BIT=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python chatbot.py --host 0.0.0.0 --port 8000
```

백그라운드 실행이 필요하면 같은 환경변수를 유지한 상태에서 아래 스크립트를 사용한다.

```bash
chmod +x start_chatbot_background_linux.sh stop_chatbot_linux.sh status_chatbot_linux.sh
./start_chatbot_background_linux.sh
./status_chatbot_linux.sh
```

시작 스크립트는 아래 작업을 자동으로 처리한다.

- `.env` 우선 반영
- `requirements-server.txt` 우선 설치
- 캐시와 임시 디렉터리를 프로젝트 내부 `.cache`로 고정
- Runpod에서 `requirements-server.txt`를 쓰면 `.venv --system-site-packages`를 사용

백그라운드 실행 시 로그 위치:

- `logs/chatbot.out.log`
- `logs/chatbot.err.log`

## 4. 환경변수

필요하면 실행 전에 아래 값을 지정할 수 있다.

```bash
export APP_CACHE_DIR=/workspace/WoW-AI/.cache
export XDG_CACHE_HOME=$APP_CACHE_DIR
export HF_HOME=$APP_CACHE_DIR/huggingface
export TRANSFORMERS_CACHE=$APP_CACHE_DIR/huggingface
export TMPDIR=$APP_CACHE_DIR/tmp
export TMP=$APP_CACHE_DIR/tmp
export TEMP=$APP_CACHE_DIR/tmp
export HF_HUB_DISABLE_XET=1
```

모델을 Hugging Face 대신 로컬 snapshot 경로로 쓸 수도 있다.

```bash
SNAPSHOT_DIR="$(find /workspace/WoW-AI/.cache/huggingface/models--Qwen--Qwen2.5-14B-Instruct/snapshots -mindepth 1 -maxdepth 1 -type d | head -n 1)"
export MODEL_ID="$SNAPSHOT_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

14B를 쓸 경우 예시:

```bash
export LOAD_IN_4BIT=0
python chatbot.py --host 0.0.0.0 --port 8000
```

## 5. 서버 확인

### 상태 확인

```bash
curl http://127.0.0.1:8000/health
```

예상 응답 예시:

```json
{
  "status": "ok",
  "model": "/workspace/WoW-AI/.cache/huggingface/.../snapshots/<hash>",
  "device": "cuda",
  "gpu": "NVIDIA A40",
  "active_sessions": 0,
  "activeSessions": 0
}
```

### 분석 호출 확인

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"category": "커피/음료", "amount": 233500, "count": 12},
      {"category": "육류/회식", "amount": 193200, "count": 4},
      {"category": "분식", "amount": 167600, "count": 7}
    ],
    "keepSession": false
  }'
```

### 스모크 테스트 스크립트

배포 직후 한 번에 상태와 분석 호출을 확인하려면 아래 스크립트를 사용한다.

```bash
python smoke_test_api.py --server-url http://127.0.0.1:8000
```

이 스크립트는 아래 순서로 동작한다.

- `GET /health`
- `POST /api/analyze`
- 핵심 응답 요약 출력

## 6. 백엔드 연동 기준

Runpod에 올라간 AI 서버는 내부적으로 아래 엔드포인트를 제공한다.

- `GET /health`
- `POST /api/analyze`
- `POST /api/chat`
- `DELETE /api/session/<id>`

하지만 공개 API는 백엔드에서 감싼다.

- 공개 API:
  - `POST /api/v1/ai/analysis`
  - `GET /api/v1/ai/analysis`
- AI 내부 API:
  - `POST /api/analyze`

즉 앱이 Runpod AI 서버를 직접 호출하지 않고, `백엔드 -> AI 서버` 구조로 가는 것을 권장한다.

## 7. 운영 팁

- 월간 리포트 생성용 호출은 `keepSession: false`를 권장한다.
- 서버 시작 직후 첫 요청은 모델 로딩 때문에 느릴 수 있다.
- 원본 학습 데이터는 레포에 없으므로 현재 가능한 것은 `추론`이며, `재학습`은 불가능하다.
- `analysis/user_category_overrides.csv`는 데모용 파일 기반 구조이므로 운영 저장소로 쓰지 않는 편이 좋다.
- 코드 수정 반영 순서:
  - 로컬에서 수정 후 `git push origin chaemok`
  - Runpod에서 `git pull origin chaemok`
  - 서버 재시작
- 이번 배포에서 겪은 문제와 해결 방법은 [RUNPOD_14B_TROUBLESHOOTING.md](/Users/SSAFY/Desktop/data_analysis/RUNPOD_14B_TROUBLESHOOTING.md)에 정리한다.
