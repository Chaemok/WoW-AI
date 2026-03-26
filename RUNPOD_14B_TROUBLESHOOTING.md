# Runpod 14B 트러블슈팅

이 문서는 `Runpod A40 + Qwen2.5-14B-Instruct` 배포 중 실제로 만난 문제와 해결 방법을 정리한 문서다.

## 1. Private 레포 clone 실패

증상:

```text
remote: Invalid username or token. Password authentication is not supported for Git operations.
fatal: Authentication failed
```

원인:

- Runpod Pod에서 private GitHub 레포를 바로 clone하려고 했다.

해결:

- 가장 빠른 방법은 레포를 잠깐 `public`으로 바꿔 clone 한 뒤 다시 `private`로 돌리는 것이다.
- 더 안전하게 가려면 GitHub PAT 또는 deploy key를 사용한다.

## 2. `No space left on device`

증상:

```text
OSError: [Errno 28] No space left on device
```

원인:

- `/workspace`가 아니라 루트(`/`)에서 작업했다.
- Hugging Face 캐시가 `/root/.cache/huggingface`로 쌓였다.
- 임시 다운로드 디렉터리도 루트 디스크를 사용했다.

해결:

- 레포는 반드시 `/workspace/WoW-AI`에 둔다.
- 아래 환경변수로 캐시와 임시 디렉터리를 `/workspace`로 고정한다.

```bash
export APP_CACHE_DIR=/workspace/WoW-AI/.cache
export XDG_CACHE_HOME=$APP_CACHE_DIR
export HF_HOME=$APP_CACHE_DIR/huggingface
export TRANSFORMERS_CACHE=$APP_CACHE_DIR/huggingface
export TMPDIR=$APP_CACHE_DIR/tmp
export TMP=$APP_CACHE_DIR/tmp
export TEMP=$APP_CACHE_DIR/tmp
```

추가 정리 명령:

```bash
rm -rf /root/.cache/huggingface
find /workspace/WoW-AI/.cache/huggingface -name "*.incomplete" -delete
find /workspace/WoW-AI/.cache/huggingface -name "*.lock" -delete
```

## 3. 4bit 로딩 실패

증상:

```text
AttributeError: 'Qwen2ForCausalLM' object has no attribute 'set_submodule'
```

원인:

- `transformers 5.x`와 Runpod 기본 `torch 2.4.x` 조합에서 `bitsandbytes 4bit` 경로가 깨졌다.

해결:

- `requirements-server.txt`에서 `transformers`를 `4.x`로 고정한다.
- 14B는 우선 `LOAD_IN_4BIT=0`으로 띄운다.

## 4. 다운로드는 끝났는데 다시 허브에서 재다운로드

증상:

- `hf download`로 이미 모델을 받았는데 `python chatbot.py` 실행 시 다시 `Fetching 8 files`가 뜬다.

원인:

- `MODEL_ID`를 Hugging Face repo id로 두고 실행했다.

해결:

- 다운로드가 끝난 뒤 snapshot 경로를 직접 `MODEL_ID`로 사용한다.

```bash
SNAPSHOT_DIR="$(find /workspace/WoW-AI/.cache/huggingface/models--Qwen--Qwen2.5-14B-Instruct/snapshots -mindepth 1 -maxdepth 1 -type d | head -n 1)"
export MODEL_ID="$SNAPSHOT_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 5. `incomplete metadata, file not fully covered`

증상:

```text
safetensors_rust.SafetensorError: Error while deserializing header: incomplete metadata, file not fully covered
```

원인:

- 14B shard 중 일부가 손상된 상태로 캐시에 남았다.

해결:

- 손상된 모델 캐시를 지우고 다시 다운로드한다.

```bash
rm -rf /workspace/WoW-AI/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct
find /workspace/WoW-AI/.cache/huggingface -name "*.incomplete" -delete
find /workspace/WoW-AI/.cache/huggingface -name "*.lock" -delete
hf download Qwen/Qwen2.5-14B-Instruct --cache-dir /workspace/WoW-AI/.cache/huggingface
```

## 6. `.env`를 14B로 바꿨는데 7B로 뜸

증상:

```text
[INFO] MODEL_ID=Qwen/Qwen2.5-7B-Instruct
```

원인:

- 셸에 남아 있던 `MODEL_ID`, `LOAD_IN_4BIT` 값이 `.env`보다 우선 적용됐다.

해결:

- 실행 전에 셸 변수를 비우거나 원하는 값으로 다시 export 한다.

```bash
unset MODEL_ID LOAD_IN_4BIT
export MODEL_ID=Qwen/Qwen2.5-14B-Instruct
export LOAD_IN_4BIT=0
```

- 시작 스크립트는 `.env`를 먼저 읽고, 호출 시 넘긴 환경변수를 다시 우선 적용하도록 수정했다.

## 7. Runpod에서 추천하는 안정 실행 순서

```bash
cd /workspace/WoW-AI
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
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
SNAPSHOT_DIR="$(find /workspace/WoW-AI/.cache/huggingface/models--Qwen--Qwen2.5-14B-Instruct/snapshots -mindepth 1 -maxdepth 1 -type d | head -n 1)"
export MODEL_ID="$SNAPSHOT_DIR"
export LOAD_IN_4BIT=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python chatbot.py --host 0.0.0.0 --port 8000
```
