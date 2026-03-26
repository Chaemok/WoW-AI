# Runpod 복붙용 명령어

아래 명령어는 `Runpod A40 Pod`에 접속한 직후 바로 붙여넣기 위한 최소 세트다.

## 1. 최초 실행

```bash
git clone https://github.com/Chaemok/WoW-AI.git
cd WoW-AI
git checkout chaemok
cp .env.example .env
chmod +x start_chatbot_background_linux.sh stop_chatbot_linux.sh status_chatbot_linux.sh
./start_chatbot_background_linux.sh
./status_chatbot_linux.sh
python smoke_test_api.py --server-url http://127.0.0.1:8000
```

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
./status_chatbot_linux.sh
curl http://127.0.0.1:8000/health
```

## 4. 서버 중지

```bash
./stop_chatbot_linux.sh
```

## 5. 모델 경로를 직접 지정할 때

```bash
export MODEL_ID=/workspace/models/Qwen2.5-7B-Instruct
./start_chatbot_background_linux.sh
```
