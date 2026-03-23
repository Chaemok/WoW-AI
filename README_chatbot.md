# 소비 유형 분석 챗봇 서버

Qwen2.5-14B-Instruct (GPU)와 GMM 클러스터 모델을 결합한 개인 소비 분석 챗봇 API 서버.  
외부 클라이언트(로컬 노트북, 앱 등)에서 HTTP로 호출하여 사용한다.

---

## 구성

```
data_analysis/          ← 이 레포
├── chatbot.py          # FastAPI 서버 (GPU 서버에서 실행)
├── client_example.py   # 로컬 클라이언트 예제
├── README_chatbot.md   # 이 문서
├── model/
│   └── gmm_model.pkl
├── gmm_predict.py
├── spending_advisor.py
└── analysis/
    └── dummy_transactions.csv
```

---

## 서버 환경

| 항목 | 내용 |
|---|---|
| GPU | NVIDIA A40 (44GB VRAM) |
| 모델 | `Qwen/Qwen2.5-14B-Instruct` (float16, ~28GB) |
| 프레임워크 | FastAPI + uvicorn |
| 서버 IP | `172.16.64.2` |
| 기본 포트 | `8000` |

---

## 서버 실행

```bash
git clone https://github.com/sumin-990416/data_analysis.git
cd data_analysis
pip install -r requirements.txt
pip install fastapi uvicorn transformers accelerate sentencepiece

python3 chatbot.py                 # 기본: 0.0.0.0:8000
python3 chatbot.py --port 9000     # 포트 변경
```

---

## API 엔드포인트

### `GET /health`
서버 및 GPU 상태 확인.

```bash
curl http://172.16.64.2:8000/health
```

응답:
```json
{
  "status": "ok",
  "model": "Qwen/Qwen2.5-14B-Instruct",
  "device": "cuda",
  "gpu": "NVIDIA A40",
  "active_sessions": 0
}
```

---

### `POST /api/analyze`
거래 내역 CSV를 분석하여 소비 유형 클러스터 분류 + AI 피드백 생성.  
응답에 포함된 `session_id`로 이후 `/api/chat` 에서 대화를 이어갈 수 있다.

**요청**
```json
{
  "csv_text": "card_tpbuz_nm_2,amt,cnt\n외식,85000,6\n커피/음료,42000,14",
  "demo": false,
  "session_id": null
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `csv_text` | string | CSV 문자열 (헤더 포함). `demo=true`면 생략 가능 |
| `demo` | bool | `true`면 내장 더미 데이터로 분석 |
| `session_id` | string | 기존 세션 이어가기. 없으면 신규 생성 |

CSV 필수 컬럼:

| 컬럼명 | 의미 |
|---|---|
| `card_tpbuz_nm_2` | 업종 소분류 (예: 외식, 커피/음료) |
| `amt` | 결제 금액 (원) |
| `cnt` | 거래 건수 |

**응답**
```json
{
  "session_id": "9bed598b-...",
  "cluster_id": 1,
  "cluster_name": "사무·서적형",
  "feedback": "### 1. 소비 패턴 진단\n...",
  "cluster_stats": [
    {"category": "커피/음료", "user_pct": 23.3, "cluster_pct": 8.9, "diff_pct": 14.4, "user_amt": 42000}
  ],
  "reduction_summary": {"커피/음료": 70050, "육류/회식": 57960},
  "total_reduction": 127010
}
```

---

### `POST /api/chat`
세션 기반 자유 대화. 분석 이후 맥락을 유지하며 추가 질문 가능.

**요청**
```json
{
  "message": "커피 외에 다른 절감 방법 알려줘",
  "session_id": "9bed598b-..."
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `message` | string | 사용자 메시지 |
| `session_id` | string | 이전 세션 ID. 없으면 새 세션 생성 |

**응답**
```json
{
  "session_id": "9bed598b-...",
  "reply": "외식비를 줄이기 위해서는..."
}
```

---

### `DELETE /api/session/{session_id}`
세션 삭제 (대화 기록 초기화).

```bash
curl -X DELETE http://172.16.64.2:8000/api/session/9bed598b-...
```

---

## 로컬 노트북에서 사용하기

### 1. 패키지 설치
```bash
pip install requests
```

### 2. `client_example.py` 사용
`client_example.py`를 로컬에 복사한 후 `SERVER_URL`을 수정한다.

```python
SERVER_URL = "http://172.16.64.2:8000"
```

실행:
```bash
python client_example.py
```

### 3. Jupyter Notebook에서 직접 사용
```python
import requests

SERVER = "http://172.16.64.2:8000"

# 더미 데이터 분석
r = requests.post(f"{SERVER}/api/analyze", json={"demo": True}, timeout=120)
result = r.json()
session_id = result["session_id"]

print(result["cluster_name"])   # 소비 유형
print(result["feedback"])       # AI 피드백
print(result["total_reduction"])# 예상 절감액

# 이어서 대화
r2 = requests.post(f"{SERVER}/api/chat",
    json={"message": "절감 방법 더 알려줘", "session_id": session_id},
    timeout=120)
print(r2.json()["reply"])
```

### 4. 내 CSV 파일로 분석
```python
with open("my_transactions.csv", encoding="utf-8") as f:
    csv_text = f.read()

r = requests.post(f"{SERVER}/api/analyze",
    json={"csv_text": csv_text}, timeout=120)
result = r.json()
```

---

## 서버 접근이 안 될 때 (방화벽/VPN)

SSH 터널링으로 우회:
```bash
# 로컬 터미널에서 실행
ssh -L 8000:localhost:8000 user@172.16.64.2

# 이후 로컬에서 아래 주소로 접근
SERVER_URL = "http://localhost:8000"
```
