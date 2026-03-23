# 소비 유형 분석 챗봇 서버

Qwen2.5-14B-Instruct (GPU)와 GMM 클러스터 모델을 결합한 개인 소비 분석 챗봇 API 서버.  
/root/workspace/data_analysis 클라이언트(로컬 노트북, 앱 등)에서 HTTP로 호출하여 사용한다.

---

## 전체 아키텍처

```
        (노트북 / 브라우저)
        │
        ├─ HTTP :5000 ──▶ run_demo.py (Flask 웹 데모 UI)
        │                       └─ /api/advise
        │                              └─ spending_advisor.analyze_and_advise()
        │                                        │
        └─ HTTP :8000 ──▶ chatbot.py (Flask API 서버, GPU)
                                ├─ /health
                                ├─ /api/analyze  ◀─── demo_app + 외부 클라이언트
                                ├─ /api/chat
                                └─ /api/session/<id> DELETE
                                        │
                                        ├─ GMM 클러스터 예측 (gmm_predict.py)
                                        └─ Qwen2.5-14B 추론 (GPU, ~28GB VRAM)
```

---

## 파일 구성

```
data_analysis/
 chatbot.py              # Flask API 서버 (GPU 서버에서 실행, :8000)
 client_example.py       # 로컬 클라이언트 예제
 README_chatbot.md       # 이 문서
 run_demo.py             # 웹 데모 서버 진입점 (:5000)
 spending_advisor.py     # GMM 분석 + chatbot 서버 연동 + 카테고리 매핑
 gmm_predict.py          # GMM 클러스터 예측
 gmm_train.py            # GMM 모델 학습 (최초 1회, 원본 데이터 필요)
 cluster_report.py       # 클러스터 리포트 생성
 cluster_definitions.py  # 클러스터 라벨/설명 정의
 model/
   └── gmm_model.pkl       # 학습된 모델 (gmm_train.py 산출물)
 analysis/
    ├── demo_app.py         # 웹 데모 Flask 앱 (:5000)
    ├── dummy_transactions.csv
    ├── merchant_category_map.csv   # 가맹점명 → 카테고리 매핑
    └── user_category_overrides.csv # 사용자 카테고리 수동 오버라이드
```

---

## 서버 환경

| 항목 | 내용 |
|---|---|
| GPU | NVIDIA A40 (44GB VRAM) |
| 모델 | `Qwen/Qwen2.5-14B-Instruct` (float16, ~28GB) |
| 프레임워크 | Flask |
| 서버 IP | `172.16.64.2` |
| chatbot 포트 | `8000` |
| 웹 데모 포트 | `5000` |

---

## 서버 실행

### 1. chatbot.py — API 서버 (필수)

```bash
cd data_analysis
pip install -r requirements.txt
pip install flask transformers accelerate sentencepiece

python3 chatbot.py                  # 기본: 0.0.0.0:8000
python3 chatbot.py --port 9000      # 포트 변경
```

.env.example .git .gitattributes .gitignore \=2.0 \=3.0 README.md README_chatbot.md __pycache__ analysis chatbot.py client_example.py cluster_definitions.py cluster_report.py data_analysis gmm_predict.py gmm_train.py model predict.py requirements.txt run_demo.py spending_advisor.py         첫 다운로드 시 약 30GB, 이후 캐시에서 로딩 (~15초).

 종료 후에도 유지하려면:

```bash
nohup python3 chatbot.py --host 0.0.0.0 --port 8000 > chatbot.log 2>&1 &
```

### 2. run_demo.py — 웹 데모 서버 (선택)

chatbot.py가 먼저 실행된 상태에서 별도 터미널로 실행:

```bash
cd data_analysis
python3 run_demo.py                 # 0.0.0.0:5000
```

`CHATBOT_URL` 환경변수로 chatbot 서버 주소를 지정할 수 있다 (기본값: `http://localhost:8000`):

```bash
CHATBOT_URL=http://172.16.64.2:8000 python3 run_demo.py
```

---

## 지원 소비 카테고리

DB `expense_category` 테이블과 일치하는 공식 카테고리 목록.  
`/api/analyze`, `/api/advise` 요청 시 `card_tpbuz_nm_2` 컬럼값으로 사용한다.

| 카테고리 | 카테고리 | 카테고리 | 카테고리 |
|---|---|---|---|
| 인터넷쇼핑 | 인테리어/가정용품 | 교통서비스 | 음/식료품소매 |
| 외식 | 제과/제빵/떡/케익 | 커피/음료 | 패스트푸드 |
| 자동차/유지비 | 시스템/통신 | 건강/기호식품 | 분식 |
| 육류/회식 | 선물/완구 | 병원/의료 | 화장품소매 |
| 공연관람 | 의약/의료품 | 건강/뷰티/마사지 | 수리서비스 |

> `제과/제빵/떡/케익`은 GMM 내부에서 `제과/제빵`으로 자동 변환됩니다.

---

## API 엔드포인트 (chatbot.py, :8000)

### `GET /health`
 및 GPU 상태 확인.

```bash
curl http://172.16.64.2:8000/health
```

:::::
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
 내역 CSV를 분석하여 소비 유형 클러스터 분류 + AI 피드백 생성.  
 포함된 `session_id`로 이후 `/api/chat`에서 대화를 이어갈 수 있다.

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
| `card_tpbuz_nm_2` | 위 카테고리 목록 중 하나 |
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
 기반 자유 대화. 분석 이후 맥락을 유지하며 추가 질문 가능.

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

### `DELETE /api/session/<session_id>`
 삭제 (대화 기록 초기화).

```bash
curl -X DELETE http://172.16.64.2:8000/api/session/9bed598b-...
```

---

## 웹 데모 엔드포인트 (demo_app.py, :5000)

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/` | 웹 UI |
| `POST` | `/api/predict` | GMM 클러스터 예측 |
| `POST` | `/api/advise` | chatbot 서버로 소비 분석 + AI 피드백 요청 |
| `POST` | `/api/preprocess-transactions` | 거래 CSV/Excel 파일 업로드 → 카테고리 집계 |
| `GET` | `/api/overrides` | 카테고리 오버라이드 목록 조회 |
| `POST` | `/api/overrides` | 카테고리 오버라이드 저장 |

`/api/advise` 요청:
```json
{
  "csv_text": "card_tpbuz_nm_2,amt,cnt\n외식,85000,6"
}
```

### CSV/Excel 업로드 흐름

```
 파일 업로드 (은행/카드 내역)
    │
    POST /api/preprocess-transactions  (multipart: bank_file, card_file)
    │    └─ 가맹점명 → 카테고리 자동 분류 (merchant_category_map.csv)
    │         └─ 미분류 가맹점: 키워드 규칙 적용
    │
    POST /api/predict                  (집계된 csv_text)
    │    └─ GMM 클러스터 예측
    │
    POST /api/advise                   (집계된 csv_text)
         └─ chatbot.py:8000/api/analyze → Qwen 피드백
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

### 3. Jupyter Notebook에서 직접 사용
```python
import requests

SERVER = "http://172.16.64.2:8000"

# 더미 데이터 분석
r = requests.post(f"{SERVER}/api/analyze", json={"demo": True}, timeout=120)
result = r.json()
session_id = result["session_id"]

print(result["cluster_name"])    # 소비 유형
print(result["feedback"])        # AI 피드백
print(result["total_reduction"]) # 예상 절감액

# 이어서 대화
r2 = requests.post(f"{SERVER}/api/chat",
    json={"message": "절감 방법 더 알려줘", "session_id": session_id},
    timeout=120)
print(r2.json()["reply"])
```

### 4. DB 카테고리로 분석
```python
# DB expense_category 기준으로 집계한 데이터 전달
csv_text = """card_tpbuz_nm_2,amt,cnt
/root/workspace/data_analysis85000,6
exit/음료,42000,14
exit/제빵/떡/케익,18000,3
exit,55000,2
"""

r = requests.post(f"{SERVER}/api/analyze",
    json={"csv_text": csv_text}, timeout=120)
result = r.json()
```

---

## 서버 접근이 안 될 때 (방화벽/VPN)

SSH 터널링으로 우회:
```bash
# 로컬 터미널에서 실행
ssh -L 8000:localhost:8000 -L 5000:localhost:5000 user@172.16.64.2

# 이후 로컬에서 아래 주소로 접근
# chatbot API:  http://localhost:8000
# 웹 데모 UI:   http://localhost:5000
```

AWS EC2라면 보안 그룹에서 8000, 5000 포트 인바운드 규칙 추가 필요.
