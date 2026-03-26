# 소비 유형 분석 AI 서버

이 저장소의 AI 서버는 `GMM 소비 유형 분류`와 `Qwen 기반 소비 피드백 생성`을 묶어서 제공한다.  
외부 공개 API는 백엔드의 `/api/v1/ai/analysis`를 기준으로 하고, 이 레포는 그 뒤에서 동작하는 내부 분석 엔진 역할을 한다.

## 현재 기준

- 내부 AI API 엔드포인트: `POST /api/analyze`
- 상태 확인 엔드포인트: `GET /health`
- 선택 기능:
  - `POST /api/chat`
  - `DELETE /api/session/<id>`
- 기본 모델: `Qwen/Qwen2.5-7B-Instruct`
- 기본 로딩 방식: `4bit` 양자화
- 추론 기준 모델 산출물: `model/gmm_model.pkl`

## 역할 분리

이 레포의 내부 주소와 백엔드 공개 주소는 다르다.

- 백엔드 공개 API
  - `POST /api/v1/ai/analysis`
  - `GET /api/v1/ai/analysis`
- AI 레포 내부 API
  - `POST /api/analyze`

즉, 앱이나 프론트가 이 레포를 직접 호출하는 구조보다는 `백엔드 -> AI 서버` 구조를 권장한다.

## 디렉터리 구조

```text
data_analysis/
├── chatbot.py
├── client_example.py
├── run_demo.py
├── spending_advisor.py
├── gmm_predict.py
├── gmm_train.py
├── cluster_report.py
├── cluster_definitions.py
├── requirements.txt
├── model/
│   └── gmm_model.pkl
└── analysis/
    ├── demo_app.py
    ├── dummy_transactions.csv
    ├── sample_monthly_items.json
    ├── merchant_category_map.csv
    └── user_category_overrides.csv
```

## 원본 데이터 관련

공개 저장소에는 대용량 원본 학습 데이터가 포함되어 있지 않다.

- 없는 것:
  - `data/*.zip` 원본 학습 데이터
  - 개인 원본 거래 파일
- 있는 것:
  - 이미 학습된 `model/gmm_model.pkl`
  - 클러스터 요약 산출물

따라서 현재 가능한 작업은 다음과 같다.

- AI 서버 실행
- 월간 소비 분석 추론
- 백엔드 연동 개발

현재 불가능한 작업은 다음과 같다.

- GMM 재학습
- 원본 데이터 기준 재통계 산출

## 실행 환경

권장 환경:

- Python 3.11 이상
- CUDA 사용 가능 GPU
- Windows 로컬 개발 또는 Linux GPU 서버

현재 코드 기본값:

- 모델: `Qwen/Qwen2.5-7B-Instruct`
- 장치: CUDA 사용 가능 시 `cuda`, 아니면 `cpu`
- 분석 응답 토큰: `ANALYZE_MAX_NEW_TOKENS=256`
- 채팅 응답 토큰: `CHAT_MAX_NEW_TOKENS=160`
- 4bit 로딩: `LOAD_IN_4BIT=1`

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

필요하면 `.env.example`을 복사해 `.env`를 만든 뒤 값을 조정한다.

```bash
copy .env.example .env
```

## 서버 실행

### 1. 내부 AI 서버 실행

```bash
python chatbot.py --host 0.0.0.0 --port 8000
```

주요 환경변수:

- `MODEL_ID`
  - 기본값: `Qwen/Qwen2.5-7B-Instruct`
  - 로컬 폴더 모델도 가능
- `LOAD_IN_4BIT`
  - `1`이면 4bit 양자화 사용
- `ANALYZE_MAX_NEW_TOKENS`
- `CHAT_MAX_NEW_TOKENS`
- `HOST`
- `PORT`

예시:

```bash
set MODEL_ID=C:\Users\SSAFY\Desktop\data_analysis\models\Qwen2.5-7B-Instruct
set LOAD_IN_4BIT=1
python chatbot.py --host 127.0.0.1 --port 8000
```

### 2. 데모 서버 실행

`run_demo.py`는 로컬 HTML 데모를 띄운다.

```bash
python run_demo.py
```

기본 주소:

- `http://127.0.0.1:5000`

## 내부 AI API

### `GET /health`

서버와 GPU 상태를 확인한다.

예시 응답:

```json
{
  "status": "ok",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "device": "cuda",
  "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU",
  "active_sessions": 0,
  "activeSessions": 0
}
```

### `POST /api/analyze`

월간 소비 분석의 핵심 엔드포인트다.  
아래 세 가지 입력 방식 중 하나를 받는다.

- `demo: true`
- `csv_text` 또는 `csvText`
- `transactions` 또는 `items`

백엔드 연동 기준으로는 `items + keepSession: false` 사용을 권장한다.

#### 권장 요청 예시

```json
{
  "items": [
    {
      "category": "커피/음료",
      "amount": 233500,
      "count": 12
    },
    {
      "category": "육류/회식",
      "amount": 193200,
      "count": 4
    },
    {
      "category": "분식",
      "amount": 167600,
      "count": 7
    }
  ],
  "keepSession": false
}
```

#### 허용 입력 키

| 키 | 설명 |
|---|---|
| `demo` | 내장 더미 데이터 사용 여부 |
| `csv_text` | CSV 문자열 입력 |
| `csvText` | camelCase CSV 문자열 입력 |
| `transactions` | 거래 또는 집계 배열 입력 |
| `items` | 권장 집계 배열 입력 |
| `keepSession` | 세션 유지 여부 |
| `keep_session` | 세션 유지 여부 레거시 키 |
| `sessionId` | 기존 세션 ID |
| `session_id` | 기존 세션 ID 레거시 키 |

#### 권장 응답 필드

이 서버는 호환성을 위해 `snake_case`와 `camelCase`를 함께 내려주지만, 새 연동에서는 아래 `camelCase` 필드를 기준으로 사용한다.

| 키 | 설명 |
|---|---|
| `sessionId` | 세션 ID |
| `cluster` | 클러스터 객체 |
| `cluster.id` | 클러스터 번호 |
| `cluster.name` | 클러스터 이름 |
| `cluster.description` | 클러스터 설명 |
| `cluster.icon` | 클러스터 아이콘 |
| `categories` | 카테고리별 소비 분석 |
| `overspending` | 절감 우선 항목 |
| `summary.totalSavable` | 총 절감 가능 금액 |
| `summary.expectedSpending` | 절감 후 예상 지출 |
| `feedback` | AI 피드백 본문 |
| `sourceTransactionCount` | 분석에 사용된 총 거래 건수 |
| `sourceTotalSpending` | 분석에 사용된 총 지출액 |

#### 응답 예시

```json
{
  "sessionId": null,
  "cluster": {
    "id": 1,
    "name": "사무·서적형",
    "description": "사무/교육용품과 서적/도서 비중이 높은 유형",
    "icon": "📚"
  },
  "categories": [
    {
      "name": "커피/음료",
      "amount": 233500,
      "myRatio": 23.3,
      "baseRatio": 8.9,
      "diff": 14.4
    }
  ],
  "overspending": [
    {
      "name": "커피/음료",
      "myRatio": 23.3,
      "baseRatio": 8.9,
      "savableAmount": 70050
    }
  ],
  "summary": {
    "totalSavable": 143670,
    "expectedSpending": 860842
  },
  "feedback": "이번 달은 커피/음료와 육류/회식 비중이 높아 우선적으로 조정하는 것이 좋습니다.",
  "sourceTransactionCount": 28,
  "sourceTotalSpending": 1004512
}
```

### `POST /api/chat`

세션 기반 후속 대화를 제공한다.  
월 1회 리포트 중심 구조에서는 필수는 아니고, 운영 중 선택 기능으로 보는 편이 맞다.

요청 예시:

```json
{
  "message": "커피 외에 다른 절감 방법 알려줘",
  "sessionId": "9bed598b-..."
}
```

응답 예시:

```json
{
  "sessionId": "9bed598b-...",
  "reply": "외식비를 줄이기 위해서는..."
}
```

### `DELETE /api/session/<session_id>`

세션 기록을 삭제한다.

## 샘플 파일

- 더미 CSV:
  - `analysis/dummy_transactions.csv`
- 월간 분석 샘플 집계:
  - `analysis/sample_monthly_items.json`

## 예제 클라이언트

`client_example.py`는 아래 두 가지 흐름을 바로 테스트할 수 있게 맞춰져 있다.

- `items + keepSession: false` 기반 월간 분석 호출
- `demo` 기반 세션 생성 후 `/api/chat` 확인

실행 전 `SERVER_URL`만 실제 서버 주소로 바꾸면 된다.

## 주의 사항

- `analysis/user_category_overrides.csv`는 현재 데모용 파일 기반 오버라이드 저장 구조다.
- 실서비스에서는 사용자 단위 저장소로 옮기는 것을 권장한다.
- 공개 API 명세는 `camelCase` 기준으로 잡고, 이 AI 서버는 내부 호환성을 위해 레거시 키를 같이 유지하는 방향으로 관리한다.
