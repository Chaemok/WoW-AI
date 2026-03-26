# 백엔드 AI 연동 가이드

이 문서는 백엔드가 Runpod 내부 AI 서버를 호출할 때 필요한 최소 규약만 빠르게 정리한 문서다.  
기준 환경은 `Runpod A40 48GB`, `Qwen/Qwen2.5-14B-Instruct`, `Container 20GB / Volume 60GB`이며, fresh Pod 재현과 스모크 테스트까지 완료했다.

## 1. 연결 구조

- 외부 클라이언트는 AI 서버를 직접 호출하지 않는다.
- 권장 구조는 `클라이언트 -> 백엔드 -> AI 서버`다.
- AI 서버는 내부 분석 엔진 역할만 맡는다.

## 2. 기본 주소

- health: `GET http://<AI_SERVER_HOST>:8000/health`
- 분석: `POST http://<AI_SERVER_HOST>:8000/api/analyze`
- 채팅: `POST http://<AI_SERVER_HOST>:8000/api/chat`
- 세션 삭제: `DELETE http://<AI_SERVER_HOST>:8000/api/session/<sessionId>`

예시:

```text
http://172.21.0.2:8000
```

실제 운영에서는 Runpod Pod 내부 주소나 프록시 주소에 맞춰 백엔드 설정값으로 분리하는 것을 권장한다.

## 3. 상태 확인

### `GET /health`

용도:

- 서버 살아있는지 확인
- 현재 모델, 디바이스, GPU 확인

예시 응답:

```json
{
  "status": "ok",
  "model": "/workspace/WoW-AI/.cache/huggingface/models--Qwen--Qwen2.5-14B-Instruct/snapshots/<hash>",
  "device": "cuda",
  "gpu": "NVIDIA A40",
  "active_sessions": 0,
  "activeSessions": 0
}
```

권장 timeout:

- `5초`

## 4. 핵심 분석 API

### `POST /api/analyze`

백엔드 연동에서는 `items + keepSession: false` 형식을 권장한다.

### 권장 요청 바디

```json
{
  "items": [
    {
      "category": "커피/음료",
      "amount": 70050,
      "count": 12
    },
    {
      "category": "외식/배달",
      "amount": 193200,
      "count": 4
    },
    {
      "category": "교통",
      "amount": 167600,
      "count": 7
    }
  ],
  "keepSession": false
}
```

### 필드 설명

- `items[].category`: 업종명 또는 카테고리명
- `items[].amount`: 합계 금액
- `items[].count`: 거래 횟수
- `keepSession`: 일반 분석만 할 때는 `false` 권장

서버는 호환성을 위해 `snake_case`와 `camelCase`를 함께 지원하지만, 새 연동에서는 `camelCase` 기준을 권장한다.

### 주요 응답 필드

```json
{
  "sessionId": null,
  "cluster": {
    "id": 1,
    "name": "사무·서적형",
    "description": "사무/서적 관련 소비 비중이 높은 유형",
    "icon": "📘"
  },
  "categories": [
    {
      "name": "커피/음료",
      "amount": 70050,
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
    "totalSavable": 156281,
    "expectedSpending": 675219
  },
  "feedback": "소비 패턴에 대한 AI 피드백",
  "sourceTransactionCount": 31,
  "sourceTotalSpending": 831500
}
```

백엔드에서 우선적으로 쓰기 좋은 필드:

- `cluster.id`
- `cluster.name`
- `cluster.description`
- `overspending`
- `summary.totalSavable`
- `summary.expectedSpending`
- `feedback`
- `sourceTransactionCount`
- `sourceTotalSpending`

권장 timeout:

- 분석: `180초`

이 값은 실제 `client_example.py` 기준과 맞춰두는 것을 권장한다.

## 5. 선택 채팅 API

### `POST /api/chat`

세션 기반 후속 대화가 필요할 때만 사용한다.

예시 요청:

```json
{
  "message": "이번 달 줄일 수 있는 소비 1가지만 짧게 말해줘",
  "sessionId": "optional-session-id"
}
```

예시 응답:

```json
{
  "sessionId": "session-id",
  "reply": "커피/음료 지출을 먼저 줄여보세요."
}
```

권장 timeout:

- 채팅: `120초`

## 6. 콜드 스타트 대응

14B는 시작 직후 바로 응답하지 않을 수 있다.

실제 검증 결과:

- fresh Pod 기준 콜드 스타트: 약 `1분 30초 ~ 2분`
- background 시작 직후에는 `health`가 실패할 수 있음
- 실제 재현에서는 `sleep 100` 뒤 health와 smoke test가 안정적으로 성공

권장 운영 방식:

1. AI 서버 재시작 직후에는 사용자 트래픽을 바로 붙이지 않는다.
2. 백엔드 또는 운영 스크립트에서 먼저 `GET /health`를 확인한다.
3. 필요하면 배포 직후 `sleep 100` 후 warm-up 확인을 수행한다.

예시:

```bash
./status_chatbot_linux.sh
sleep 100
./status_chatbot_linux.sh
curl http://127.0.0.1:8000/health
python smoke_test_api.py --server-url http://127.0.0.1:8000
```

## 7. 권장 호출 순서

일반 분석 요청 흐름:

1. 백엔드가 AI 서버 `GET /health`를 확인
2. `POST /api/analyze` 호출
3. 필요한 핵심 필드만 백엔드 DTO로 매핑
4. 프론트에는 백엔드 공개 API 형식으로 재가공해 응답

배포 직후 흐름:

1. AI 서버 재기동
2. `sleep 100`
3. `GET /health`
4. `smoke_test_api.py` 또는 샘플 analyze 호출
5. 정상 확인 후 트래픽 연결

## 8. 에러 대응 메모

- `Connection refused`
  - 서버가 아직 모델 로딩 중일 가능성이 큼
- `Disk quota exceeded`
  - Pod 스토리지 설정 확인 필요
- `No space left on device`
  - root 캐시나 tmp 경로가 `/workspace`로 안 잡혔을 가능성 큼
- `incomplete metadata, file not fully covered`
  - Hugging Face shard 손상 가능성, 캐시 정리 후 재다운로드 필요

트러블슈팅 상세 내용은 [RUNPOD_14B_TROUBLESHOOTING.md](C:\Users\SSAFY\Desktop\S14P21D106\ai_repo\RUNPOD_14B_TROUBLESHOOTING.md)를 참고한다.

## 9. 한 줄 요약

- 백엔드는 `GET /health`와 `POST /api/analyze`만 우선 붙이면 된다.
- 분석 요청은 `items + keepSession: false`를 권장한다.
- 14B는 콜드 스타트가 길 수 있으니 배포 직후에는 `sleep 100` 후 확인하는 흐름이 안전하다.
