# 소비 유형 예측 데모

## 환경 설정
권장 환경은 Python 3.11 이상이다.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

GMS 분류 셀을 실행할 때는 루트에 `.env` 파일을 만들고 아래 값을 넣어야 한다.

```env
GMS_KEY=your-real-gms-key
```

예시는 [.env.example](.env.example) 에 두었다.

공개 저장소에는 개인 거래 원본과 대용량 학습 원본을 포함하지 않는다.

- 개인 거래 원본: `bank.csv`, `card.csv`, 은행/카드 원본 엑셀 파일
- 학습 원본: `data/*.zip`
- 개인 분석 산출물: `analysis/personal_transactions*.csv`, `analysis/user_category_overrides.csv`

학습이 필요하면 `data/` 아래에 월별 ZIP 파일을 직접 넣고 `python gmm_train.py` 를 실행하면 된다.

## 실행 방법
아래 한 줄만 실행하면 HTML 데모 서버가 켜지고 브라우저가 열린다.

- [run_demo.py](run_demo.py)

실행 명령:

python run_demo.py

기본 주소:
- http://127.0.0.1:5000

## 화면에서 할 수 있는 것
- 카테고리 선택
- 금액 입력 후 Enter로 거래 추가
- 필요하면 `cnt` 조정
- 예상 클러스터 선택
- 실제 예측 결과 확인
- 예상과 실제 비교

## 기본 입력 방식
화면에서 아래 순서로 입력하면 된다.

1. 카테고리 선택
2. 금액 입력
3. Enter 또는 `행 추가` 클릭
4. 거래 목록 누적
5. `예측 실행` 클릭

입력된 거래 목록은 내부적으로 CSV 형태로 변환되어 예측에 사용된다.

## 입력 데이터 형식
필수 컬럼은 3개다.

| 컬럼명 | 의미 |
|---|---|
| `card_tpbuz_nm_2` | 업종 소분류 |
| `amt` | 결제 금액 |
| `cnt` | 거래 건수 |

예시:

```csv
card_tpbuz_nm_2,amt,cnt
외식,15000,1
커피/음료,4500,1
육류/회식,38000,1
분식,7000,1
```

## 더미데이터
기본 더미데이터 파일:
- [analysis/dummy_transactions.csv](analysis/dummy_transactions.csv)

HTML 화면에서 바로 불러와 테스트할 수 있다.

## card_tpbuz_nm_2 기준표
- [analysis/card_tpbuz_nm_2_mapping.md](analysis/card_tpbuz_nm_2_mapping.md): 사람이 읽기 좋은 설명 문서
- [analysis/card_tpbuz_nm_2_mapping.csv](analysis/card_tpbuz_nm_2_mapping.csv): 원본값, 정제값, 상태(`keep`/`merge`/`exclude`) 표

## 예측 결과
현재 화면에서는 아래 내용을 보여준다.
- 실제 소비 유형명
- 유형 설명
- 기준 시그니처
- 주요 지출 항목
- 고정 클러스터 기준 수치
- 전체 클러스터 확률
- 예상 vs 실제 비교

## 주요 파일
- [gmm_train.py](gmm_train.py): 모델 학습
- [gmm_predict.py](gmm_predict.py): 개인 소비 유형 예측
- [cluster_report.py](cluster_report.py): 클러스터 해석 리포트 생성
- [cluster_definitions.py](cluster_definitions.py): 고정 클러스터 이름/설명
- [analysis/demo_app.py](analysis/demo_app.py): HTML 데모 서버
- [analysis/templates/predict_demo.html](analysis/templates/predict_demo.html): 데모 화면

## 참고
학습이 끝나서 [model/gmm_model.pkl](model/gmm_model.pkl) 이 존재해야 데모가 동작한다.

# data_analysis
