"""HTML demo server for prediction, upload preprocessing, and merchant override management."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import os
import requests as http_requests
import pandas as pd
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=PROJECT_ROOT / '.env')
load_dotenv(dotenv_path=Path(__file__).resolve().parent / '.env')

GMM_KEY = os.getenv("GMS_KEY")
print(f"[DEBUG] GMM_KEY 로드: {bool(GMM_KEY)}")  # 확인용

from cluster_definitions import FINAL_CLUSTER_DEFINITIONS, FINAL_CLUSTER_LABELS  # noqa: E402
from gmm_predict import get_available_categories, predict_spending_type  # noqa: E402
from spending_advisor import analyze_and_advise  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
DUMMY_CSV_PATH = BASE_DIR / 'dummy_transactions.csv'
OVERRIDE_CSV_PATH = BASE_DIR / 'user_category_overrides.csv'
MERCHANT_MAP_PATH = BASE_DIR / 'merchant_category_map.csv'
OVERRIDE_COLUMNS = ['merchant_name', 'category', 'reason']
CARD_LIKE_TYPES = {'체크카드', '카드결제', '신한카드'}
EXCLUDE_LABEL = '제외'

# 가맹점명 키워드 → 카테고리 자동 분류 규칙 (맵에 없는 신규 가맹점에 적용)
# 순서대로 매칭 시도하며 첫 번째로 맞는 규칙 적용
KEYWORD_CATEGORY_RULES: list[tuple[str, str]] = [
    ('카페', '커피/음료'),
    ('커피', '커피/음료'),
    ('coffee', '커피/음료'),
    ('cafe', '커피/음료'),
    ('베이커리', '제과/제빵'),
    ('빵', '제과/제빵'),
    ('bakery', '제과/제빵'),
    ('편의점', '음/식료품소매'),
    ('마트', '음/식료품소매'),
    ('슈퍼', '음/식료품소매'),
    ('약국', '의약/의료품'),
    ('병원', '병원/의료'),
    ('의원', '병원/의료'),
    ('주유', '자동차/유지비'),
    ('주차', '자동차/유지비'),
    ('택시', '교통서비스'),
    ('버스', '교통서비스'),
    ('지하철', '교통서비스'),
]

CATEGORY_HELP = {
    '외식': '한식, 일식/수산물, 별식/퓨전요리, 양식, 중식, 부페를 통합한 입력값이다.',
}

app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'))


def load_dummy_csv_text() -> str:
    return DUMMY_CSV_PATH.read_text(encoding='utf-8').strip()


def decode_csv_bytes(data: bytes) -> str:
    for encoding in ('utf-8-sig', 'utf-8', 'cp949', 'euc-kr'):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace')


def _is_excel_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in {'.xls', '.xlsx'}


def excel_bytes_to_csv_text(data: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    engine = 'xlrd' if suffix == '.xls' else 'openpyxl'
    df = pd.read_excel(io.BytesIO(data), engine=engine, header=None)
    return df.to_csv(index=False, header=False)


def find_header_row(csv_text: str, header_name: str) -> int:
    for index, line in enumerate(csv_text.splitlines()):
        if line.startswith(f'{header_name},'):
            return index
    raise ValueError(f'헤더 {header_name} 를 찾지 못했습니다.')


def load_bank_transactions_from_text(csv_text: str) -> pd.DataFrame:
    header_row = find_header_row(csv_text, '거래일자')
    bank = pd.read_csv(io.StringIO(csv_text), skiprows=header_row, engine='python')
    bank['출금(원)'] = pd.to_numeric(bank['출금(원)'], errors='coerce').fillna(0)
    bank = bank[bank['출금(원)'] > 0].copy()
    bank['transaction_datetime'] = pd.to_datetime(
        bank['거래일자'].astype(str) + ' ' + bank['거래시간'].astype(str),
        errors='coerce',
    )
    bank = bank[bank['transaction_datetime'].notna()].copy()
    bank['payment_method'] = bank['적요'].map(lambda value: '카드' if value in CARD_LIKE_TYPES else '계좌')
    bank['merchant_name'] = bank['내용'].fillna('').astype(str).str.strip()
    bank['transaction_detail'] = bank['적요'].fillna('').astype(str).str.strip()
    bank['amount'] = bank['출금(원)'].astype(int)
    bank['source'] = 'bank'
    bank['source_order'] = range(len(bank))
    return bank[[
        'transaction_datetime',
        'payment_method',
        'amount',
        'merchant_name',
        'transaction_detail',
        'source',
        'source_order',
    ]]


def load_card_transactions_from_text(csv_text: str) -> pd.DataFrame:
    header_row = find_header_row(csv_text, '거래일')
    card = pd.read_csv(io.StringIO(csv_text), skiprows=header_row, engine='python')
    card = card[card['거래일'] != '합계'].copy()
    card['이용금액'] = (
        card['이용금액']
        .astype(str)
        .str.replace(',', '', regex=False)
        .pipe(pd.to_numeric, errors='coerce')
        .fillna(0)
    )
    card = card[card['이용금액'] > 0].copy()
    card['transaction_datetime'] = pd.to_datetime(card['거래일'], format='%Y.%m.%d', errors='coerce')
    card = card[card['transaction_datetime'].notna()].copy()
    card['payment_method'] = '카드'
    card['merchant_name'] = card['가맹점명'].fillna('').astype(str).str.strip()
    card['transaction_detail'] = card['상품구분'].fillna('').astype(str).str.strip()
    card['amount'] = card['이용금액'].astype(int)
    card['source'] = 'card'
    card['source_order'] = range(len(card))
    return card[[
        'transaction_datetime',
        'payment_method',
        'amount',
        'merchant_name',
        'transaction_detail',
        'source',
        'source_order',
    ]]


def build_transactions(bank_csv_text: str | None, card_csv_text: str | None) -> pd.DataFrame:
    frames = []
    if bank_csv_text:
        frames.append(load_bank_transactions_from_text(bank_csv_text))
    if card_csv_text:
        frames.append(load_card_transactions_from_text(card_csv_text))
    if not frames:
        raise ValueError('bank 또는 card CSV 중 하나는 업로드해야 합니다.')

    transactions = pd.concat(frames, ignore_index=True)
    transactions = transactions.sort_values(
        by=['transaction_datetime', 'source_order'],
        ascending=[True, True],
    ).reset_index(drop=True)
    transactions['cumulative_amount'] = transactions['amount'].cumsum()
    transactions = transactions.sort_values(
        by=['transaction_datetime', 'source_order'],
        ascending=[False, False],
    ).reset_index(drop=True)
    return transactions


def get_override_categories() -> list[str]:
    return list(get_available_categories())


def empty_override_df() -> pd.DataFrame:
    return pd.DataFrame(columns=OVERRIDE_COLUMNS)


def load_user_overrides() -> pd.DataFrame:
    if OVERRIDE_CSV_PATH.exists():
        override_df = pd.read_csv(OVERRIDE_CSV_PATH)
    else:
        override_df = empty_override_df()

    for column in OVERRIDE_COLUMNS:
        if column not in override_df.columns:
            override_df[column] = ''

    override_df = override_df[OVERRIDE_COLUMNS].copy()
    for column in OVERRIDE_COLUMNS:
        override_df[column] = override_df[column].fillna('').astype(str).str.strip()

    override_df = override_df.loc[override_df['merchant_name'].ne('')].drop_duplicates(
        subset=['merchant_name'], keep='last'
    )
    return override_df.reset_index(drop=True)


def load_merchant_category_map() -> pd.DataFrame:
    expected_columns = ['merchant_name', 'card_tpbuz_nm_2', 'classification_reason']
    if MERCHANT_MAP_PATH.exists():
        merchant_df = pd.read_csv(MERCHANT_MAP_PATH)
    else:
        merchant_df = pd.DataFrame(columns=expected_columns)

    for column in expected_columns:
        if column not in merchant_df.columns:
            merchant_df[column] = ''

    merchant_df = merchant_df[expected_columns].copy()
    merchant_df['merchant_name'] = merchant_df['merchant_name'].fillna('').astype(str).str.strip()
    merchant_df['card_tpbuz_nm_2'] = merchant_df['card_tpbuz_nm_2'].fillna('').astype(str).str.strip()
    merchant_df['classification_reason'] = merchant_df['classification_reason'].fillna('').astype(str).str.strip()
    merchant_df = merchant_df.loc[merchant_df['merchant_name'].ne('')].drop_duplicates(
        subset=['merchant_name'], keep='last'
    )

    override_df = load_user_overrides()
    if override_df.empty:
        return merchant_df.reset_index(drop=True)

    override_df = override_df.rename(
        columns={
            'category': 'card_tpbuz_nm_2',
            'reason': 'classification_reason',
        }
    )
    override_df['classification_reason'] = override_df['classification_reason'].replace(
        '', '사용자 지정 카테고리 재설정입니다.'
    )

    merged = merchant_df.set_index('merchant_name')
    merged.update(override_df.set_index('merchant_name'))
    merged = merged.reset_index()

    missing_override = override_df.loc[~override_df['merchant_name'].isin(merged['merchant_name'])]
    if not missing_override.empty:
        merged = pd.concat([merged, missing_override], ignore_index=True)

    return merged.drop_duplicates(subset=['merchant_name'], keep='last').reset_index(drop=True)


def save_user_overrides(rows: list[dict]) -> pd.DataFrame:
    categories = set(get_override_categories())
    cleaned_rows = []

    for item in rows:
        merchant_name = str(item.get('merchant_name', '') or '').strip()
        category = str(item.get('category', '') or '').strip()
        reason = str(item.get('reason', '') or '').strip()

        if not merchant_name:
            continue
        if not category:
            raise ValueError(f'{merchant_name}: category 값이 비어 있습니다.')
        if category not in categories:
            raise ValueError(f'{merchant_name}: 허용되지 않은 카테고리입니다. ({category})')

        cleaned_rows.append(
            {
                'merchant_name': merchant_name,
                'category': category,
                'reason': reason,
            }
        )

    override_df = pd.DataFrame(cleaned_rows, columns=OVERRIDE_COLUMNS)
    if not override_df.empty:
        override_df = override_df.drop_duplicates(subset=['merchant_name'], keep='last')

    OVERRIDE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    override_df.to_csv(OVERRIDE_CSV_PATH, index=False, encoding='utf-8-sig')
    return override_df.reset_index(drop=True)


def load_override_candidates(limit: int = 120) -> list[dict]:
    merchant_df = load_merchant_category_map()
    if merchant_df.empty:
        return []

    merchant_df = merchant_df.rename(
        columns={
            'card_tpbuz_nm_2': 'current_category',
            'classification_reason': 'current_reason',
        }
    )
    merchant_df['merchant_name'] = merchant_df['merchant_name'].fillna('').astype(str).str.strip()
    merchant_df['current_category'] = merchant_df['current_category'].fillna('').astype(str).str.strip()
    merchant_df['current_reason'] = merchant_df['current_reason'].fillna('').astype(str).str.strip()
    merchant_df = merchant_df.loc[merchant_df['merchant_name'].ne('')]

    merchant_df['priority'] = merchant_df['current_category'].eq('제외').astype(int)
    merchant_df = merchant_df.sort_values(
        ['priority', 'merchant_name'],
        ascending=[False, True],
    ).head(limit)

    return merchant_df[['merchant_name', 'current_category', 'current_reason']].to_dict('records')


def _classify_by_keyword(merchant_name: str) -> tuple[str, str] | None:
    """키워드 규칙으로 카테고리 추론. 매칭되면 (category, reason) 반환, 없으면 None."""
    name_lower = merchant_name.lower()
    for keyword, category in KEYWORD_CATEGORY_RULES:
        if keyword.lower() in name_lower:
            return category, f"가맹점명에 '{keyword}' 키워드가 포함되어 자동 분류됐습니다."
    return None


def _classify_by_gms(merchant_name: str) -> tuple[str, str] | None:
    """GMS GPT API로 가맹점 카테고리 분류"""
    if not GMM_KEY:
        return None

    VALID_CATEGORIES = [
        "인터넷쇼핑", "인테리어/가정용품", "교통서비스", "음/식료품소매",
        "외식", "제과/제빵/떡/케익", "커피/음료", "패스트푸드",
        "자동차/유지비", "시스템/통신", "건강/기호식품", "분식",
        "육류/회식", "선물/완구", "병원/의료", "화장품소매",
        "공연관람", "의약/의료품", "건강/뷰티/마사지", "수리서비스",
    ]

    try:
        response = http_requests.post(
            "https://gms.ssafy.io/gmsapi/api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GMM_KEY}",
            },
            json={
                "model": "gpt-5.2",
                "messages": [
                    {
                        "role": "developer",
                        "content": f"다음 가맹점명을 아래 카테고리 중 하나로만 분류해. 카테고리명만 정확히 답해. 다른 말 하지 마.\n카테고리: {', '.join(VALID_CATEGORIES)}"
                    },
                    {
                        "role": "user",
                        "content": merchant_name
                    }
                ],
                "max_completion_tokens": 30,
                "temperature": 0,
            },
            timeout=10,
        )

        print(f"[GMS] status_code: {response.status_code}")
        print(f"[GMS] 응답 전체: {response.json()}")  # ← 추가

        data = response.json()
        category = data["choices"][0]["message"]["content"].strip()

        if category in VALID_CATEGORIES:
            return category, 'GMS AI 자동 분류'
        return None

    except Exception as e:
        print(f"[GMS] 에러: {e}")
        return None


def build_prediction_state(transactions: pd.DataFrame) -> dict:
    print(f"[DEBUG] GMM_KEY = '{GMM_KEY}'")
    merchant_map = load_merchant_category_map()
    labeled = transactions.merge(merchant_map, on='merchant_name', how='left')

    # 맵에 없는 가맹점에 키워드 규칙 적용
    unmatched = labeled['card_tpbuz_nm_2'].isna()
    if unmatched.any():
        def _apply_keyword(row):
            # 1단계: 키워드 매칭
            result = _classify_by_keyword(row['merchant_name'])
            if result:
                row['card_tpbuz_nm_2'], row['classification_reason'] = result
                return row

            # 2단계: GMS GPT API 분류
            gms_result = _classify_by_gms(row['merchant_name'])
            if gms_result:
                row['card_tpbuz_nm_2'], row['classification_reason'] = gms_result
                return row

            # 3단계: 실패 → 제외
            row['card_tpbuz_nm_2'] = EXCLUDE_LABEL
            row['classification_reason'] = '분류 맵에 없는 merchant_name 입니다.'
            return row

        labeled.loc[unmatched] = labeled.loc[unmatched].apply(_apply_keyword, axis=1)

    labeled['card_tpbuz_nm_2'] = labeled['card_tpbuz_nm_2'].fillna(EXCLUDE_LABEL)
    labeled['classification_reason'] = labeled['classification_reason'].fillna('분류 맵에 없는 merchant_name 입니다.')

    included = (
        labeled[labeled['card_tpbuz_nm_2'] != EXCLUDE_LABEL]
        .groupby('card_tpbuz_nm_2', as_index=False)
        .agg(amt=('amount', 'sum'), cnt=('amount', 'size'))
        .sort_values(['amt', 'cnt'], ascending=[False, False])
        .reset_index(drop=True)
    )

    excluded = (
        labeled[labeled['card_tpbuz_nm_2'] == EXCLUDE_LABEL]
        .groupby(['merchant_name', 'classification_reason'], as_index=False)
        .agg(
            amount=('amount', 'sum'),
            cnt=('amount', 'size'),
            payment_methods=('payment_method', lambda s: ', '.join(sorted(set(s.astype(str))))),
            sources=('source', lambda s: ', '.join(sorted(set(s.astype(str))))),
        )
        .sort_values(['amount', 'cnt', 'merchant_name'], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    items = labeled.assign(
        transaction_date=labeled['transaction_datetime'].dt.strftime('%Y-%m-%d'),
        status=labeled['card_tpbuz_nm_2'].eq(EXCLUDE_LABEL).map(
            {True: 'needs-category', False: 'classified'}
        ),
    )[
        [
            'transaction_date',
            'merchant_name',
            'transaction_detail',
            'amount',
            'card_tpbuz_nm_2',
            'classification_reason',
            'status',
        ]
    ].to_dict('records')

    return {
        'records': included.to_dict('records'),
        'items': items,
        'excluded_rows': excluded.to_dict('records'),
        'transaction_count': int(len(transactions)),
        'included_amount': int(included['amt'].sum()) if not included.empty else 0,
        'excluded_amount': int(excluded['amount'].sum()) if not excluded.empty else 0,
        'total_amount': int(transactions['amount'].sum()),
    }


@app.get('/')
def index():
    return render_template(
        'predict_demo.html',
        cluster_definitions=FINAL_CLUSTER_DEFINITIONS,
        categories=get_available_categories(),
        category_help=CATEGORY_HELP,
        dummy_csv=load_dummy_csv_text(),
        override_rows=load_user_overrides().to_dict('records'),
        override_candidates=load_override_candidates(),
        override_path='analysis/user_category_overrides.csv',
    )


@app.get('/api/overrides')
def api_get_overrides():
    return jsonify(
        {
            'rows': load_user_overrides().to_dict('records'),
            'candidates': load_override_candidates(),
            'path': 'analysis/user_category_overrides.csv',
        }
    )


@app.post('/api/overrides')
def api_save_overrides():
    payload = request.get_json(silent=True) or {}
    rows = payload.get('rows') or []
    if not isinstance(rows, list):
        return jsonify({'error': 'rows 는 배열이어야 합니다.'}), 400

    try:
        override_df = save_user_overrides(rows)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify(
        {
            'message': f'override {len(override_df)}건 저장 완료',
            'rows': override_df.to_dict('records'),
        }
    )


@app.post('/api/preprocess-transactions')
def api_preprocess_transactions():
    bank_file = request.files.get('bank_file')
    card_file = request.files.get('card_file')

    if (bank_file is None or not bank_file.filename) and (card_file is None or not card_file.filename):
        return jsonify({'error': 'bank 또는 card CSV 파일을 하나 이상 업로드해야 합니다.'}), 400

    def _read_file_as_csv_text(f) -> str | None:
        if not (f and f.filename):
            return None
        raw = f.read()
        if _is_excel_filename(f.filename):
            return excel_bytes_to_csv_text(raw, f.filename)
        return decode_csv_bytes(raw)

    try:
        bank_csv_text = _read_file_as_csv_text(bank_file)
        card_csv_text = _read_file_as_csv_text(card_file)
        transactions = build_transactions(bank_csv_text, card_csv_text)
        payload = build_prediction_state(transactions)
    except Exception as exc:  # pragma: no cover
        return jsonify({'error': str(exc)}), 400

    return jsonify(
        {
            'message': '업로드한 거래 CSV 전처리 완료',
            **payload,
        }
    )


@app.post('/api/predict')
def api_predict():
    payload = request.get_json(silent=True) or {}
    csv_text = (payload.get('csv_text') or '').strip()
    expected_cluster_id = payload.get('expected_cluster_id')

    if not csv_text:
        return jsonify({'error': 'CSV 입력이 비어 있습니다.'}), 400

    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:  # pragma: no cover
        return jsonify({'error': f'CSV 파싱 실패: {exc}'}), 400

    required_cols = {'card_tpbuz_nm_2', 'amt', 'cnt'}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        return jsonify({'error': f'필수 컬럼 누락: {", ".join(missing_cols)}'}), 400

    try:
        result = predict_spending_type(df)
    except Exception as exc:  # pragma: no cover
        return jsonify({'error': str(exc)}), 400

    expected = None
    matches = None
    if expected_cluster_id not in (None, ''):
        expected_cluster_id = int(expected_cluster_id)
        expected = {
            'cluster_id': expected_cluster_id,
            'cluster_name': FINAL_CLUSTER_LABELS.get(expected_cluster_id, f'Cluster {expected_cluster_id}'),
            'cluster_description': FINAL_CLUSTER_DEFINITIONS.get(expected_cluster_id, {}).get('description', ''),
        }
        matches = expected_cluster_id == result['cluster_id']

    return jsonify(
        {
            'expected': expected,
            'actual': result,
            'matches': matches,
            'row_count': int(len(df)),
            'total_amount': int(df['amt'].sum()),
            'total_count': int(df['cnt'].sum()),
        }
    )


def run_demo_server(host: str = '127.0.0.1', port: int = 5000, debug: bool = False):
    """HTML demo server 실행"""
    app.run(host=host, port=port, debug=debug, use_reloader=False)


@app.post('/api/advise')
def api_advise():
    """CSV 거래 내역 → Qwen2.5-14B 소비 분석 피드백 (chatbot.py 서버 연동)"""
    payload = request.get_json(silent=True) or {}
    csv_text = (payload.get('csv_text') or '').strip()

    if not csv_text:
        return jsonify({'error': 'csv_text가 비어 있습니다.'}), 400

    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        return jsonify({'error': f'CSV 파싱 실패: {exc}'}), 400

    required_cols = {'card_tpbuz_nm_2', 'amt', 'cnt'}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        return jsonify({'error': f'필수 컬럼 누락: {", ".join(missing_cols)}'}), 400

    try:
        feedback, reduction_dict, cluster_stats = analyze_and_advise(df=df, verbose=False)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    return jsonify({
        'feedback': feedback,
        'reduction_summary': reduction_dict,
        'total_reduction': sum(reduction_dict.values()),
        'cluster_stats': cluster_stats,
    })


if __name__ == '__main__':
    run_demo_server(debug=True)
