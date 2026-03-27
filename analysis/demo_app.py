"""HTML demo server for prediction, upload preprocessing, and merchant override management."""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests as http_requests
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / '.env')
load_dotenv(Path(__file__).resolve().parent / '.env')

from cluster_definitions import FINAL_CLUSTER_DEFINITIONS, FINAL_CLUSTER_LABELS  # noqa: E402
from gmm_predict import get_available_categories, predict_spending_type  # noqa: E402
from spending_advisor import analyze_and_advise  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
DUMMY_CSV_PATH = BASE_DIR / 'dummy_transactions.csv'
OVERRIDE_CSV_PATH = BASE_DIR / 'user_category_overrides.csv'
MERCHANT_MAP_PATH = BASE_DIR / 'merchant_category_map.csv'
GMS_KEY = (
    os.getenv('CATEGORY_LLM_API_KEY')
    or os.getenv('GMS_KEY')
    or os.getenv('OPENAI_API_KEY')
    or ''
).strip()
GMS_BASE_URL = os.getenv('CATEGORY_LLM_BASE_URL', 'https://gms.ssafy.io/gmsapi/api.openai.com/v1').rstrip('/')
GMS_MODEL = os.getenv('CATEGORY_LLM_MODEL', 'gpt-5.2')
GMS_TIMEOUT_SEC = int(os.getenv('CATEGORY_LLM_TIMEOUT_SEC', '20'))
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
    ('바게뜨', '제과/제빵'),
    ('파리바게뜨', '제과/제빵'),
    ('편의점', '음/식료품소매'),
    ('마트', '음/식료품소매'),
    ('슈퍼', '음/식료품소매'),
    ('gs25', '음/식료품소매'),
    ('cu', '음/식료품소매'),
    ('세븐일레븐', '음/식료품소매'),
    ('이마트24', '음/식료품소매'),
    ('다이소', '인테리어/가정용품'),
    ('올리브영', '화장품소매'),
    ('약국', '의약/의료품'),
    ('병원', '병원/의료'),
    ('의원', '병원/의료'),
    ('주유', '자동차/유지비'),
    ('주차', '자동차/유지비'),
    ('택시', '교통서비스'),
    ('버스', '교통서비스'),
    ('지하철', '교통서비스'),
]

# 소비처가 아닌 금융성/이체성 거래는 제외를 유지해야 한다.
# 다만 merchant map에 잘못 누적된 "제외"가 많아서, 명백한 소비처 키워드가 있는 경우에는
# 아래 키워드 규칙으로 다시 살려낼 수 있게 한다.
FINANCIAL_EXCLUDE_KEYWORDS: tuple[str, ...] = (
    '충전',
    '이체',
    '수수료',
    '카드대금',
    '카드사용알림서비스',
    '자동이체',
    '계좌이체',
    '송금',
    '결제대행',
)

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


def _normalize_merchant_key(value: str) -> str:
    """상호명을 머지 키 용도로 정규화한다.

    엑셀 원본은 공백, 법인 표기, 특수문자 차이 때문에 같은 상호가 다르게 들어오는 일이 많다.
    merchant map exact match 실패를 줄이기 위해 비교용 키를 별도로 만든다.
    """
    text = str(value or '').strip().lower()
    if not text:
        return ''

    text = text.replace('(주)', '').replace('㈜', '').replace('주식회사', '')
    text = re.sub(r'[\s\-_()/.,·]+', '', text)
    return text


def _has_financial_exclude_signal(merchant_name: str) -> bool:
    name = str(merchant_name or '')
    return any(keyword in name for keyword in FINANCIAL_EXCLUDE_KEYWORDS)


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
    bank['merchant_key'] = bank['merchant_name'].map(_normalize_merchant_key)
    bank['transaction_detail'] = bank['적요'].fillna('').astype(str).str.strip()
    bank['amount'] = bank['출금(원)'].astype(int)
    bank['source'] = 'bank'
    bank['source_order'] = range(len(bank))
    return bank[[
        'transaction_datetime',
        'payment_method',
        'amount',
        'merchant_name',
        'merchant_key',
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
    card['merchant_key'] = card['merchant_name'].map(_normalize_merchant_key)
    card['transaction_detail'] = card['상품구분'].fillna('').astype(str).str.strip()
    card['amount'] = card['이용금액'].astype(int)
    card['source'] = 'card'
    card['source_order'] = range(len(card))
    return card[[
        'transaction_datetime',
        'payment_method',
        'amount',
        'merchant_name',
        'merchant_key',
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
    override_df['merchant_key'] = override_df['merchant_name'].map(_normalize_merchant_key)

    override_df = override_df.loc[override_df['merchant_key'].ne('')].drop_duplicates(
        subset=['merchant_key'], keep='last'
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
    merchant_df['merchant_key'] = merchant_df['merchant_name'].map(_normalize_merchant_key)
    merchant_df['card_tpbuz_nm_2'] = merchant_df['card_tpbuz_nm_2'].fillna('').astype(str).str.strip()
    merchant_df['classification_reason'] = merchant_df['classification_reason'].fillna('').astype(str).str.strip()
    merchant_df = merchant_df.loc[merchant_df['merchant_key'].ne('')].drop_duplicates(
        subset=['merchant_key'], keep='last'
    )

    # 과거 merchant map에는 잘못 누적된 "제외"가 섞여 있다.
    # 명백한 소비처 키워드가 보이면 제외를 그대로 믿지 말고 다시 소비 카테고리로 복구한다.
    exclude_mask = merchant_df['card_tpbuz_nm_2'].eq(EXCLUDE_LABEL)
    for idx in merchant_df.index[exclude_mask]:
        merchant_name = merchant_df.at[idx, 'merchant_name']
        if _has_financial_exclude_signal(merchant_name):
            continue

        keyword_result = _classify_by_keyword(merchant_name)
        if keyword_result is None:
            continue

        repaired_category, repaired_reason = keyword_result
        merchant_df.at[idx, 'card_tpbuz_nm_2'] = repaired_category
        merchant_df.at[idx, 'classification_reason'] = (
            f"기존 exclude merchant map을 재검토해 소비처로 복구했습니다. {repaired_reason}"
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

    merged = merchant_df.set_index('merchant_key')
    merged.update(override_df.set_index('merchant_key'))
    merged = merged.reset_index()

    missing_override = override_df.loc[~override_df['merchant_key'].isin(merged['merchant_key'])]
    if not missing_override.empty:
        merged = pd.concat([merged, missing_override], ignore_index=True)

    return merged.drop_duplicates(subset=['merchant_key'], keep='last').reset_index(drop=True)


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
    merchant_df['merchant_key'] = merchant_df['merchant_name'].map(_normalize_merchant_key)
    merchant_df['current_category'] = merchant_df['current_category'].fillna('').astype(str).str.strip()
    merchant_df['current_reason'] = merchant_df['current_reason'].fillna('').astype(str).str.strip()
    merchant_df = merchant_df.loc[merchant_df['merchant_key'].ne('')]

    merchant_df['priority'] = merchant_df['current_category'].eq('제외').astype(int)
    merchant_df = merchant_df.sort_values(
        ['priority', 'merchant_name'],
        ascending=[False, True],
    ).head(limit)

    return merchant_df[['merchant_name', 'current_category', 'current_reason']].to_dict('records')


def _classify_by_keyword(merchant_name: str) -> tuple[str, str] | None:
    """키워드 규칙으로 카테고리 추론. 매칭되면 (category, reason) 반환, 없으면 None."""
    name_lower = merchant_name.lower()
    merchant_key = _normalize_merchant_key(merchant_name)
    for keyword, category in KEYWORD_CATEGORY_RULES:
        normalized_keyword = _normalize_merchant_key(keyword)
        if keyword.lower() in name_lower or (normalized_keyword and normalized_keyword in merchant_key):
            return category, f'가맹점명에 \'{keyword}\' 키워드가 포함되어 자동 분류됐습니다.'
    return None


# Legacy reference kept for weekend debugging.
# This was the pre-hybrid flow: merchant_key-based map -> keyword -> exclude.
# We keep the old implementation under a different name instead of deleting it,
# so we can compare behavior quickly if the new GMS-assisted path regresses.
def _legacy_build_prediction_state_without_llm(transactions: pd.DataFrame) -> dict:
    merchant_map = load_merchant_category_map()
    labeled = transactions.merge(merchant_map, on='merchant_key', how='left', suffixes=('', '_mapped'))
    if 'merchant_name_mapped' in labeled.columns:
        labeled = labeled.drop(columns=['merchant_name_mapped'])

    # 맵에 없는 가맹점에 키워드 규칙 적용
    unmatched = labeled['card_tpbuz_nm_2'].isna()
    if unmatched.any():
        def _apply_keyword(row):
            result = _classify_by_keyword(row['merchant_name'])
            if result:
                row['card_tpbuz_nm_2'], row['classification_reason'] = result
            else:
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

    return {
        'records': included.to_dict('records'),
        'excluded_rows': excluded.to_dict('records'),
        'transaction_count': int(len(transactions)),
        'included_amount': int(included['amt'].sum()) if not included.empty else 0,
        'excluded_amount': int(excluded['amount'].sum()) if not excluded.empty else 0,
        'total_amount': int(transactions['amount'].sum()),
    }


def _should_retry_excluded_map(row: pd.Series) -> bool:
    mapped_category = str(row.get('card_tpbuz_nm_2') or '').strip()
    if mapped_category != EXCLUDE_LABEL:
        return False
    return not _has_financial_exclude_signal(str(row.get('merchant_name') or ''))


def _classify_by_gms(row: pd.Series) -> tuple[str, str] | None:
    if not GMS_KEY:
        return None

    allowed_categories = list(get_available_categories())
    request_body = {
        'model': GMS_MODEL,
        'temperature': 0,
        'messages': [
            {
                'role': 'developer',
                'content': (
                    "다음 거래를 허용된 카테고리 중 정확히 하나로만 분류해. "
                    "카테고리명만 답하지 말고 JSON으로 답해. "
                    f"허용 카테고리: {', '.join(allowed_categories)}. "
                    f"애매하면 {EXCLUDE_LABEL} 로 답해."
                ),
            },
            {
                'role': 'user',
                'content': (
                    f"merchant_name={row['merchant_name']}\n"
                    f"transaction_detail={row['transaction_detail']}\n"
                    f"payment_method={row['payment_method']}\n"
                    f"amount={int(row['amount'])}"
                ),
            },
        ],
        'response_format': {'type': 'json_object'},
        'max_tokens': 120,
    }

    last_error = None
    for attempt in range(2):
        try:
            response = http_requests.post(
                f'{GMS_BASE_URL}/chat/completions',
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {GMS_KEY}',
                },
                json=request_body,
                timeout=GMS_TIMEOUT_SEC,
            )
            response.raise_for_status()
            data = response.json()
            content = data['choices'][0]['message']['content']
            parsed = json.loads(content)
            break
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(1)
                continue
            return None

    category = str(parsed.get('category') or '').strip()
    reason = str(parsed.get('reason') or '').strip()
    if category not in allowed_categories and category != EXCLUDE_LABEL:
        return None
    if not reason:
        reason = 'GMS classified this merchant from the available transaction fields.'
    return category, f'GMS({GMS_MODEL}) classified this merchant: {reason}'


def _resolve_unmatched_category_with_gms(
    row: pd.Series,
) -> tuple[str, str, bool, bool]:
    """Resolve an unmatched merchant with keyword-first, direct-GMS-second order.

    Legacy reference:
    - We previously used hybrid_category_classifier.GMSCategoryClassifier here.
    - The older teammate demo_app.py also preferred keyword -> GMS -> exclude order.
    - We now call GMS directly in this file, but keep the old intent documented
      so weekend debugging can compare the two paths quickly.
    """
    merchant_name = str(row['merchant_name'])
    keyword_result = _classify_by_keyword(merchant_name)
    if keyword_result:
        category, reason = keyword_result
        return category, reason, False, True

    gms_result = _classify_by_gms(row)
    if gms_result and gms_result[0] != EXCLUDE_LABEL:
        category, reason = gms_result
        return category, reason, True, False

    if gms_result:
        return EXCLUDE_LABEL, gms_result[1], True, False
    return EXCLUDE_LABEL, 'No merchant map match and no keyword/GMS match.', False, False


def build_prediction_state(transactions: pd.DataFrame) -> dict:
    merchant_map = load_merchant_category_map()
    labeled = transactions.merge(merchant_map, on='merchant_key', how='left', suffixes=('', '_mapped'))
    if 'merchant_name_mapped' in labeled.columns:
        labeled = labeled.drop(columns=['merchant_name_mapped'])

    merchant_map_classified = int(labeled['card_tpbuz_nm_2'].notna().sum())
    gms_attempted = 0
    gms_classified = 0
    keyword_classified = 0
    gms_cache: dict[str, tuple[str, str, bool, bool]] = {}

    # We retry two kinds of rows:
    # 1) no merchant-map match at all
    # 2) merchant map said "exclude", but the merchant does not look financial
    #    (this protects us from polluted exclude rows in merchant_category_map.csv).
    unresolved_mask = labeled['card_tpbuz_nm_2'].isna() | labeled.apply(_should_retry_excluded_map, axis=1)
    if unresolved_mask.any():
        unresolved = labeled.loc[unresolved_mask].copy()
        for idx, row in unresolved.iterrows():
            merchant_cache_key = str(row.get('merchant_key') or '')
            resolved_from_cache = False
            if merchant_cache_key and merchant_cache_key in gms_cache:
                category, reason, gms_used, keyword_used = gms_cache[merchant_cache_key]
                resolved_from_cache = True
            else:
                category, reason, gms_used, keyword_used = _resolve_unmatched_category_with_gms(row)
                if merchant_cache_key:
                    gms_cache[merchant_cache_key] = (category, reason, gms_used, keyword_used)
            if gms_used and not resolved_from_cache:
                gms_attempted += 1
                if category != EXCLUDE_LABEL:
                    gms_classified += 1
            if keyword_used and not resolved_from_cache:
                keyword_classified += 1
            labeled.at[idx, 'card_tpbuz_nm_2'] = category
            labeled.at[idx, 'classification_reason'] = reason

    labeled['card_tpbuz_nm_2'] = labeled['card_tpbuz_nm_2'].fillna(EXCLUDE_LABEL)
    labeled['classification_reason'] = labeled['classification_reason'].fillna(
        'No merchant map match and no keyword/GMS match.'
    )

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

    return {
        'records': included.to_dict('records'),
        'excluded_rows': excluded.to_dict('records'),
        'transaction_count': int(len(transactions)),
        'included_amount': int(included['amt'].sum()) if not included.empty else 0,
        'excluded_amount': int(excluded['amount'].sum()) if not excluded.empty else 0,
        'total_amount': int(transactions['amount'].sum()),
        'mapping_stats': {
            'merchant_map_classified': merchant_map_classified,
            'llm_enabled': bool(GMS_KEY),
            'llm_model': GMS_MODEL if GMS_KEY else None,
            'llm_attempted': gms_attempted,
            'llm_classified': gms_classified,
            'keyword_classified': keyword_classified,
            'excluded_transaction_count': int((labeled['card_tpbuz_nm_2'] == EXCLUDE_LABEL).sum()),
        },
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
