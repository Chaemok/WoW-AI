"""
소비 유형 분석 챗봇 API 서버

- GMM 모델로 소비 유형을 예측한다.
- Qwen 모델로 소비 피드백을 생성한다.
- 기존 응답은 유지하면서 백엔드 연동용 camelCase 응답도 함께 제공한다.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import torch
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# 레포 루트를 파이썬 경로에 추가한다.
DATA_ANALYSIS_ROOT = Path(__file__).resolve().parent
if str(DATA_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ANALYSIS_ROOT))

# 프로젝트 루트의 .env를 우선 로드한다.
load_dotenv(DATA_ANALYSIS_ROOT / ".env")

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANALYZE_MAX_NEW_TOKENS = int(os.environ.get("ANALYZE_MAX_NEW_TOKENS", "256"))
CHAT_MAX_NEW_TOKENS = int(os.environ.get("CHAT_MAX_NEW_TOKENS", "160"))
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "1") == "1"

SYSTEM_PROMPT = (
    "당신은 한국어로 대화하는 개인 소비 습관 분석 전문가 챗봇입니다. "
    "사용자의 소비 데이터를 분석하고 소비 패턴 진단과 절감 방향을 친절하고 구체적으로 안내해 주세요. "
    "분석 결과가 있을 경우 그 맥락을 유지하며 대화합니다."
)

# 클러스터별 대표 아이콘을 고정해 리포트 화면에서 바로 쓸 수 있게 한다.
CLUSTER_ICON_MAP = {
    0: "🛍️",
    1: "📚",
    2: "🚗",
    3: "🩺",
    4: "🍜",
    5: "🏥",
    6: "🎁",
    7: "☕",
}

_DEMO_DF = pd.DataFrame(
    [
        {"card_tpbuz_nm_2": "외식", "amt": 85000, "cnt": 6},
        {"card_tpbuz_nm_2": "커피/음료", "amt": 42000, "cnt": 14},
        {"card_tpbuz_nm_2": "육류/회식", "amt": 120000, "cnt": 3},
        {"card_tpbuz_nm_2": "편의점", "amt": 35000, "cnt": 10},
        {"card_tpbuz_nm_2": "온라인쇼핑", "amt": 98000, "cnt": 4},
        {"card_tpbuz_nm_2": "음/식료품소매", "amt": 60000, "cnt": 4},
        {"card_tpbuz_nm_2": "자동차/유지비", "amt": 45000, "cnt": 1},
    ]
)

_DUMMY_CSV_PATH = DATA_ANALYSIS_ROOT / "analysis" / "dummy_transactions.csv"


# 세션 기반 대화는 기존 호환성을 위해 유지한다.
sessions: dict[str, list[dict]] = {}
tokenizer = None
model = None


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


def load_model() -> None:
    """LLM을 한 번만 로드한다."""
    global tokenizer, model

    print(f"[INFO] 디바이스: {DEVICE}")
    if DEVICE == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"[INFO] 모델 로딩 중: {MODEL_ID} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model_kwargs = {"device_map": "auto"}
    if LOAD_IN_4BIT and DEVICE == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16 if DEVICE == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    model.eval()
    print("[INFO] 모델 로딩 완료")


def generate(history: list[dict], max_new_tokens: int, *, do_sample: bool = False) -> str:
    """Qwen 채팅 템플릿을 사용해 응답을 생성한다."""
    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "remove_invalid_values": True,
        "renormalize_logits": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = 0.7
        generation_kwargs["top_p"] = 0.9
    with torch.no_grad():
        try:
            output_ids = model.generate(**inputs, **generation_kwargs)
        except RuntimeError as exc:
            if not do_sample or "probability tensor contains either" not in str(exc):
                raise

            # 샘플링 logits가 불안정해지면 greedy decoding으로 한 번 더 시도한다.
            fallback_kwargs = dict(generation_kwargs)
            fallback_kwargs["do_sample"] = False
            fallback_kwargs.pop("temperature", None)
            fallback_kwargs.pop("top_p", None)
            output_ids = model.generate(**inputs, **fallback_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _get_json_body() -> dict:
    body = request.get_json(silent=True)
    return body if isinstance(body, dict) else {}


def _first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _load_demo_df() -> pd.DataFrame:
    if _DUMMY_CSV_PATH.exists():
        return pd.read_csv(_DUMMY_CSV_PATH)
    return _DEMO_DF.copy()


def _normalize_analysis_df(df: pd.DataFrame) -> pd.DataFrame:
    """분석에 필요한 최소 컬럼과 타입을 맞춘다."""
    required = {"card_tpbuz_nm_2", "amt", "cnt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {', '.join(sorted(missing))}")

    normalized = df.copy()
    normalized["card_tpbuz_nm_2"] = normalized["card_tpbuz_nm_2"].fillna("").astype(str).str.strip()
    normalized["amt"] = pd.to_numeric(normalized["amt"], errors="coerce")
    normalized["cnt"] = pd.to_numeric(normalized["cnt"], errors="coerce")
    normalized = normalized.dropna(subset=["amt", "cnt"])
    normalized = normalized[
        (normalized["card_tpbuz_nm_2"] != "")
        & (normalized["amt"] > 0)
        & (normalized["cnt"] > 0)
    ].copy()

    if normalized.empty:
        raise ValueError("유효한 분석 대상 거래가 없습니다.")

    normalized["amt"] = normalized["amt"].round().astype(int)
    normalized["cnt"] = normalized["cnt"].round().astype(int)
    return normalized.reset_index(drop=True)


def _load_csv_text_df(csv_text: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        raise ValueError(f"CSV 파싱 오류: {exc}") from exc
    return _normalize_analysis_df(df)


def _load_transactions_df(rows: list[dict]) -> pd.DataFrame:
    """개별 거래 목록이나 집계 목록을 분석용 포맷으로 맞춘다."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("transactions 또는 items는 비어 있지 않은 배열이어야 합니다.")

    normalized_rows: list[dict] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{index}번째 항목이 객체가 아닙니다.")

        category = _first_non_none(
            row.get("card_tpbuz_nm_2"),
            row.get("category"),
        )
        amount = _first_non_none(
            row.get("amt"),
            row.get("amount"),
        )
        count = _first_non_none(
            row.get("cnt"),
            row.get("count"),
            row.get("transactionCount"),
            1,
        )

        normalized_rows.append(
            {
                "card_tpbuz_nm_2": category,
                "amt": amount,
                "cnt": count,
            }
        )

    df = pd.DataFrame(normalized_rows)
    df = _normalize_analysis_df(df)

    # 같은 카테고리의 개별 거래를 묶어서 GMM 입력 형태로 맞춘다.
    return (
        df.groupby("card_tpbuz_nm_2", as_index=False)
        .agg({"amt": "sum", "cnt": "sum"})
        .sort_values("amt", ascending=False)
        .reset_index(drop=True)
    )


def _get_input_df(body: dict) -> pd.DataFrame:
    if body.get("demo", False):
        return _normalize_analysis_df(_load_demo_df())

    csv_text = _first_non_none(body.get("csv_text"), body.get("csvText"))
    if csv_text:
        return _load_csv_text_df(csv_text)

    transactions = _first_non_none(body.get("transactions"), body.get("items"))
    if transactions is not None:
        return _load_transactions_df(transactions)

    raise ValueError("csv_text, csvText, transactions, items 중 하나를 제공해 주세요.")


def _get_cluster_icon(cluster_id: int) -> str:
    return CLUSTER_ICON_MAP.get(cluster_id, "📊")


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _resolve_tip_copy(category_name: str, gap: float, savable_amount: int) -> tuple[str, str, str]:
    """카테고리별로 바로 읽히는 짧은 행동 문구를 만든다."""
    if category_name == "커피/음료":
        return (
            "커피·음료",
            "커피 구매 횟수부터 먼저 줄여보세요",
            (
                f"커피/음료 비중이 기준보다 {gap:.1f}%p 높습니다. "
                f"이번 달에는 주간 구매 횟수를 먼저 정해두고, 약 {savable_amount:,}원 정도를 아끼는 것을 목표로 잡아보세요."
            ),
        )

    if category_name in {"외식", "음식배달서비스", "패스트푸드", "분식", "육류/회식", "주점"}:
        return (
            "식비",
            f"{category_name} 지출의 빈도부터 조절해보세요",
            (
                f"{category_name} 비중이 기준보다 {gap:.1f}%p 높습니다. "
                f"배달·외식 횟수나 추가 주문 기준을 먼저 정하면 약 {savable_amount:,}원 정도의 절감 여지를 만들 수 있습니다."
            ),
        )

    if category_name == "서적/도서":
        return (
            "책 구매",
            "읽을 책만 남기고 구매 순서를 정해보세요",
            (
                f"서적/도서 지출이 기준보다 {gap:.1f}%p 높습니다. "
                f"바로 읽을 책 1권만 먼저 고르고 나머지는 보류하면 약 {savable_amount:,}원 정도를 줄이는 데 도움이 됩니다."
            ),
        )

    if category_name == "사무/교육용품":
        return (
            "학습 준비물",
            "사무·교육용품은 재고부터 확인해보세요",
            (
                f"사무/교육용품 비중이 기준보다 {gap:.1f}%p 높습니다. "
                f"이미 가진 물건을 먼저 쓰고, 꼭 필요한 품목만 사면 약 {savable_amount:,}원 정도를 줄일 수 있습니다."
            ),
        )

    if category_name == "교육/학원":
        return (
            "교육비",
            "교육비는 실제 이용 빈도부터 점검해보세요",
            (
                f"교육/학원 지출이 기준보다 {gap:.1f}%p 높습니다. "
                f"정기 결제 중 실제로 활용하지 않는 항목이 없는지 먼저 점검하면 약 {savable_amount:,}원 정도를 아낄 수 있습니다."
            ),
        )

    if category_name in {"인터넷쇼핑", "의복/의류", "패션잡화"}:
        return (
            "쇼핑",
            "온라인 장바구니를 하루만 더 묵혀보세요",
            (
                f"{category_name} 비중이 기준보다 {gap:.1f}%p 높습니다. "
                f"즉시 결제 대신 하루만 더 보류해도 충동구매를 줄여 약 {savable_amount:,}원 정도 절약할 수 있습니다."
            ),
        )

    if category_name in {"숙박", "여행/유학대행"}:
        return (
            "여행·숙박",
            "여행·숙박 지출은 이번 달 예산 상한을 먼저 정해보세요",
            (
                f"{category_name} 비중이 기준보다 {gap:.1f}%p 높습니다. "
                f"건별 예산 상한을 먼저 정하면 큰 지출 한 번이 흔들리는 것을 막고 약 {savable_amount:,}원 정도를 조절할 수 있습니다."
            ),
        )

    return (
        category_name,
        f"{category_name} 지출부터 먼저 다듬어보세요",
        (
            f"{category_name} 비중이 기준보다 {gap:.1f}%p 높습니다. "
            f"이번 달에는 이 항목에서 약 {savable_amount:,}원 정도를 줄이는 것을 1차 목표로 두는 편이 가장 효율적입니다."
        ),
    )


def _build_action_tip(category_name: str, savable_amount: int) -> str:
    """goal.action_tip과 소비 날씨 설명에서 함께 쓸 한 문장 행동 제안."""
    if category_name == "커피/음료":
        return (
            f"커피/음료는 단가보다 빈도가 지출을 키우기 쉬운 항목입니다. "
            f"이번 달에는 주간 구매 횟수를 먼저 정해 약 {savable_amount:,}원 정도를 줄여보세요."
        )

    if category_name in {"외식", "음식배달서비스", "패스트푸드", "분식", "육류/회식", "주점"}:
        return (
            f"{category_name}는 한 번의 금액보다 반복 빈도가 누적되기 쉬운 항목입니다. "
            f"외식·배달 횟수 기준을 먼저 정해 약 {savable_amount:,}원 정도를 조절해보세요."
        )

    if category_name == "서적/도서":
        return (
            f"서적/도서는 '바로 읽을 것만 산다'는 기준 하나만 세워도 지출이 안정됩니다. "
            f"이번 달에는 약 {savable_amount:,}원 정도를 줄이는 흐름을 만들어보세요."
        )

    if category_name == "사무/교육용품":
        return (
            f"사무/교육용품은 재고 확인만 해도 중복 구매를 꽤 줄일 수 있습니다. "
            f"이미 가진 물건을 먼저 쓰는 기준으로 약 {savable_amount:,}원 정도를 아껴보세요."
        )

    if category_name == "교육/학원":
        return (
            f"교육/학원은 고정비 성격이 강하니 실제 이용 중인 항목부터 점검해보세요. "
            f"불필요한 결제만 정리해도 약 {savable_amount:,}원 정도 절감할 수 있습니다."
        )

    if category_name in {"숙박", "여행/유학대행"}:
        return (
            f"{category_name} 지출은 건당 금액이 커서 예산 상한을 먼저 정하는 게 효과적입니다. "
            f"이번 달에는 약 {savable_amount:,}원 정도를 조절하는 흐름을 목표로 해보세요."
        )

    return (
        f"{category_name}부터 관리해보세요. 현재 비중이 기준보다 높아 절감 여지가 크고, "
        f"약 {savable_amount:,}원 정도를 줄일 수 있습니다."
    )


def _build_tip_item(order: int, item: dict) -> dict:
    # overspending 상위 항목을 바로 카드형 UI에 붙일 수 있게 정리한다.
    category_name = item["name"]
    gap = round(float(item["myRatio"]) - float(item["baseRatio"]), 1)
    savable_amount = int(item["savableAmount"])
    keyword, title, description = _resolve_tip_copy(category_name, gap, savable_amount)

    return {
        "order": order,
        "keyword": keyword,
        "title": title,
        "description": description,
    }


def _build_goal(summary: dict, overspending: list[dict]) -> dict:
    # goal은 "이번 리포트에서 가장 먼저 뭘 하면 되는지"를 한 문장으로 주는 용도다.
    total_savable = int(summary["totalSavable"])
    expected_spending = int(summary["expectedSpending"])

    if overspending:
        top = overspending[0]
        action_tip = _build_action_tip(top["name"], int(top["savableAmount"]))
    else:
        action_tip = (
            "현재 소비 패턴은 기준과 크게 다르지 않습니다. "
            "무리하게 줄이기보다 지금의 소비 리듬을 유지하면서 큰 지출만 한 번씩 점검해보세요."
        )

    return {
        "savableAmount": total_savable,
        "expectedSpending": expected_spending,
        "actionTip": action_tip,
        "savable_amount": total_savable,
        "expected_spending": expected_spending,
        "action_tip": action_tip,
    }


def _build_weather(summary: dict, overspending: list[dict], categories: list[dict], total_amount: int) -> dict:
    # 소비날씨는 별도 모델이 아니라 리포트 결과를 요약한 표시용 지표다.
    # score가 높을수록 안정적인 소비 패턴으로 본다.
    total_savable = int(summary["totalSavable"])
    savable_ratio = (total_savable / total_amount) if total_amount else 0.0
    overspending_count = len(overspending)
    top_category_ratio = max((float(item["myRatio"]) for item in categories), default=0.0)

    # 절감 가능 금액, 과소비 항목 수, 특정 카테고리 편중을 함께 반영한다.
    score = 100.0
    score -= min(42.0, savable_ratio * 120.0)
    score -= min(30.0, overspending_count * 6.0)
    score -= max(0.0, top_category_ratio - 35.0) * 0.8
    score = _clamp(round(score), 0, 100)

    if score >= 80:
        code, label = "SUNNY", "맑음"
    elif score >= 60:
        code, label = "PARTLY_CLOUDY", "구름조금"
    elif score >= 40:
        code, label = "CLOUDY", "흐림"
    elif score >= 20:
        code, label = "RAINY", "비"
    else:
        code, label = "STORMY", "폭우"

    if not overspending:
        reason = "절감 여지가 크지 않고 소비 분포도 비교적 안정적입니다. 지금의 소비 리듬을 유지해도 좋습니다."
    else:
        top = overspending[0]
        reason = (
            f"{top['name']} 비중이 기준보다 높고 최근 소비 흐름에서 약 {total_savable:,}원 정도의 조절 여지가 보여 "
            f"이번 달 소비 날씨를 {label} 단계로 해석했습니다."
        )

    return {
        "code": code,
        "label": label,
        "score": int(score),
        "reason": reason,
    }


def _get_display_category_name(spending_advisor_module, category: str) -> str:
    # spending_advisor의 표시용 매핑이 깨졌을 때는 원래 45개 카테고리명을 그대로 쓴다.
    display_name = spending_advisor_module.normalize_to_db(category)
    if not display_name:
        return category
    if ("?" in display_name or "\ufffd" in display_name) and ("?" not in category and "\ufffd" not in category):
        return category
    return display_name


def run_analysis(df: pd.DataFrame) -> dict:
    """분석 결과를 백엔드가 바로 쓰기 쉬운 구조로 정리한다."""
    import gmm_predict as _gp
    import spending_advisor as _sa

    user_result = _gp.predict_spending_type(df)
    cluster_id = int(user_result["cluster_id"])
    cluster_name = user_result.get("cluster_name", f"Cluster {cluster_id}")
    cluster_description = user_result.get("cluster_description", "")
    cluster_icon = _get_cluster_icon(cluster_id)

    user_amounts = _sa._user_category_amounts(df)
    total_amount = int(sum(user_amounts.values()))
    reduction_targets = _sa._build_reduction_targets(user_amounts, cluster_id)
    prompt = _sa._build_prompt(user_result, user_amounts, reduction_targets, total_amount)

    cluster_row = _gp._cluster_means.loc[cluster_id] if cluster_id in _gp._cluster_means.index else None
    full_cluster_pct_map: dict[str, float] = {}
    if cluster_row is not None:
        for feat_col, value in cluster_row.items():
            category = str(feat_col).replace("비율_", "")
            full_cluster_pct_map[category] = round(float(value) * 100, 1)

    # categories는 리포트/DB 저장의 기본 상세 목록이고,
    # clusterStats는 사용자 비율 vs 클러스터 기준 비율을 비교하는 내부 요약입니다.
    cluster_stats = []
    categories = []
    for category, amount in sorted(user_amounts.items(), key=lambda item: -item[1]):
        display_name = _get_display_category_name(_sa, category)
        user_pct = round(amount / total_amount * 100, 1) if total_amount else 0.0
        cluster_pct = full_cluster_pct_map.get(category, 0.0)
        diff_pct = round(user_pct - cluster_pct, 1)

        cluster_stats.append(
            {
                "category": display_name,
                "user_amt": int(amount),
                "user_pct": user_pct,
                "cluster_pct": cluster_pct,
                "diff_pct": diff_pct,
            }
        )
        categories.append(
            {
                "name": display_name,
                "amount": int(amount),
                "myRatio": user_pct,
                "baseRatio": cluster_pct,
                "diff": diff_pct,
            }
        )

    # overspending은 "어디서 줄일 수 있는지"만 따로 추린 절감 후보 목록입니다.
    overspending = []
    reduction_summary: dict[str, int] = {}
    for target in reduction_targets:
        display_name = _get_display_category_name(_sa, target["category"])
        savable_amount = int(target["suggested_reduction_amt"])
        reduction_summary[display_name] = reduction_summary.get(display_name, 0) + savable_amount
        overspending.append(
            {
                "name": display_name,
                "myRatio": round(float(target["user_pct"]), 1),
                "baseRatio": round(float(target["cluster_pct"]), 1),
                "savableAmount": savable_amount,
            }
        )

    # summary/goal/weather는 백엔드가 바로 저장하거나 화면에 붙일 수 있는
    # 상위 레벨 리포트 필드입니다.
    total_savable = int(sum(reduction_summary.values()))
    summary = {
        "totalSavable": total_savable,
        "expectedSpending": max(total_amount - total_savable, 0),
    }
    tips = [_build_tip_item(order, item) for order, item in enumerate(overspending[:3], start=1)]
    goal = _build_goal(summary, overspending)
    weather = _build_weather(summary, overspending, categories, total_amount)
    # categoryScheme은 현재 리포트가 어떤 카테고리 축 위에서 계산됐는지
    # 백엔드/프론트가 헷갈리지 않도록 함께 내려준다.
    category_scheme = {
        "type": "gmm_features",
        "version": 1,
        "count": len(_gp.get_available_categories()),
        "categories": _gp.get_available_categories(),
    }

    return {
        "prompt": prompt,
        "reportVersion": "analysis-report-v2",
        "clusterId": cluster_id,
        "clusterName": cluster_name,
        "clusterDescription": cluster_description,
        "clusterIcon": cluster_icon,
        "cluster": {
            "id": cluster_id,
            "name": cluster_name,
            "description": cluster_description,
            "icon": cluster_icon,
        },
        "clusterStats": cluster_stats,
        "categories": categories,
        "overspending": overspending,
        "tips": tips,
        "goal": goal,
        "weather": weather,
        "categoryScheme": category_scheme,
        "reductionSummary": reduction_summary,
        "summary": summary,
        "sourceTransactionCount": int(df["cnt"].sum()),
        "sourceTotalSpending": total_amount,
    }


def _build_history(session_id: str | None, keep_session: bool) -> tuple[str | None, list[dict]]:
    """월간 리포트 생성 시에는 세션 저장을 끌 수 있게 한다."""
    if keep_session:
        current_session_id = session_id or str(uuid.uuid4())
        if current_session_id not in sessions:
            sessions[current_session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        return current_session_id, sessions[current_session_id]

    return None, [{"role": "system", "content": SYSTEM_PROMPT}]


def _append_assistant_reply(history: list[dict], message: str, keep_session: bool) -> None:
    if keep_session:
        history.append({"role": "assistant", "content": message})


def _build_fallback_feedback(analysis: dict, reason: str | None = None) -> str:
    """생성이 실패해도 데모와 백엔드 저장은 이어갈 수 있게 최소 피드백을 만든다."""
    cluster = analysis["cluster"]
    overspending = analysis.get("overspending", [])
    summary = analysis["summary"]

    lines = [
        f"이번 소비 패턴은 {cluster['name']} 유형으로 분석됐습니다.",
        f"현재 절감 가능 금액은 약 {summary['totalSavable']:,}원이며, 예상 지출은 {summary['expectedSpending']:,}원입니다.",
    ]

    if overspending:
        top_targets = ", ".join(item["name"] for item in overspending[:2])
        lines.append(f"우선적으로 조정할 항목은 {top_targets}입니다.")

    lines.append("상세 수치 기반 리포트는 정상 생성됐으니, 우선 상위 과소비 항목부터 점검해보세요.")

    if reason:
        lines.append(f"(자동 생성 피드백 사용: {reason})")

    return " ".join(lines)


def _build_report_response(analysis: dict, session_id: str | None, feedback: str) -> dict:
    # /api/analyze는 백엔드가 바로 저장하기 쉬운 리포트 형태만 내려준다.
    response = {
        "reportVersion": analysis["reportVersion"],
        "cluster_id": analysis["clusterId"],
        "cluster_name": analysis["clusterName"],
        "cluster_description": analysis["clusterDescription"],
        "clusterId": analysis["clusterId"],
        "clusterName": analysis["clusterName"],
        "clusterDescription": analysis["clusterDescription"],
        "cluster": analysis["cluster"],
        "cluster_stats": analysis["clusterStats"],
        "clusterStats": analysis["clusterStats"],
        "categories": [
            {
                "name": item["name"],
                "amount": item["amount"],
                "my_ratio": item["myRatio"],
                "base_ratio": item["baseRatio"],
                "diff": item["diff"],
            }
            for item in analysis["categories"]
        ],
        "overspending": [
            {
                "name": item["name"],
                "my_ratio": item["myRatio"],
                "base_ratio": item["baseRatio"],
                "savable_amount": item["savableAmount"],
            }
            for item in analysis["overspending"]
        ],
        "feedback": feedback,
        "reduction_summary": analysis["reductionSummary"],
        "reductionSummary": analysis["reductionSummary"],
        "summary": {
            "total_savable": analysis["summary"]["totalSavable"],
            "expected_spending": analysis["summary"]["expectedSpending"],
        },
        "tips": analysis["tips"],
        "goal": {
            "savable_amount": analysis["goal"]["savable_amount"],
            "expected_spending": analysis["goal"]["expected_spending"],
            "action_tip": analysis["goal"]["action_tip"],
        },
        "weather": analysis["weather"],
        "categoryScheme": analysis["categoryScheme"],
        "source_transaction_count": analysis["sourceTransactionCount"],
        "source_total_spending": analysis["sourceTotalSpending"],
        "sourceTransactionCount": analysis["sourceTransactionCount"],
        "sourceTotalSpending": analysis["sourceTotalSpending"],
    }

    # 채팅 세션을 이어붙여야 하는 경우에만 session id를 추가로 내려준다.
    if session_id is not None:
        response["session_id"] = session_id
        response["sessionId"] = session_id

    return response


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "active_sessions": len(sessions),
        "activeSessions": len(sessions),
    }


@app.post("/api/analyze")
def analyze():
    body = _get_json_body()
    keep_session = body.get("keepSession", body.get("keep_session", True))
    session_id = _first_non_none(body.get("sessionId"), body.get("session_id"))

    try:
        df = _get_input_df(body)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        analysis = run_analysis(df)
    except Exception as exc:
        return jsonify({"error": f"분석 오류: {exc}"}), 500

    current_session_id, history = _build_history(session_id, bool(keep_session))
    history.append({"role": "user", "content": analysis["prompt"]})
    try:
        feedback = generate(history, max_new_tokens=ANALYZE_MAX_NEW_TOKENS, do_sample=False)
    except Exception as exc:
        app.logger.exception("analyze text generation failed")
        feedback = _build_fallback_feedback(analysis, str(exc))
    _append_assistant_reply(history, feedback, bool(keep_session))

    response = _build_report_response(
        analysis=analysis,
        session_id=current_session_id if bool(keep_session) else None,
        feedback=feedback,
    )
    return jsonify(response)


@app.post("/api/chat")
def chat():
    body = _get_json_body()
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message가 비어 있습니다."}), 400

    session_id = _first_non_none(body.get("sessionId"), body.get("session_id")) or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    sessions[session_id].append({"role": "user", "content": message})
    try:
        reply = generate(sessions[session_id], max_new_tokens=CHAT_MAX_NEW_TOKENS, do_sample=True)
    except Exception as exc:
        app.logger.exception("chat text generation failed")
        return jsonify({"error": f"텍스트 생성 오류: {exc}"}), 500
    sessions[session_id].append({"role": "assistant", "content": reply})

    return jsonify(
        {
            "session_id": session_id,
            "sessionId": session_id,
            "reply": reply,
        }
    )


@app.delete("/api/session/<session_id>")
def delete_session(session_id: str):
    if session_id not in sessions:
        return jsonify({"error": "세션을 찾을 수 없습니다."}), 404

    del sessions[session_id]
    return jsonify({"message": f"세션 {session_id} 삭제됨"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model()
    app.run(host=args.host, port=args.port)
