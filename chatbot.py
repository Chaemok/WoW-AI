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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# 레포 루트를 파이썬 경로에 추가한다.
DATA_ANALYSIS_ROOT = Path(__file__).resolve().parent
if str(DATA_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ANALYSIS_ROOT))


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


def generate(history: list[dict], max_new_tokens: int) -> str:
    """Qwen 채팅 템플릿을 사용해 응답을 생성한다."""
    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

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

    cluster_stats = []
    categories = []
    for category, amount in sorted(user_amounts.items(), key=lambda item: -item[1]):
        display_name = _sa.normalize_to_db(category)
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

    overspending = []
    reduction_summary: dict[str, int] = {}
    for target in reduction_targets:
        display_name = _sa.normalize_to_db(target["category"])
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

    total_savable = int(sum(reduction_summary.values()))
    summary = {
        "totalSavable": total_savable,
        "expectedSpending": max(total_amount - total_savable, 0),
    }

    return {
        "prompt": prompt,
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
    feedback = generate(history, max_new_tokens=ANALYZE_MAX_NEW_TOKENS)
    _append_assistant_reply(history, feedback, bool(keep_session))

    response = {
        # 기존 응답 키
        "session_id": current_session_id,
        "cluster_id": analysis["clusterId"],
        "cluster_name": analysis["clusterName"],
        "cluster_description": analysis["clusterDescription"],
        "cluster_icon": analysis["clusterIcon"],
        "feedback": feedback,
        "cluster_stats": analysis["clusterStats"],
        "reduction_summary": analysis["reductionSummary"],
        "total_reduction": analysis["summary"]["totalSavable"],
        "source_transaction_count": analysis["sourceTransactionCount"],
        "source_total_spending": analysis["sourceTotalSpending"],
        # 백엔드 연동용 camelCase 키
        "sessionId": current_session_id,
        "clusterId": analysis["clusterId"],
        "clusterName": analysis["clusterName"],
        "clusterDescription": analysis["clusterDescription"],
        "clusterIcon": analysis["clusterIcon"],
        "cluster": analysis["cluster"],
        "clusterStats": analysis["clusterStats"],
        "categories": analysis["categories"],
        "overspending": analysis["overspending"],
        "reductionSummary": analysis["reductionSummary"],
        "summary": analysis["summary"],
        "totalReduction": analysis["summary"]["totalSavable"],
        "sourceTransactionCount": analysis["sourceTransactionCount"],
        "sourceTotalSpending": analysis["sourceTotalSpending"],
    }
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
    reply = generate(sessions[session_id], max_new_tokens=CHAT_MAX_NEW_TOKENS)
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
