"""
소비 유형 분석 챗봇 API 서버
────────────────────────────────────────────────────────────────
[역할]
  - Qwen2.5-14B-Instruct (로컬 GPU)로 대화
  - data_analysis 레포의 GMM 모델 연동
  - Flask HTTP 서버로 외부 클라이언트에 서빙

[실행]
  python chatbot.py            (기본: 0.0.0.0:8000)
  python chatbot.py --port 9000

[전제]
  model/gmm_model.pkl 이 있어야 함 (gmm_train.py 먼저 실행)

[엔드포인트]
  GET  /health                 서버 상태 확인
  POST /api/analyze            CSV 소비 분석 + AI 피드백
  POST /api/chat               세션 기반 자유 대화
  DELETE /api/session/{id}     세션 삭제
────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import sys
import os
import uuid
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from flask import Flask, request, jsonify

# 레포 루트 (chatbot.py 와 같은 디렉토리)
DATA_ANALYSIS_ROOT = Path(__file__).resolve().parent
if str(DATA_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ANALYSIS_ROOT))

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "당신은 한국어로 대화하는 개인 소비 습관 분석 전문가 챗봇입니다. "
    "사용자의 소비 데이터를 분석하고, 소비 패턴 진단과 절감 방향을 친절하고 구체적으로 안내해주세요. "
    "분석 결과가 있을 경우 그 맥락을 유지하며 대화합니다."
)

# 세션 스토어 (메모리 내 대화 히스토리)
sessions: dict[str, list[dict]] = {}

# 모델 전역 변수
tokenizer = None
model = None

# ─────────────────────────────────────────────────────────────
# Flask 앱
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


# ─────────────────────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────────────────────
def load_model():
    global tokenizer, model
    print(f"[INFO] 디바이스: {DEVICE}")
    if DEVICE == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"[INFO] 모델 로딩 중: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("[INFO] 모델 로딩 완료!")


# ─────────────────────────────────────────────────────────────
# Qwen 추론
# ─────────────────────────────────────────────────────────────
def generate(history: list[dict], max_new_tokens: int = 1024) -> str:
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


# ─────────────────────────────────────────────────────────────
# 소비 분석
# ─────────────────────────────────────────────────────────────
_DEMO_DF = pd.DataFrame([
    {"card_tpbuz_nm_2": "외식",          "amt": 85000,  "cnt": 6},
    {"card_tpbuz_nm_2": "커피/음료",      "amt": 42000,  "cnt": 14},
    {"card_tpbuz_nm_2": "육류/회식",      "amt": 120000, "cnt": 3},
    {"card_tpbuz_nm_2": "편의점",         "amt": 35000,  "cnt": 10},
    {"card_tpbuz_nm_2": "온라인쇼핑",     "amt": 98000,  "cnt": 4},
    {"card_tpbuz_nm_2": "음/식료품소매",  "amt": 60000,  "cnt": 4},
    {"card_tpbuz_nm_2": "자동차/유지비",  "amt": 45000,  "cnt": 1},
])

_DUMMY_CSV_PATH = DATA_ANALYSIS_ROOT / "analysis" / "dummy_transactions.csv"


def run_analysis(df: pd.DataFrame) -> tuple[str, int, str, dict, list]:
    """GMM 예측 + 프롬프트 생성. Returns (prompt, cluster_id, cluster_name, reduction_dict, cluster_stats)"""
    import gmm_predict as _gp
    import spending_advisor as _sa

    user_result = _gp.predict_spending_type(df)
    cluster_id = int(user_result["cluster_id"])
    cluster_name = user_result.get("cluster_name", f"Cluster {cluster_id}")

    user_amounts = _sa._user_category_amounts(df)
    total_amt = sum(user_amounts.values())
    reduction_targets = _sa._build_reduction_targets(user_amounts, cluster_id)
    prompt = _sa._build_prompt(user_result, user_amounts, reduction_targets, total_amt)

    cluster_row = (
        _gp._cluster_means.loc[cluster_id]
        if cluster_id in _gp._cluster_means.index else None
    )
    full_cluster_pct_map: dict[str, float] = {}
    if cluster_row is not None:
        for feat_col, val in cluster_row.items():
            cat = str(feat_col).replace("비율_", "")
            full_cluster_pct_map[cat] = round(float(val) * 100, 1)

    cluster_stats = sorted(
        [
            {
                "category": cat,
                "user_amt": int(amt),
                "user_pct": round(amt / total_amt * 100, 1),
                "cluster_pct": full_cluster_pct_map.get(cat, 0.0),
                "diff_pct": round(amt / total_amt * 100 - full_cluster_pct_map.get(cat, 0.0), 1),
            }
            for cat, amt in user_amounts.items()
        ],
        key=lambda x: -x["user_amt"],
    )

    reduction_dict = {
        t["category"]: t["suggested_reduction_amt"] for t in reduction_targets
    }

    return prompt, cluster_id, cluster_name, reduction_dict, cluster_stats


# ─────────────────────────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "active_sessions": len(sessions),
    }


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True) or {}
    demo = body.get("demo", False)
    csv_text = body.get("csv_text")
    sid = body.get("session_id") or str(uuid.uuid4())

    # 데이터 로드
    if demo:
        if _DUMMY_CSV_PATH.exists():
            df = pd.read_csv(_DUMMY_CSV_PATH)
        else:
            df = _DEMO_DF.copy()
    elif csv_text:
        import io
        try:
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception as e:
            return jsonify({"error": f"CSV 파싱 오류: {e}"}), 400
        required = {"card_tpbuz_nm_2", "amt", "cnt"}
        missing = required - set(df.columns)
        if missing:
            return jsonify({"error": f"필수 컬럼 누락: {', '.join(sorted(missing))}"}), 400
    else:
        return jsonify({"error": "csv_text 또는 demo=true 중 하나를 제공하세요."}), 400

    try:
        prompt, cluster_id, cluster_name, reduction_dict, cluster_stats = run_analysis(df)
    except Exception as e:
        return jsonify({"error": f"분석 오류: {e}"}), 500

    # 세션 생성 또는 이어가기
    if sid not in sessions:
        sessions[sid] = [{"role": "system", "content": SYSTEM_PROMPT}]

    sessions[sid].append({"role": "user", "content": prompt})
    feedback = generate(sessions[sid], max_new_tokens=1024)
    sessions[sid].append({"role": "assistant", "content": feedback})

    return jsonify({
        "session_id": sid,
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "feedback": feedback,
        "cluster_stats": cluster_stats,
        "reduction_summary": reduction_dict,
        "total_reduction": sum(reduction_dict.values()),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message가 비어있습니다."}), 400

    sid = body.get("session_id") or str(uuid.uuid4())
    if sid not in sessions:
        sessions[sid] = [{"role": "system", "content": SYSTEM_PROMPT}]

    sessions[sid].append({"role": "user", "content": message})
    reply = generate(sessions[sid], max_new_tokens=512)
    sessions[sid].append({"role": "assistant", "content": reply})

    return jsonify({"session_id": sid, "reply": reply})


@app.route("/api/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    if session_id not in sessions:
        return jsonify({"error": "세션을 찾을 수 없습니다."}), 404
    del sessions[session_id]
    return jsonify({"message": f"세션 {session_id} 삭제됨"})


# ─────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model()
    app.run(host=args.host, port=args.port)

