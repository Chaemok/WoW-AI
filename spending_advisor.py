"""
spending_advisor.py
────────────────────────────────────────────────────────────
[역할] GMS_KEY(Gemini API)로 사용자 소비 유형을 분석하고
       클러스터 대비 과소비 항목과 구체적인 절감 피드백을 생성한다.
[전제]
  - model/gmm_model.pkl 이 존재해야 함 (gmm_train.py 먼저 실행)
  - 루트 .env 에 GMS_KEY=<your-gemini-key> 설정
[사용]
  python spending_advisor.py --demo                         # 더미 데이터 테스트
  python spending_advisor.py --csv analysis/dummy_transactions.csv
────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import pandas as pd

# .env 자동 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 없으면 환경변수를 직접 설정해야 함

# 프로젝트 루트가 sys.path 에 있어야 gmm_predict 임포트 가능
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gmm_predict as _gp  # noqa: E402
from gmm_predict import predict_spending_type  # noqa: E402


# ─────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────

def _load_csv(path: str) -> pd.DataFrame:
    """CSV → DataFrame, 필수 컬럼 검증"""
    df = pd.read_csv(path)
    required = {"card_tpbuz_nm_2", "amt", "cnt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {', '.join(sorted(missing))}")
    return df


def _user_category_amounts(df: pd.DataFrame) -> dict[str, float]:
    """cnt > 0 거래만 사용해 카테고리별 실 지출 금액 계산"""
    valid = df[df["cnt"] > 0].copy()
    # gmm_predict 의 category_map(통폐합) 동일 적용
    valid = valid.copy()
    valid["refined_category"] = valid["card_tpbuz_nm_2"].map(
        lambda x: _gp._category_map.get(x, x)
    )
    valid = valid[valid["refined_category"] != "제외"]
    return valid.groupby("refined_category")["amt"].sum().to_dict()


def _build_reduction_targets(
    user_amounts: dict[str, float],
    cluster_id: int,
    excess_threshold_pct: float = 3.0,
    max_reduction_ratio: float = 0.30,
) -> list[dict]:
    """
    사용자 카테고리 비율 vs 클러스터 기준 비율 비교 →
    과소비 항목과 권장 절감액 리스트 반환

    Parameters
    ----------
    excess_threshold_pct : 클러스터 평균보다 이 %p 이상 초과해야 절감 대상으로 표시
    max_reduction_ratio  : 제안 절감액은 현재 지출의 이 비율을 넘지 않는다
    """
    total_amt = sum(user_amounts.values())
    if total_amt == 0:
        return []

    cluster_info = _gp._cluster_summary.get(cluster_id, {})
    cluster_pct_map: dict[str, float] = {
        feat["category"]: float(feat["cluster_pct"])
        for feat in cluster_info.get("top_features", [])
    }

    targets = []
    for cat, amt in sorted(user_amounts.items(), key=lambda x: -x[1]):
        user_pct = round(amt / total_amt * 100, 1)
        cluster_pct = cluster_pct_map.get(cat, 0.0)
        excess = round(user_pct - cluster_pct, 1)

        if excess > excess_threshold_pct:
            # 초과분만큼 줄이되 현재 지출의 max_reduction_ratio 상한
            suggested_reduction_pct = min(excess, user_pct * max_reduction_ratio)
            suggested_reduction_amt = round(amt * suggested_reduction_pct / 100)
            targets.append(
                {
                    "category": cat,
                    "user_pct": user_pct,
                    "cluster_pct": cluster_pct,
                    "user_amt": int(amt),
                    "excess_pct": excess,
                    "suggested_reduction_pct": round(suggested_reduction_pct, 1),
                    "suggested_reduction_amt": suggested_reduction_amt,
                }
            )

    return sorted(targets, key=lambda x: -x["excess_pct"])


def _build_prompt(
    user_result: dict,
    user_amounts: dict[str, float],
    reduction_targets: list[dict],
    total_amt: float,
) -> str:
    """Gemini 에 보낼 분석 프롬프트 생성"""

    cluster_name = user_result.get("cluster_name_fixed", user_result.get("cluster_name", ""))
    cluster_desc = user_result.get(
        "cluster_description_fixed", user_result.get("cluster_description", "")
    )
    top2 = user_result.get("소속확률_top2", [])

    # 사용자 소비 현황 텍스트
    breakdown_lines = "\n".join(
        f"  - {cat}: {int(amt):,}원 ({round(amt / total_amt * 100, 1)}%)"
        for cat, amt in sorted(user_amounts.items(), key=lambda x: -x[1])
    )

    # 주요 확률 텍스트
    prob_text = ""
    if top2:
        prob_text = f"\n(소속 확률 1위: {top2[0][0]} {top2[0][1]}%)"

    # 절감 대상 텍스트
    if reduction_targets:
        reduction_lines = "\n".join(
            f"  - {t['category']}: "
            f"사용자 {t['user_pct']}% vs 클러스터 기준 {t['cluster_pct']:.1f}% "
            f"(+{t['excess_pct']}%p 초과) → 약 {t['suggested_reduction_amt']:,}원 절감 가능"
            for t in reduction_targets
        )
    else:
        reduction_lines = "  - 클러스터 평균 대비 크게 초과하는 항목이 없습니다."

    prompt = f"""당신은 개인 소비 습관 분석 전문가입니다.
아래 정보를 바탕으로 구체적인 소비 절감 피드백을 한국어로 작성해주세요.

---
## 사용자 소비 유형
- 클러스터 이름: {cluster_name}{prob_text}
- 유형 설명: {cluster_desc}

## 월별 소비 현황 (총 {int(total_amt):,}원)
{breakdown_lines}

## 클러스터 기준 대비 과소비 의심 항목 (절감 우선순위)
{reduction_lines}
---

## 작성 요청
아래 구조에 맞춰 마크다운 형식으로 피드백을 작성해주세요.

### 1. 소비 유형 요약
이 사람의 소비 패턴과 유형 특성을 2~3문장으로 요약하세요.

### 2. 절감 우선순위 항목
각 과소비 항목에 대해:
- 현재 지출액과 권장 목표 금액
- 절감이 필요한 이유 (클러스터 기준 대비)
- 구체적인 실천 방법 1~2가지

### 3. 이 유형의 소비 함정
이 소비 유형에서 흔히 빠지는 지출 패턴이나 주의점을 경고해주세요.

### 4. 총 절감 가능 예상액
모든 절감 항목의 예상 절감액 합계와 절감 후 예상 총 지출을 명시해주세요.
"""
    return prompt


# ─────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────

def analyze_and_advise(
    csv_path: str | None = None,
    df: pd.DataFrame | None = None,
    verbose: bool = True,
) -> str:
    """
    거래 내역 CSV 또는 DataFrame → 클러스터 분석 + Gemini 절감 피드백

    Parameters
    ----------
    csv_path : CSV 파일 경로 (df 지정 시 무시)
    df       : 직접 전달할 DataFrame
    verbose  : 중간 로그 출력 여부

    Returns
    -------
    str : Gemini 가 생성한 마크다운 피드백 텍스트
    """
    # GMS_KEY 확인
    gms_key = os.environ.get("GMS_KEY", "").strip()
    if not gms_key:
        raise EnvironmentError(
            "GMS_KEY 환경변수가 설정되지 않았습니다.\n"
            "루트 .env 파일에 GMS_KEY=<your-gms-key> 를 추가하거나\n"
            "환경변수를 직접 설정하세요."
        )

    # openai 임포트 (설치 확인)
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai 패키지가 없습니다.\n"
            "pip install openai 를 실행하세요."
        )

    # 1) 데이터 로드
    if df is None:
        if csv_path is None:
            raise ValueError("csv_path 또는 df 중 하나를 제공해야 합니다.")
        df = _load_csv(csv_path)

    # 2) 클러스터 예측
    if verbose:
        print("🔍  소비 유형 클러스터 분석 중...")
    user_result = predict_spending_type(df)
    cluster_id = user_result["cluster_id"]

    if verbose:
        print(f"✅  클러스터 {cluster_id}: {user_result.get('cluster_name_fixed', '')}")
        print(f"    {user_result.get('cluster_description_fixed', '')}\n")

    # 3) 카테고리별 실 지출액 집계 (category_map 동일 적용)
    user_amounts = _user_category_amounts(df)
    total_amt = sum(user_amounts.values())

    # 4) 절감 대상 분석
    reduction_targets = _build_reduction_targets(user_amounts, cluster_id)

    if verbose and reduction_targets:
        print("📊  과소비 항목 (클러스터 기준 초과):")
        for t in reduction_targets:
            print(
                f"    {t['category']:18s} "
                f"{t['user_pct']:>5.1f}% vs 기준 {t['cluster_pct']:>5.1f}% "
                f"→ 약 {t['suggested_reduction_amt']:,}원 절감 가능"
            )
        print()

    # 5) Gemini 프롬프트 생성 및 호출
    prompt = _build_prompt(user_result, user_amounts, reduction_targets, total_amt)

    if verbose:
        print("🤖  GMS AI 피드백 생성 중...")

    client = OpenAI(
        api_key=gms_key,
        base_url="https://gms.ssafy.io/gmsapi/api.openai.com/v1",
    )
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "developer", "content": "Answer in Korean. 당신은 한국어로 답변하는 개인 소비 습관 분석 전문가입니다."},
            {"role": "user", "content": prompt},
        ],
    )
    feedback: str = response.choices[0].message.content

    return feedback


# ─────────────────────────────────────────────────────────────
# 더미 데이터 (--demo 실행용)
# ─────────────────────────────────────────────────────────────

_DEMO_DF = pd.DataFrame(
    [
        {"card_tpbuz_nm_2": "외식",          "amt": 90000,  "cnt": 7},
        {"card_tpbuz_nm_2": "커피/음료",     "amt": 68000,  "cnt": 16},
        {"card_tpbuz_nm_2": "분식",          "amt": 26000,  "cnt": 5},
        {"card_tpbuz_nm_2": "육류/회식",     "amt": 130000, "cnt": 3},
        {"card_tpbuz_nm_2": "편의점",        "amt": 38000,  "cnt": 9},
        {"card_tpbuz_nm_2": "화장품소매",    "amt": 52000,  "cnt": 2},
        {"card_tpbuz_nm_2": "의약/의료품",   "amt": 16000,  "cnt": 2},
        {"card_tpbuz_nm_2": "음/식료품소매", "amt": 60000,  "cnt": 4},
        {"card_tpbuz_nm_2": "자동차/유지비", "amt": 45000,  "cnt": 1},
    ]
)


# ─────────────────────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="소비 유형 분석 + Gemini 절감 피드백",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python spending_advisor.py --demo
  python spending_advisor.py --csv analysis/dummy_transactions.csv
        """,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--csv", type=str, metavar="PATH", help="거래 내역 CSV 파일 경로")
    group.add_argument("--demo", action="store_true", help="더미 데이터로 테스트")
    args = parser.parse_args()

    try:
        if args.demo:
            print("[더미 데이터로 실행합니다.]\n")
            feedback = analyze_and_advise(df=_DEMO_DF)
        elif args.csv:
            feedback = analyze_and_advise(csv_path=args.csv)
        else:
            print("[옵션 미지정 → 더미 데이터로 실행합니다. --csv 로 실제 파일을 지정하세요.]\n")
            feedback = analyze_and_advise(df=_DEMO_DF)

        print("\n" + "═" * 64)
        print("  💰  소비 절감 피드백  (Powered by Gemini)")
        print("═" * 64)
        print(feedback)
        print("═" * 64)

    except (EnvironmentError, ImportError, ValueError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        sys.exit(1)
