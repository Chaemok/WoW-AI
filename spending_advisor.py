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
    """cnt > 0 거래만 사용해 카테고리별 실 지출 금액 계산.
    DB 카테고리명이 들어오면 GMM 내부명으로 정규화한 뒤 통폐합 맵 적용.
    """
    valid = df[df["cnt"] > 0].copy()
    # DB 카테고리 → GMM 내부 카테고리 정규화 먼저
    valid["card_tpbuz_nm_2"] = valid["card_tpbuz_nm_2"].map(
        lambda x: _DB_TO_GMM.get(x, x)
    )
    # gmm_predict 의 category_map(통폐합) 적용
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

    # _cluster_means 에서 해당 클러스터의 전체 카테고리 실제 평균 비율 사용
    cluster_row = _gp._cluster_means.loc[cluster_id] if cluster_id in _gp._cluster_means.index else None
    cluster_pct_map: dict[str, float] = {}
    if cluster_row is not None:
        for feat_col, val in cluster_row.items():
            cat = str(feat_col).replace("비율_", "")
            cluster_pct_map[cat] = round(float(val) * 100, 1)

    targets = []
    for cat, amt in sorted(user_amounts.items(), key=lambda x: -x[1]):
        user_pct = round(amt / total_amt * 100, 1)
        cluster_pct = cluster_pct_map.get(cat, 0.0)
        excess = round(user_pct - cluster_pct, 1)

        if excess > excess_threshold_pct:
            # 클러스터 수준까지 줄이는 이상적 절감액 (총지출 기준 %p → 원화 환산)
            ideal_reduction_amt = total_amt * excess / 100
            # 단, 현재 카테고리 지출의 max_reduction_ratio 를 상한으로 적용
            suggested_reduction_amt = round(min(ideal_reduction_amt, amt * max_reduction_ratio))
            suggested_reduction_pct = round(suggested_reduction_amt / total_amt * 100, 1)
            targets.append(
                {
                    "category": cat,
                    "user_pct": user_pct,
                    "cluster_pct": cluster_pct,
                    "user_amt": int(amt),
                    "excess_pct": excess,
                    "suggested_reduction_pct": suggested_reduction_pct,
                    "suggested_reduction_amt": suggested_reduction_amt,
                }
            )

    return sorted(targets, key=lambda x: -x["excess_pct"])



# DB의 expense_category 테이블과 일치하는 공식 카테고리 목록
EXPENSE_CATEGORIES: list[str] = [
    "인터넷쇼핑", "인테리어/가정용품", "교통서비스", "음/식료품소매",
    "외식", "제과/제빵/떡/케익", "커피/음료", "패스트푸드",
    "자동차/유지비", "시스템/통신", "건강/기호식품", "분식",
    "육류/회식", "선물/완구", "병원/의료", "화장품소매",
    "공연관람", "의약/의료품", "건강/뷰티/마사지", "수리서비스",
]

# DB 카테고리 → GMM 내부 카테고리 매핑 (이름이 다른 것만)
_DB_TO_GMM: dict[str, str] = {
    "제과/제빵/떡/케익": "제과/제빵",
    "건강/기호식품":     "건강/기호식품",  # GMM에 없으면 그대로 사용
}

# GMM 내부 카테고리 → DB 카테고리 역매핑 (피드백 표시용)
_GMM_TO_DB: dict[str, str] = {v: k for k, v in _DB_TO_GMM.items() if k != v}


def normalize_to_gmm(category: str) -> str:
    """DB 카테고리명 → GMM 내부 카테고리명으로 변환"""
    return _DB_TO_GMM.get(category, category)


def normalize_to_db(category: str) -> str:
    """GMM 내부 카테고리명 → DB 카테고리명으로 변환 (피드백 표시용)"""
    return _GMM_TO_DB.get(category, category)


def _build_prompt(
    user_result: dict,
    user_amounts: dict[str, float],
    reduction_targets: list[dict],
    total_amt: float,
) -> str:
    """Qwen2.5-14B 에 보낼 분석 프롬프트 생성"""

    cluster_name = user_result.get("cluster_name", "")
    cluster_desc = user_result.get("cluster_description", "")
    top2 = user_result.get("소속확률_top2", [])

    # 사용자 소비 현황 텍스트 (DB 카테고리명으로 표시)
    breakdown_lines = "\n".join(
        f"  - {normalize_to_db(cat)}: {int(amt):,}원 ({round(amt / total_amt * 100, 1)}%)"
        for cat, amt in sorted(user_amounts.items(), key=lambda x: -x[1])
    )

    # 주요 확률 텍스트
    prob_text = ""
    if top2:
        prob_text = f"\n(소속 확률 1위: {top2[0][0]} {top2[0][1]}%)"

    # 절감 대상 텍스트 (DB 카테고리명으로 표시)
    if reduction_targets:
        reduction_lines = "\n".join(
            f"  - {normalize_to_db(t['category'])}: "
            f"사용자 {t['user_pct']}% vs 클러스터 기준 {t['cluster_pct']:.1f}% "
            f"(+{t['excess_pct']}%p 초과) → 약 {t['suggested_reduction_amt']:,}원 절감 가능"
            for t in reduction_targets
        )
    else:
        reduction_lines = "  - 클러스터 평균 대비 크게 초과하는 항목이 없습니다."

    # 공식 카테고리 목록 (AI 참조용)
    category_list = ", ".join(EXPENSE_CATEGORIES)

    prompt = f"""당신은 개인 소비 습관 분석 전문가입니다.
아래 데이터를 바탕으로 이 사람의 소비 생활 전반에 대한 종합적인 피드백을 한국어로 작성해주세요.
항목별 나열이 아니라, 이 사람의 소비 패턴이 어떤 삶의 방식을 반영하는지, 어떤 방향으로 바꿔가면 좋을지를 중심으로 서술해주세요.

---
## 지원 소비 카테고리 (아래 분류명만 사용할 것)
{category_list}

## 사용자 소비 유형
- 클러스터 이름: {cluster_name}{prob_text}
- 유형 설명: {cluster_desc}

## 월별 소비 현황 (총 {int(total_amt):,}원)
{breakdown_lines}

## 클러스터 기준 대비 절감 권장 항목
{reduction_lines}
---

## 작성 요청
아래 구조에 맞춰 마크다운으로 작성해주세요.
카테고리명을 언급할 때는 반드시 위 "지원 소비 카테고리" 목록에 있는 이름을 그대로 사용해주세요.

### 1. 소비 패턴 진단
숫자 나열 없이, 이 사람이 어떤 소비 습관을 가진 사람인지 2~3문장으로 묘사해주세요.
이 소비 패턴이 어떤 생활 방식이나 감정 상태에서 비롯됐을지 추측해서 공감적인 톤으로 써주세요.

### 2. 핵심 개선 방향
개별 항목 하나하나가 아니라, 이 사람에게 가장 효과적인 소비 습관 변화 방향 2~3가지를 제안해주세요.
구체적이고 실천 가능한 행동 변화 중심으로 써주세요.

### 3. 이 유형이 빠지기 쉬운 함정
이 소비 유형에서 반복적으로 나타나는 소비 패턴의 심리적·구조적 원인을 짚어주세요.

### 4. 이번 달 절감 목표
총 절감 가능 예상액과 절감 후 예상 지출을 제시하고,
"이번 달만큼은 이것 하나만 바꿔보세요" 식으로 가장 우선할 행동 한 가지를 마무리로 제안해주세요.
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
    거래 내역 CSV 또는 DataFrame → 클러스터 분석 + GMS AI 절감 피드백

    Parameters
    ----------
    csv_path : CSV 파일 경로 (df 지정 시 무시)
    df       : 직접 전달할 DataFrame
    verbose  : 중간 로그 출력 여부

    Returns
    -------
    tuple[str, dict[str, int], list[dict]]
        - feedback       : GMS AI 가 생성한 마크다운 피드백 텍스트
        - reduction_dict : {카테고리명: 권장_절감액(원)} 딕셔너리
        - cluster_stats  : 카테고리별 사용자 vs 클러스터 기준 비교 리스트
    """
    # 1) 데이터 로드
    if df is None:
        if csv_path is None:
            raise ValueError("csv_path 또는 df 중 하나를 제공해야 합니다.")
        df = _load_csv(csv_path)

    # 2) chatbot.py (Qwen2.5-14B) 에 CSV 전송 → 피드백 수신
    #    chatbot.py 가 GMM 예측 + Qwen 피드백 + 절감 분석을 모두 담당한다.
    chatbot_url = os.environ.get("CHATBOT_URL", "http://localhost:8000")

    if verbose:
        print("🤖  Qwen2.5-14B 피드백 생성 중 (chatbot 서버 호출)...")

    try:
        import requests as _requests
        resp = _requests.post(
            f"{chatbot_url}/api/analyze",
            json={"csv_text": df.to_csv(index=False)},
            timeout=300,
        )
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"chatbot 서버 호출 실패: {exc}\n"
            f"chatbot.py 가 {chatbot_url} 에서 실행 중인지 확인하세요.\n"
            "실행 명령: python3 chatbot.py"
        ) from exc

    data = resp.json()
    feedback: str = data["feedback"]
    reduction_dict: dict[str, int] = data["reduction_summary"]
    cluster_stats: list[dict] = data["cluster_stats"]

    if verbose:
        cluster_name = data.get("cluster_name", "")
        cluster_id = data.get("cluster_id", "")
        print(f"✅  클러스터 {cluster_id}: {cluster_name}\n")

    return feedback, reduction_dict, cluster_stats


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
            feedback, reduction_dict, cluster_stats = analyze_and_advise(df=_DEMO_DF)
        elif args.csv:
            feedback, reduction_dict, cluster_stats = analyze_and_advise(csv_path=args.csv)
        else:
            print("[옵션 미지정 → 더미 데이터로 실행합니다. --csv 로 실제 파일을 지정하세요.]\n")
            feedback, reduction_dict, cluster_stats = analyze_and_advise(df=_DEMO_DF)

        print("\n" + "─" * 72)
        print("  📊  카테고리별 소비 현황 vs 클러스터 기준")
        print("─" * 72)
        print(f"  {'카테고리':16s}  {'지출금액':>10s}  {'내 비율':>7s}  {'기준 비율':>8s}  {'차이':>7s}")
        print("  " + "─" * 68)
        for s in cluster_stats:
            diff_str = f"{s['diff_pct']:+.1f}%p"
            print(
                f"  {s['category']:16s}  {s['user_amt']:>10,}원"
                f"  {s['user_pct']:>6.1f}%"
                f"  {s['cluster_pct']:>7.1f}%"
                f"  {diff_str:>8s}"
            )

        print("\n" + "─" * 72)
        print("  📉  항목별 권장 절감액")
        print("─" * 72)
        for cat, amt in sorted(reduction_dict.items(), key=lambda x: -x[1]):
            print(f"  {cat:18s}  {amt:>8,}원")
        print(f"  {'합계':18s}  {sum(reduction_dict.values()):>8,}원")

        print("\n" + "═" * 64)
        print("  💰  종합 소비 피드백  (Powered by GMS AI)")
        print("═" * 64)
        print(feedback)
        print("═" * 64)

    except (EnvironmentError, ImportError, ValueError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        sys.exit(1)
