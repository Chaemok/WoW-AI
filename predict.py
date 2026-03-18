"""
predict.py
────────────────────────────────────────────────────────────
[역할] CSV 거래 내역 → 클러스터 분석 + GMS AI 절감 피드백 한 번에 실행
[전제]
  - model/gmm_model.pkl 존재 (gmm_train.py 먼저 실행)
  - 루트 .env 에 GMS_KEY=<your-gms-key> 설정
[사용]
  python predict.py                                # 더미 데이터
  python predict.py --csv analysis/dummy_transactions.csv
────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# .env 자동 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gmm_predict import predict_spending_type          # noqa: E402
from spending_advisor import analyze_and_advise        # noqa: E402


def run(csv_path: str | None = None, df: pd.DataFrame | None = None) -> None:
    # ── 1. 데이터 로드 ──────────────────────────────────────
    if df is None:
        if csv_path is None:
            raise ValueError("csv_path 또는 df 를 제공하세요.")
        df = pd.read_csv(csv_path)

    # ── 2. 클러스터 예측 ────────────────────────────────────
    result = predict_spending_type(df)
    cid    = result["cluster_id"]
    name   = result["cluster_name"]
    desc   = result["cluster_description"]
    top2   = result["소속확률_top2"]
    probs  = result["전체확률"]

    print("\n" + "═" * 64)
    print(f"  🏷️   소비 유형 : Cluster {cid}  —  {name}")
    print("═" * 64)
    print(f"  {desc}")
    print(f"\n  1위: {top2[0][0]}  ({top2[0][1]}%)")
    print(f"  2위: {top2[1][0]}  ({top2[1][1]}%)")
    print(f"\n  전체 확률: {probs}")

    # ── 3. 카테고리별 현황 + 절감액 + GMS AI 피드백 ────────
    feedback, reduction_dict, cluster_stats = analyze_and_advise(df=df, verbose=True)

    print("\n" + "─" * 72)
    print("  📊  카테고리별 소비 현황 vs 클러스터 기준")
    print("─" * 72)
    print(f"  {'카테고리':16s}  {'지출금액':>10s}  {'내 비율':>7s}  {'기준 비율':>8s}  {'차이':>8s}")
    print("  " + "─" * 58)
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
    if reduction_dict:
        for cat, amt in sorted(reduction_dict.items(), key=lambda x: -x[1]):
            print(f"  {cat:18s}  {amt:>8,}원")
        print(f"  {'합계':18s}  {sum(reduction_dict.values()):>8,}원")
    else:
        print("  클러스터 기준 대비 크게 초과하는 항목이 없습니다.")

    print("\n" + "═" * 64)
    print("  💰  종합 소비 피드백  (Powered by GMS AI)")
    print("═" * 64)
    print(feedback)
    print("═" * 64)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="소비 유형 예측 + GMS AI 절감 피드백",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python predict.py                                    # 더미 데이터
  python predict.py --csv analysis/dummy_transactions.csv
        """,
    )
    parser.add_argument("--csv", type=str, metavar="PATH", help="거래 내역 CSV 파일 경로")
    args = parser.parse_args()

    try:
        if args.csv:
            run(csv_path=args.csv)
        else:
            print("[CSV 미지정 → 더미 데이터로 실행합니다.]\n")
            dummy = pd.read_csv(PROJECT_ROOT / "analysis" / "dummy_transactions.csv")
            run(df=dummy)
    except (EnvironmentError, ImportError, ValueError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        sys.exit(1)
