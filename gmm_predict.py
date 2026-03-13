"""
gmm_predict.py
────────────────────────────────────────────────────────────
[역할] 저장된 GMM 모델을 로드해서 개인 소비 유형 예측
[전제] gmm_train.py를 먼저 실행해서 model/gmm_model.pkl이 있어야 함
[사용]
    from gmm_predict import predict_spending_type
    result = predict_spending_type(user_df)
────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import joblib
import os
from cluster_definitions import get_cluster_description, get_cluster_name

# ─────────────────────────────────────────────────────────────
# 모델 로드 (모듈 임포트 시 1회만 실행)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'gmm_model.pkl')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"모델 파일이 없습니다: {MODEL_PATH}\n"
        "먼저 `python gmm_train.py` 를 실행해 주세요."
    )

_bundle       = joblib.load(MODEL_PATH)
_gmm          = _bundle.get('gmm')
_cluster_method = _bundle.get('cluster_method', 'gmm')
_cluster_model = _bundle.get('cluster_model')
_cluster_centroids = _bundle.get('cluster_centroids')
_distance_scale = float(_bundle.get('distance_scale', 1.0))
_scaler       = _bundle['scaler']
_pca          = _bundle.get('pca')
_feature_cols = _bundle['feature_cols']
_auto_labels  = _bundle['auto_labels']
_final_labels = _bundle.get('final_labels', {})
_category_map = _bundle['category_map']
_cluster_means = _bundle['cluster_means']
_cluster_summary = {
    item['cluster_id']: item
    for item in _bundle.get('cluster_summary', [])
}

print(f"Model loaded | clusters: {_bundle['best_k']} | features: {len(_feature_cols)}")


def _distance_to_probabilities(distance_matrix: np.ndarray) -> np.ndarray:
    scale = _distance_scale if _distance_scale > 0 else 1.0
    scaled = -distance_matrix / scale
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def _get_cluster_summary(cluster_id: int) -> dict:
    return _cluster_summary.get(cluster_id, {})


def _get_top_features(cluster_id: int) -> list[dict]:
    summary = _get_cluster_summary(cluster_id)
    top_features = summary.get('top_features', [])
    if top_features:
        return top_features

    fallback_features = []
    for rank in range(1, 6):
        category = summary.get(f'top{rank}_category')
        if not category:
            continue
        fallback_features.append({
            'category': category,
            'cluster_pct': float(summary.get(f'top{rank}_cluster_pct', 0.0)),
            'overall_pct': float(summary.get(f'top{rank}_overall_pct', 0.0)),
            'lift_vs_overall': float(summary.get(f'top{rank}_lift_vs_overall', 0.0)),
        })
    return fallback_features


def _build_dynamic_cluster_name(cluster_id: int) -> str:
    top_features = _get_top_features(cluster_id)
    if not top_features:
        return get_cluster_name(cluster_id, _auto_labels[cluster_id])

    distinctive = [
        feature['category']
        for feature in sorted(top_features, key=lambda item: item.get('lift_vs_overall', 0.0), reverse=True)
        if float(feature.get('lift_vs_overall', 0.0)) >= 1.2
    ]
    if len(distinctive) < 2:
        distinctive = [feature['category'] for feature in top_features[:2]]
    if not distinctive:
        return get_cluster_name(cluster_id, _auto_labels[cluster_id])
    return ' · '.join(distinctive[:2]) + ' 중심형'


def _build_dynamic_cluster_description(cluster_id: int) -> str:
    top_features = _get_top_features(cluster_id)
    if not top_features:
        summary = _get_cluster_summary(cluster_id)
        return summary.get('cluster_description', get_cluster_description(cluster_id))

    summary_parts = []
    for feature in sorted(top_features, key=lambda item: item.get('lift_vs_overall', 0.0), reverse=True)[:3]:
        summary_parts.append(
            f"{feature['category']} {float(feature.get('cluster_pct', 0.0)):.1f}% ({float(feature.get('lift_vs_overall', 0.0)):.2f}배)"
        )
    return ', '.join(summary_parts) + ' 비중이 상대적으로 높은 유형'


def get_available_categories() -> list[str]:
    """입력 가능한 최종 카테고리 목록 반환"""
    return [col.replace('비율_', '') for col in _feature_cols]


# ─────────────────────────────────────────────────────────────
# [핵심 함수] 개인 거래 데이터 → 소비 유형 예측
# ─────────────────────────────────────────────────────────────
def predict_spending_type(individual_df: pd.DataFrame) -> dict:
    """
    개인 카드 결제 내역 DataFrame → 소비 유형 클러스터 예측

    Parameters
    ----------
    individual_df : pd.DataFrame
        필수 컬럼:
          - card_tpbuz_nm_2 (str) : 가맹점 소분류명
          - amt             (num) : 결제 금액
          - cnt             (int) : 결제 건수 (0이면 제외)

    Returns
    -------
    dict
        {
          'cluster_id'   : int,          # 클러스터 번호
          'cluster_name' : str,          # 자동 생성 라벨
          '주요_소비_항목': dict,         # {카테고리: '비율%'}
          '소속확률_top2': list[tuple],   # [(라벨, 확률%), ...]
          '전체확률'     : dict,          # {C0: %, C1: %, ...}
        }
    """
    # 1) 기본 필터링
    df = individual_df[individual_df['cnt'] > 0].copy()

    if df.empty:
        raise ValueError("유효한 거래 데이터가 없습니다 (cnt > 0 조건 미충족)")

    # 2) 소분류 정제 (통폐합 & 제외 적용)
    df['refined_category'] = df['card_tpbuz_nm_2'].map(
        lambda x: _category_map.get(x, x)
    )
    df = df[df['refined_category'] != '제외'].copy()

    if df.empty:
        raise ValueError("카테고리 정제 후 유효한 거래가 없습니다.")

    # 3) 카테고리별 지출 비율 계산
    cat_amt = df.groupby('refined_category')['amt'].sum()
    cat_ratio = (cat_amt / cat_amt.sum()).to_dict()

    # 4) 학습 피처 순서에 맞게 벡터 구성 (없는 카테고리는 0)
    feat_vector = np.array(
        [[cat_ratio.get(col.replace('비율_', ''), 0.0) for col in _feature_cols]]
    )
    feat_scaled = _scaler.transform(feat_vector)
    feat_model = _pca.transform(feat_scaled) if _pca is not None else feat_scaled

    # 5) GMM 예측
    if _cluster_method == 'agglomerative':
        distance_matrix = np.linalg.norm(
            feat_model[:, None, :] - _cluster_centroids[None, :, :],
            axis=2,
        )
        probabilities = _distance_to_probabilities(distance_matrix)[0]
        cluster_id = int(distance_matrix.argmin(axis=1)[0])
    else:
        cluster_id = int(_gmm.predict(feat_model)[0])
        probabilities = _gmm.predict_proba(feat_model)[0]

    # 6) 결과 정리
    top2_idx = probabilities.argsort()[::-1][:2]
    top_items = sorted(
        [(col.replace('비율_', ''), v) for col, v in zip(_feature_cols, feat_vector[0]) if v > 0.01],
        key=lambda x: -x[1]
    )[:6]

    cluster_summary = _cluster_summary.get(cluster_id, {})
    dynamic_cluster_name = _build_dynamic_cluster_name(cluster_id)
    dynamic_cluster_description = _build_dynamic_cluster_description(cluster_id)
    fixed_cluster_name = get_cluster_name(cluster_id, _auto_labels[cluster_id])
    fixed_cluster_description = get_cluster_description(cluster_id)

    return {
        'cluster_id':    cluster_id,
        'cluster_name':  dynamic_cluster_name,
        'cluster_name_fixed': fixed_cluster_name,
        'cluster_signature': cluster_summary.get('cluster_signature', _auto_labels[cluster_id]),
        'cluster_description': dynamic_cluster_description,
        'cluster_description_fixed': cluster_summary.get(
            'cluster_description',
            fixed_cluster_description
        ),
        'cluster_headline': cluster_summary.get('headline', ''),
        'cluster_reference_top_features': _get_top_features(cluster_id),
        '주요_소비_항목': {item: f'{v*100:.1f}%' for item, v in top_items},
        '소속확률_top2': [
            (_build_dynamic_cluster_name(int(i)), round(float(probabilities[i]) * 100, 1))
            for i in top2_idx
        ],
        '전체확률': {
            f'C{i}': round(float(p) * 100, 1)
            for i, p in enumerate(probabilities)
        },
    }


def print_result(result: dict):
    """예측 결과를 보기 좋게 출력"""
    print("\n" + "=" * 55)
    print(f"  소비 유형 : Cluster {result['cluster_id']} — {result['cluster_name']}")
    print("=" * 55)
    print(f"  유형 설명 : {result['cluster_description']}")
    if result.get('cluster_name_fixed') and result['cluster_name_fixed'] != result['cluster_name']:
        print(f"  참고 고정 라벨 : {result['cluster_name_fixed']}")
    if result.get('cluster_description_fixed') and result['cluster_description_fixed'] != result['cluster_description']:
        print(f"  참고 기존 설명 : {result['cluster_description_fixed']}")
    print(f"  기준 시그니처 : {result['cluster_signature']}")
    print(f"  🥇 유형 1위 : {result['소속확률_top2'][0][0]}  ({result['소속확률_top2'][0][1]}%)")
    print(f"  🥈 유형 2위 : {result['소속확률_top2'][1][0]}  ({result['소속확률_top2'][1][1]}%)")
    if result['cluster_headline']:
        print(f"  해석 기준 : {result['cluster_headline']}")
    print(f"\n  📊 주요 지출 항목:")
    for cat, ratio in result['주요_소비_항목'].items():
        bar = '█' * int(float(ratio.replace('%', '')) / 2)
        print(f"    {cat:18s} {ratio:>7s}  {bar}")
    if result['cluster_reference_top_features']:
        print(f"\n  📌 클러스터 핵심 특징:")
        for feat in result['cluster_reference_top_features'][:3]:
            print(
                f"    {feat['category']:18s} {feat['cluster_pct']:>6.2f}% | "
                f"전체 {feat['overall_pct']:>6.2f}% | {feat['lift_vs_overall']:.2f}배"
            )
    print(f"\n  전체 클러스터 확률: {result['전체확률']}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────
# 데모 실행 (직접 실행 시)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 예시: 수동으로 만든 가상 개인 거래 내역
    sample_transactions = pd.DataFrame([
        # 외식파 시뮬레이션
        {'card_tpbuz_nm_2': '외식',     'amt': 55000,  'cnt': 4},
        {'card_tpbuz_nm_2': '커피/음료', 'amt': 32000,  'cnt': 8},
        {'card_tpbuz_nm_2': '분식',     'amt': 18000,  'cnt': 3},
        {'card_tpbuz_nm_2': '고기요리', 'amt': 85000,  'cnt': 2},
        {'card_tpbuz_nm_2': '패스트푸드','amt': 14000,  'cnt': 3},
        {'card_tpbuz_nm_2': '일반병원', 'amt': 25000,  'cnt': 1},
        {'card_tpbuz_nm_2': '의약/의료품','amt': 8000,  'cnt': 2},
        {'card_tpbuz_nm_2': '종합소매점','amt': 45000,  'cnt': 5},  # → 제외됨
    ])

    print("\n[데모] 가상 개인 거래 내역으로 소비 유형 예측")
    result = predict_spending_type(sample_transactions)
    print_result(result)
