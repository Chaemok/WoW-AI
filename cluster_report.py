"""
cluster_report.py
────────────────────────────────────────────────────────────
[역할] 저장된 GMM 모델 기준으로 클러스터 해석 리포트 생성
[전제] model/gmm_model.pkl 이 존재해야 함
[실행] python cluster_report.py
[출력]
    - model/cluster_summary.csv
    - model/cluster_summary.json
    - model/cluster_summary.md
────────────────────────────────────────────────────────────
"""

import json
import os
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from cluster_definitions import get_cluster_description, get_cluster_name

warnings.filterwarnings('ignore', category=InconsistentVersionWarning)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'gmm_model.pkl')


def persist_bundle_metadata(bundle: dict, summary: list[dict]) -> None:
    """고정 라벨/해석 정보를 모델 번들에 다시 저장"""
    bundle['final_labels'] = {
        int(item['cluster_id']): item['cluster_name']
        for item in summary
    }
    bundle['cluster_summary'] = summary
    joblib.dump(bundle, MODEL_PATH)


def build_cluster_summary(bundle: dict, top_n: int = 5) -> list[dict]:
    """모델 번들에서 클러스터 해석용 요약 생성"""
    cluster_means = bundle['cluster_means']
    auto_labels = bundle['auto_labels']
    overall_means = bundle.get('overall_feature_means', cluster_means.mean())
    cluster_sizes = bundle.get('cluster_sizes')

    summary = []
    for cid in sorted(cluster_means.index):
        row = cluster_means.loc[cid].sort_values(ascending=False)
        peer_means = cluster_means.drop(index=cid, errors='ignore')
        top_features = []

        for feature, value in row.head(top_n).items():
            overall_value = float(overall_means.get(feature, 0.0))
            peer_series = peer_means[feature] if not peer_means.empty else pd.Series(dtype=float)
            peer_max = float(peer_series.max()) if not peer_series.empty else 0.0
            peer_mean = float(peer_series.mean()) if not peer_series.empty else 0.0

            top_features.append({
                'feature': feature,
                'category': feature.replace('비율_', ''),
                'cluster_pct': round(float(value) * 100, 2),
                'overall_pct': round(overall_value * 100, 2),
                'peer_avg_pct': round(peer_mean * 100, 2),
                'peer_max_pct': round(peer_max * 100, 2),
                'lift_vs_overall': round(float(value / overall_value), 2) if overall_value > 0 else None,
                'gap_vs_overall_pctp': round((float(value) - overall_value) * 100, 2),
                'gap_vs_peer_max_pctp': round((float(value) - peer_max) * 100, 2),
            })

        headline_parts = []
        for item in top_features[:2]:
            headline_parts.append(
                f"{item['category']} {item['cluster_pct']:.1f}% "
                f"(전체 평균 {item['overall_pct']:.1f}%, {item['lift_vs_overall']:.2f}배, "
                f"타 클러스터 최고 대비 {item['gap_vs_peer_max_pctp']:+.1f}%p)"
            )

        profile_count = None
        profile_share_pct = None
        if cluster_sizes is not None:
            total_size = int(cluster_sizes.sum())
            profile_count = int(cluster_sizes.get(cid, 0))
            profile_share_pct = round(profile_count / total_size * 100, 2) if total_size else 0.0

        summary.append({
            'cluster_id': int(cid),
            'cluster_name': get_cluster_name(cid, auto_labels[cid]),
            'cluster_signature': auto_labels[cid],
            'cluster_description': get_cluster_description(cid),
            'profile_count': profile_count,
            'profile_share_pct': profile_share_pct,
            'top_features': top_features,
            'headline': ' / '.join(headline_parts),
        })

    return summary


def save_reports(summary: list[dict]) -> None:
    """CSV/JSON/Markdown 리포트 저장"""
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    os.makedirs(model_dir, exist_ok=True)

    rows = []
    markdown_lines = [
        '# GMM 클러스터 해석 리포트',
        '',
        '- 해석 기준: 각 클러스터의 상위 소비 카테고리와 전체 평균/타 클러스터 대비 차이',
        '',
    ]

    for item in summary:
        row = {
            'cluster_id': item['cluster_id'],
            'cluster_name': item['cluster_name'],
            'cluster_signature': item.get('cluster_signature', ''),
            'cluster_description': item.get('cluster_description', ''),
            'profile_count': item['profile_count'],
            'profile_share_pct': item['profile_share_pct'],
            'headline': item['headline'],
        }

        markdown_lines.append(f"## Cluster {item['cluster_id']} — {item['cluster_name']}")
        markdown_lines.append('')
        if item.get('cluster_signature'):
            markdown_lines.append(f"- 기준 시그니처: {item['cluster_signature']}")
        if item.get('cluster_description'):
            markdown_lines.append(f"- 설명: {item['cluster_description']}")
        if item['profile_count'] is not None:
            markdown_lines.append(
                f"- 프로파일 수: {item['profile_count']}개 ({item['profile_share_pct']:.2f}%)"
            )
        markdown_lines.append(f"- 해석: {item['headline']}")
        markdown_lines.append('')
        markdown_lines.append('| 항목 | 클러스터 비중 | 전체 평균 | 타 클러스터 평균 | 타 클러스터 최고 | 전체 대비 배수 | 최고 대비 차이 |')
        markdown_lines.append('|---|---:|---:|---:|---:|---:|---:|')

        for rank, feat in enumerate(item['top_features'], start=1):
            row[f'top{rank}_category'] = feat['category']
            row[f'top{rank}_cluster_pct'] = feat['cluster_pct']
            row[f'top{rank}_overall_pct'] = feat['overall_pct']
            row[f'top{rank}_peer_avg_pct'] = feat['peer_avg_pct']
            row[f'top{rank}_peer_max_pct'] = feat['peer_max_pct']
            row[f'top{rank}_lift_vs_overall'] = feat['lift_vs_overall']
            row[f'top{rank}_gap_vs_peer_max_pctp'] = feat['gap_vs_peer_max_pctp']

            markdown_lines.append(
                f"| {feat['category']} | {feat['cluster_pct']:.2f}% | {feat['overall_pct']:.2f}% | "
                f"{feat['peer_avg_pct']:.2f}% | {feat['peer_max_pct']:.2f}% | "
                f"{feat['lift_vs_overall']:.2f}배 | {feat['gap_vs_peer_max_pctp']:+.2f}%p |"
            )

        markdown_lines.append('')
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        os.path.join(model_dir, 'cluster_summary.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    with open(os.path.join(model_dir, 'cluster_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(model_dir, 'cluster_summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_lines))


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {MODEL_PATH}\n"
            '먼저 `python gmm_train.py` 를 실행해 주세요.'
        )

    bundle = joblib.load(MODEL_PATH)
    summary = build_cluster_summary(bundle)
    persist_bundle_metadata(bundle, summary)
    save_reports(summary)

    print('\n✅ 클러스터 해석 리포트 생성 완료')
    print('   - model/cluster_summary.csv')
    print('   - model/cluster_summary.json')
    print('   - model/cluster_summary.md')
    print('\n[핵심 해석]')
    for item in summary:
        print(f"  Cluster {item['cluster_id']} | {item['cluster_name']}")
        print(f"    → {item['headline']}")


if __name__ == '__main__':
    main()
