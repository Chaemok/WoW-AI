"""
gmm_train.py
────────────────────────────────────────────────────────────
[역할] 카드소비 데이터로 GMM 클러스터링 학습 후 모델 저장
[실행] python gmm_train.py  (최초 1회 or 데이터 갱신 시)
[출력] model/gmm_model.pkl  (예측 시 이 파일만 로드)
────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import zipfile
import os
import json
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from cluster_definitions import FINAL_CLUSTER_DEFINITIONS, get_cluster_description, get_cluster_name
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

FIXED_K = 8
BALANCE_STRATA_COLS = ['age', 'sex', 'top2_category_pattern']
MIN_BALANCED_PER_STRATUM = 2
MAX_BALANCED_PER_STRATUM = 3
BALANCE_RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────
# ① 데이터 로딩 (지역 컬럼 미사용 — 로딩 시에도 제거)
# ─────────────────────────────────────────────────────────────
data_dir = "data"
zip_files = {
    "2507": "카드소비 데이터_202507.zip",
    "2508": "카드소비 데이터_202508.zip",
    "2509": "카드소비 데이터_202509.zip",
    "2510": "카드소비 데이터_202510.zip",
    "2511": "카드소비 데이터_202511.zip",
    "2512": "카드소비 데이터_202512.zip",
}

# 실제로 필요한 컬럼만 로드 → 메모리 절약
LOAD_COLS = ['age', 'sex', 'card_tpbuz_nm_2', 'amt', 'cnt']

all_dfs = []
for month, zipname in zip_files.items():
    zip_path = os.path.join(data_dir, zipname)
    monthly_dfs = []
    with zipfile.ZipFile(zip_path) as zf:
        for filename in sorted(zf.namelist()):
            with zf.open(filename) as f:
                df_city = pd.read_csv(f, usecols=LOAD_COLS)  # 지역 컬럼 자체를 아예 안 읽음
                df_city['month'] = month
                monthly_dfs.append(df_city)
    combined = pd.concat(monthly_dfs, ignore_index=True)
    all_dfs.append(combined)
    print(f"✅ {month} 로드 완료 → shape: {combined.shape}")

df_all = pd.concat(all_dfs, ignore_index=True)
print(f"\n✅ 전체 합치기 완료 → shape: {df_all.shape}")

# ─────────────────────────────────────────────────────────────
# ② 전처리
# ─────────────────────────────────────────────────────────────
df_all = df_all[df_all['cnt'] > 0].copy()
df_all['건당가격'] = df_all['amt'] / df_all['cnt']

def remove_outliers_iqr(group):
    Q1 = group['건당가격'].quantile(0.25)
    Q3 = group['건당가격'].quantile(0.75)
    IQR = Q3 - Q1
    return group[(group['건당가격'] >= Q1 - 1.5*IQR) & (group['건당가격'] <= Q3 + 1.5*IQR)]

before = len(df_all)
df_all = pd.concat(
    [remove_outliers_iqr(group) for _, group in df_all.groupby('card_tpbuz_nm_2')],
    ignore_index=True,
)
print(f"   이상치 제거: {before:,} → {len(df_all):,}행")

# ─────────────────────────────────────────────────────────────
# ③ 소분류 카테고리 정제 & 통폐합
# ─────────────────────────────────────────────────────────────
CATEGORY_MAP = {
    # 제외 항목 (특색 없음 / B2B / 포괄적)
    '종합소매점': '제외',
    '기타결제': '제외', '기타용품': '제외', '기타의료': '제외', '기타교육': '제외',
    '공공기관': '제외', '기업': '제외', '단체': '제외', '종교': '제외',
    '회비/공과금': '제외', '휴게소/대형업체': '제외', '가례서비스': '제외',
    '제조/도매': '제외', '전문서비스': '제외', '광고/인쇄/인화': '제외',
    '금융상품/서비스': '제외', '무점포서비스': '제외', '보안/운송': '제외', '부동산': '제외',

    # 통폐합 항목
    '일반병원': '병원/의료', '종합병원': '병원/의료', '특화병원': '병원/의료',
    '차량관리/부품': '자동차/유지비', '차량관리/서비스': '자동차/유지비',
    '차량판매': '자동차/유지비', '연료판매': '자동차/유지비',
    '유아교육': '교육/학원', '입시학원': '교육/학원', '외국어학원': '교육/학원',
    '기술/직업교육학원': '교육/학원', '예체능계학원': '교육/학원', '독서실/고시원': '교육/학원',
    '유흥주점': '주점', '간이주점': '주점',
    '고기요리': '육류/회식', '닭/오리요리': '육류/회식',
    '한식': '외식',
    '일식/수산물': '외식', '별식/퓨전요리': '외식',
    '양식': '외식', '중식': '외식', '부페': '외식',
    '사우나/휴게시설': '건강/뷰티/마사지', '미용서비스': '건강/뷰티/마사지',
    '요가/단전/마사지': '건강/뷰티/마사지',
}

df_all['refined_category'] = df_all['card_tpbuz_nm_2'].map(lambda x: CATEGORY_MAP.get(x, x))
df_all = df_all[df_all['refined_category'] != '제외'].copy()
print(f"   카테고리 정제 후: {len(df_all):,}행")

# ─────────────────────────────────────────────────────────────
# ④ (month, age, sex) 단위 소비 비율 프로파일 생성
# ─────────────────────────────────────────────────────────────
print("\n피처 피벗 생성 중...")
sub_pivot = (
    df_all.groupby(['month', 'age', 'sex', 'refined_category'])['amt']
    .sum()
    .reset_index()
    .pivot_table(index=['month', 'age', 'sex'], columns='refined_category', values='amt', fill_value=0)
)

sub_ratio = sub_pivot.div(sub_pivot.sum(axis=1), axis=0).round(4)
sub_ratio.columns = [f'비율_{c}' for c in sub_ratio.columns]
sub_ratio = sub_ratio.reset_index()

feature_cols = [c for c in sub_ratio.columns if c.startswith('비율_')]


def add_profile_balance_keys(profile_df, ratio_cols):
    profile_df = profile_df.copy()
    top2_features = profile_df[ratio_cols].apply(
        lambda row: row.sort_values(ascending=False).head(2).index.tolist(),
        axis=1,
    )
    profile_df['dominant_category'] = top2_features.str[0].str.replace('비율_', '', regex=False)
    profile_df['secondary_category'] = top2_features.str[1].str.replace('비율_', '', regex=False)
    profile_df['top2_category_pattern'] = (
        profile_df['dominant_category'] + ' + ' + profile_df['secondary_category']
    )
    profile_df['dominant_ratio'] = profile_df[ratio_cols].max(axis=1).round(4)
    return profile_df


def build_balanced_training_profiles(profile_df):
    stratum_counts = (
        profile_df.groupby(BALANCE_STRATA_COLS, dropna=False)
        .size()
        .rename('original_count')
        .reset_index()
        .sort_values('original_count', ascending=False)
    )
    n_strata = max(len(stratum_counts), 1)
    target_per_stratum = int(
        np.clip(
            np.ceil(len(profile_df) / n_strata),
            MIN_BALANCED_PER_STRATUM,
            MAX_BALANCED_PER_STRATUM,
        )
    )

    sampled_frames = []
    for _, group in profile_df.groupby(BALANCE_STRATA_COLS, dropna=False, sort=False):
        sampled_frames.append(
            group.sample(
                n=target_per_stratum,
                replace=len(group) < target_per_stratum,
                random_state=BALANCE_RANDOM_STATE,
            )
        )

    balanced_df = pd.concat(sampled_frames, ignore_index=True)
    balanced_counts = (
        balanced_df.groupby(BALANCE_STRATA_COLS, dropna=False)
        .size()
        .rename('balanced_count')
        .reset_index()
    )
    balance_report = stratum_counts.merge(balanced_counts, on=BALANCE_STRATA_COLS, how='left')
    balance_report['balanced_count'] = balance_report['balanced_count'].fillna(0).astype(int)
    balance_report['sampling_action'] = np.where(
        balance_report['original_count'] < target_per_stratum,
        'upsampled',
        np.where(
            balance_report['original_count'] > target_per_stratum,
            'downsampled',
            'kept',
        ),
    )
    return balanced_df, balance_report, target_per_stratum


sub_ratio = add_profile_balance_keys(sub_ratio, feature_cols)
balanced_sub_ratio, balance_report, balance_target_per_stratum = build_balanced_training_profiles(sub_ratio)

print(f"📊 피처 수: {len(feature_cols)}개 | 프로파일 수: {len(sub_ratio)}개")
print(f"   프로파일 단위: month + age + sex | 고정 클러스터 수: {FIXED_K}")
print(
    f"   균형 strata 기준: {' + '.join(BALANCE_STRATA_COLS)} | "
    f"strata {len(balance_report)}개 | strata당 목표 {balance_target_per_stratum}개"
)
print(
    f"   균형 학습 프로파일: {len(sub_ratio)}개 → {len(balanced_sub_ratio)}개 "
    f"(up {int((balance_report['sampling_action'] == 'upsampled').sum())} / "
    f"down {int((balance_report['sampling_action'] == 'downsampled').sum())})"
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(balanced_sub_ratio[feature_cols].values)
X_full_scaled = scaler.transform(sub_ratio[feature_cols].values)
X_train_model = X_train_scaled
X_full_model = X_full_scaled


def compute_centroids(features, labels, cluster_ids):
    return np.vstack([features[labels == cid].mean(axis=0) for cid in cluster_ids])


def compute_distance_matrix(features, centroids):
    return np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)


def distance_to_probabilities(distance_matrix, distance_scale):
    scaled = -distance_matrix / distance_scale
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

# ─────────────────────────────────────────────────────────────
# ⑤ 고정 K=8로 GMM 학습
# ─────────────────────────────────────────────────────────────
n_samples = len(balanced_sub_ratio)
if n_samples < FIXED_K:
    raise ValueError(
        f"고정 클러스터 수 {FIXED_K}개를 학습하기에 프로파일 수가 부족합니다: {n_samples}개"
    )

best_k = FIXED_K
cluster_model = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
train_cluster_labels = cluster_model.fit_predict(X_train_model)
cluster_ids = np.arange(best_k)
train_centroids = compute_centroids(X_train_model, train_cluster_labels, cluster_ids)
full_distance_matrix = compute_distance_matrix(X_full_model, train_centroids)
full_cluster_labels = full_distance_matrix.argmin(axis=1)
distance_scale = float(np.median(full_distance_matrix[full_distance_matrix > 0])) if np.any(full_distance_matrix > 0) else 1.0
if distance_scale <= 0:
    distance_scale = 1.0
full_probabilities = distance_to_probabilities(full_distance_matrix, distance_scale)
silhouette = silhouette_score(X_full_model, full_cluster_labels) if len(np.unique(full_cluster_labels)) > 1 else float('nan')
top2_cluster_share_pct = (
    pd.Series(full_cluster_labels).value_counts(normalize=True).sort_values(ascending=False).head(2).sum() * 100
)

print(f"\n[고정 K 학습] K = {best_k}")
print(f"  silhouette: {silhouette:10.4f}")
print(f"  top2 share: {top2_cluster_share_pct:10.2f}%")

os.makedirs("model", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(['Silhouette'], [silhouette], color='royalblue')
axes[0].set_title(f'Agglomerative K={best_k} Silhouette')
axes[1].bar(['Top2 Share'], [top2_cluster_share_pct], color='tomato')
axes[1].set_title(f'Agglomerative K={best_k} Top2 Share')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}%'))
plt.suptitle('Agglomerative 고정 K 진단', fontsize=13)
plt.tight_layout()
plt.savefig('model/gmm_fixed_k_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ─────────────────────────────────────────────────────────────
# ⑥ 최종 Agglomerative 할당
# ─────────────────────────────────────────────────────────────
sub_ratio['cluster'] = full_cluster_labels

# 클러스터 자동 라벨링
cluster_means = sub_ratio.groupby('cluster')[feature_cols].mean()

def make_behavior_label(row, top_n=2):
    top = row.nlargest(top_n)
    return ' · '.join([f"{c.replace('비율_', '')}({v*100:.0f}%)" for c, v in top.items()])

AUTO_LABELS = {i: make_behavior_label(cluster_means.loc[i]) for i in range(best_k)}
FINAL_LABELS = {i: get_cluster_name(i, AUTO_LABELS[i]) for i in range(best_k)}
print("\n[클러스터 라벨]")
for k, v in AUTO_LABELS.items():
    print(f"  Cluster {k}: {FINAL_LABELS[k]} | 기준 시그니처: {v}")


def build_cluster_summary(sub_ratio_df, cluster_means_df, labels, top_n=5):
    """클러스터별 해석용 요약 자료 생성"""
    overall_means = sub_ratio_df[feature_cols].mean()
    cluster_sizes = sub_ratio_df['cluster'].value_counts().sort_index()
    total_profiles = int(len(sub_ratio_df))
    summary = []

    for cid in sorted(cluster_means_df.index):
        row = cluster_means_df.loc[cid].sort_values(ascending=False)
        peer_means = cluster_means_df.drop(index=cid, errors='ignore')
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

        summary.append({
            'cluster_id': int(cid),
            'cluster_name': labels[cid],
            'cluster_signature': AUTO_LABELS[cid],
            'cluster_description': get_cluster_description(cid),
            'profile_count': int(cluster_sizes.get(cid, 0)),
            'profile_share_pct': round(float(cluster_sizes.get(cid, 0)) / total_profiles * 100, 2),
            'top_features': top_features,
            'headline': ' / '.join(headline_parts),
        })

    return summary, overall_means, cluster_sizes


def save_cluster_reports(summary, sub_ratio_df, probability_cols):
    """클러스터 해석 보고서 저장"""
    summary_rows = []
    markdown_lines = [
        '# Agglomerative 클러스터 해석 리포트',
        '',
        f'- 기준 단위: (month, age, sex) 소비 프로파일 {len(sub_ratio_df)}개',
        f'- 최종 클러스터 수: {best_k}개',
        '- 해석 기준: 각 클러스터의 상위 소비 카테고리와 전체 평균/타 클러스터 대비 차이',
        '',
    ]

    for item in summary:
        row = {
            'cluster_id': item['cluster_id'],
            'cluster_name': item['cluster_name'],
            'cluster_signature': item['cluster_signature'],
            'cluster_description': item['cluster_description'],
            'profile_count': item['profile_count'],
            'profile_share_pct': item['profile_share_pct'],
            'headline': item['headline'],
        }

        markdown_lines.extend([
            f"## Cluster {item['cluster_id']} — {item['cluster_name']}",
            '',
            f"- 기준 시그니처: {item['cluster_signature']}",
            f"- 설명: {item['cluster_description']}",
            f"- 프로파일 수: {item['profile_count']}개 ({item['profile_share_pct']:.2f}%)",
            f"- 해석: {item['headline']}",
            '',
            '| 항목 | 클러스터 비중 | 전체 평균 | 타 클러스터 평균 | 타 클러스터 최고 | 전체 대비 배수 | 최고 대비 차이 |',
            '|---|---:|---:|---:|---:|---:|---:|',
        ])

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
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv('model/cluster_summary.csv', index=False, encoding='utf-8-sig')

    with open('model/cluster_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open('model/cluster_summary.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_lines))

    assignment_cols = ['month', 'age', 'sex', 'cluster', 'cluster_name'] + probability_cols
    sub_ratio_df[assignment_cols].sort_values(['cluster', 'month', 'age', 'sex']).to_csv(
        'model/cluster_assignments.csv',
        index=False,
        encoding='utf-8-sig'
    )

    print("\n✅ 클러스터 해석 자료 저장 완료")
    print("   - model/cluster_summary.csv")
    print("   - model/cluster_summary.json")
    print("   - model/cluster_summary.md")
    print("   - model/cluster_assignments.csv")


balance_report.to_csv('model/balance_profile_summary.csv', index=False, encoding='utf-8-sig')
print("   - model/balance_profile_summary.csv")

# ─────────────────────────────────────────────────────────────
# ⑦ 시각화: 클러스터별 소비 항목 Top8 바차트
# ─────────────────────────────────────────────────────────────
n_cols = min(best_k, 4)
n_rows = (best_k + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = np.array(axes).flatten()
colors = plt.cm.tab10.colors

for cid in range(best_k):
    ax = axes[cid]
    top8 = cluster_means.loc[cid].nlargest(8)
    labels = [c.replace('비율_', '') for c in top8.index]
    vals = top8.values * 100
    bars = ax.barh(labels[::-1], vals[::-1],
                   color=[colors[cid % len(colors)]] * len(labels),
                   edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
    ax.set_title(f'Cluster {cid}\n{FINAL_LABELS[cid]}\n{AUTO_LABELS[cid]}', fontsize=10, fontweight='bold')
    ax.set_xlabel('지출 비율 (%)')
    ax.set_xlim(0, max(vals) * 1.25 + 3)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

for i in range(best_k, len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Agglomerative 클러스터별 주요 소비 항목 Top8\n(순수 소비 행동 유형)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('model/cluster_top8.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ─────────────────────────────────────────────────────────────
# ⑧ 클러스터 해석 자료 생성/저장
# ─────────────────────────────────────────────────────────────
proba_cols = [f'prob_C{i}' for i in range(best_k)]
sub_ratio[proba_cols] = np.round(full_probabilities, 4)
sub_ratio['cluster_name'] = sub_ratio['cluster'].map(FINAL_LABELS)

cluster_summary, overall_feature_means, cluster_sizes = build_cluster_summary(
    sub_ratio_df=sub_ratio,
    cluster_means_df=cluster_means,
    labels=FINAL_LABELS,
    top_n=5,
)
save_cluster_reports(cluster_summary, sub_ratio, proba_cols)

print("\n[클러스터 해석 핵심]")
for item in cluster_summary:
    print(
        f"  Cluster {item['cluster_id']} ({item['cluster_name']}) | "
        f"프로파일 {item['profile_count']}개 ({item['profile_share_pct']:.1f}%)"
    )
    print(f"    기준 시그니처: {item['cluster_signature']}")
    print(f"    → {item['headline']}")

# ─────────────────────────────────────────────────────────────
# ⑨ 모델 저장 (예측 시 이 파일만 로드하면 됨)
# ─────────────────────────────────────────────────────────────
model_bundle = {
    'cluster_method': 'agglomerative',
    'gmm':          None,
    'cluster_model': cluster_model,
    'cluster_centroids': train_centroids,
    'distance_scale': distance_scale,
    'scaler':       scaler,        # StandardScaler
    'pca':          None,
    'feature_cols': feature_cols,  # 피처 컬럼 목록 (순서 중요)
    'auto_labels':  AUTO_LABELS,   # {cluster_id: 라벨문자열}
    'final_labels': FINAL_LABELS,  # {cluster_id: 고정 라벨}
    'final_definitions': FINAL_CLUSTER_DEFINITIONS,
    'category_map': CATEGORY_MAP,  # 소분류 통폐합 딕셔너리
    'best_k':       best_k,
    'cluster_means': cluster_means,  # 클러스터별 평균 (분석용)
    'cluster_sizes': cluster_sizes,
    'overall_feature_means': overall_feature_means,
    'cluster_summary': cluster_summary,
    'balance_strata_cols': BALANCE_STRATA_COLS,
    'balance_target_per_stratum': balance_target_per_stratum,
    'balance_profile_count_original': int(len(sub_ratio)),
    'balance_profile_count_train': int(len(balanced_sub_ratio)),
    'balance_report': balance_report,
    'silhouette_score': float(silhouette),
    'top2_cluster_share_pct': float(top2_cluster_share_pct),
}

joblib.dump(model_bundle, 'model/gmm_model.pkl')
print("\n✅ 모델 저장 완료 → model/gmm_model.pkl")
print(f"   - 클러스터 수  : {best_k}개")
print(f"   - 피처 수      : {len(feature_cols)}개")
print(f"   - 원본 프로파일: {len(sub_ratio)}개")
print(f"   - 학습 프로파일: {len(balanced_sub_ratio)}개")
