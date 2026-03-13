"""Compare alternative unsupervised clustering methods against the current GMM pipeline."""

from __future__ import annotations

import json
import os
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FIXED_K = 8
BALANCE_STRATA_COLS = ["age", "sex", "top2_category_pattern"]
MIN_BALANCED_PER_STRATUM = 2
MAX_BALANCED_PER_STRATUM = 3
BALANCE_RANDOM_STATE = 42

DATA_DIR = "data"
ZIP_FILES = {
    "2507": "카드소비 데이터_202507.zip",
    "2508": "카드소비 데이터_202508.zip",
    "2509": "카드소비 데이터_202509.zip",
    "2510": "카드소비 데이터_202510.zip",
    "2511": "카드소비 데이터_202511.zip",
    "2512": "카드소비 데이터_202512.zip",
}
LOAD_COLS = ["age", "sex", "card_tpbuz_nm_2", "amt", "cnt"]

MODEL_DIR = "model"
RESULT_JSON_PATH = os.path.join(MODEL_DIR, "alternative_cluster_comparison.json")
RESULT_MD_PATH = os.path.join(MODEL_DIR, "alternative_cluster_comparison.md")
RESULT_CSV_PATH = os.path.join(MODEL_DIR, "alternative_cluster_comparison.csv")

PERSONAL_INPUT_PATH = os.path.join("analysis", "personal_training_input.csv")
GMM_MODEL_PATH = os.path.join(MODEL_DIR, "gmm_model.pkl")


CATEGORY_MAP = {
    "종합소매점": "제외",
    "기타결제": "제외", "기타용품": "제외", "기타의료": "제외", "기타교육": "제외",
    "공공기관": "제외", "기업": "제외", "단체": "제외", "종교": "제외",
    "회비/공과금": "제외", "휴게소/대형업체": "제외", "가례서비스": "제외",
    "제조/도매": "제외", "전문서비스": "제외", "광고/인쇄/인화": "제외",
    "금융상품/서비스": "제외", "무점포서비스": "제외", "보안/운송": "제외", "부동산": "제외",
    "일반병원": "병원/의료", "종합병원": "병원/의료", "특화병원": "병원/의료",
    "차량관리/부품": "자동차/유지비", "차량관리/서비스": "자동차/유지비",
    "차량판매": "자동차/유지비", "연료판매": "자동차/유지비",
    "유아교육": "교육/학원", "입시학원": "교육/학원", "외국어학원": "교육/학원",
    "기술/직업교육학원": "교육/학원", "예체능계학원": "교육/학원", "독서실/고시원": "교육/학원",
    "유흥주점": "주점", "간이주점": "주점",
    "고기요리": "육류/회식", "닭/오리요리": "육류/회식",
    "한식": "외식",
    "일식/수산물": "외식", "별식/퓨전요리": "외식",
    "양식": "외식", "중식": "외식", "부페": "외식",
    "사우나/휴게시설": "건강/뷰티/마사지", "미용서비스": "건강/뷰티/마사지",
    "요가/단전/마사지": "건강/뷰티/마사지",
}


def remove_outliers_iqr(group: pd.DataFrame) -> pd.DataFrame:
    q1 = group["건당가격"].quantile(0.25)
    q3 = group["건당가격"].quantile(0.75)
    iqr = q3 - q1
    return group[
        (group["건당가격"] >= q1 - 1.5 * iqr)
        & (group["건당가격"] <= q3 + 1.5 * iqr)
    ]


def load_training_profiles() -> tuple[pd.DataFrame, list[str]]:
    all_dfs = []
    for month, zipname in ZIP_FILES.items():
        zip_path = os.path.join(DATA_DIR, zipname)
        monthly_dfs = []
        with zipfile.ZipFile(zip_path) as zf:
            for filename in sorted(zf.namelist()):
                with zf.open(filename) as f:
                    df_city = pd.read_csv(f, usecols=LOAD_COLS)
                    df_city["month"] = month
                    monthly_dfs.append(df_city)
        all_dfs.append(pd.concat(monthly_dfs, ignore_index=True))

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all[df_all["cnt"] > 0].copy()
    df_all["건당가격"] = df_all["amt"] / df_all["cnt"]
    df_all = pd.concat(
        [remove_outliers_iqr(group) for _, group in df_all.groupby("card_tpbuz_nm_2")],
        ignore_index=True,
    )
    df_all["refined_category"] = df_all["card_tpbuz_nm_2"].map(lambda x: CATEGORY_MAP.get(x, x))
    df_all = df_all[df_all["refined_category"] != "제외"].copy()

    sub_pivot = (
        df_all.groupby(["month", "age", "sex", "refined_category"])["amt"]
        .sum()
        .reset_index()
        .pivot_table(index=["month", "age", "sex"], columns="refined_category", values="amt", fill_value=0)
    )
    sub_ratio = sub_pivot.div(sub_pivot.sum(axis=1), axis=0).round(4)
    sub_ratio.columns = [f"비율_{col}" for col in sub_ratio.columns]
    sub_ratio = sub_ratio.reset_index()
    feature_cols = [col for col in sub_ratio.columns if col.startswith("비율_")]
    return sub_ratio, feature_cols


def add_profile_balance_keys(profile_df: pd.DataFrame, ratio_cols: list[str]) -> pd.DataFrame:
    profile_df = profile_df.copy()
    top2_features = profile_df[ratio_cols].apply(
        lambda row: row.sort_values(ascending=False).head(2).index.tolist(),
        axis=1,
    )
    profile_df["dominant_category"] = top2_features.str[0].str.replace("비율_", "", regex=False)
    profile_df["secondary_category"] = top2_features.str[1].str.replace("비율_", "", regex=False)
    profile_df["top2_category_pattern"] = (
        profile_df["dominant_category"] + " + " + profile_df["secondary_category"]
    )
    return profile_df


def build_balanced_training_profiles(profile_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    stratum_counts = (
        profile_df.groupby(BALANCE_STRATA_COLS, dropna=False)
        .size()
        .rename("original_count")
        .reset_index()
        .sort_values("original_count", ascending=False)
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
        .rename("balanced_count")
        .reset_index()
    )
    balance_report = stratum_counts.merge(balanced_counts, on=BALANCE_STRATA_COLS, how="left")
    balance_report["balanced_count"] = balance_report["balanced_count"].fillna(0).astype(int)
    return balanced_df, balance_report, target_per_stratum


def build_personal_feature_vector(feature_cols: list[str]) -> np.ndarray:
    personal_df = pd.read_csv(PERSONAL_INPUT_PATH)
    personal_df = personal_df[personal_df["cnt"] > 0].copy()
    personal_df["refined_category"] = personal_df["card_tpbuz_nm_2"].map(lambda x: CATEGORY_MAP.get(x, x))
    personal_df = personal_df[personal_df["refined_category"] != "제외"].copy()
    cat_amt = personal_df.groupby("refined_category")["amt"].sum()
    cat_ratio = (cat_amt / cat_amt.sum()).to_dict()
    return np.array([[cat_ratio.get(col.replace("비율_", ""), 0.0) for col in feature_cols]])


def top_signature_from_centroid(centroid: np.ndarray, feature_cols: list[str], top_n: int = 3) -> str:
    pairs = sorted(zip(feature_cols, centroid), key=lambda item: item[1], reverse=True)[:top_n]
    return " · ".join(
        f"{feature.replace('비율_', '')}({value * 100:.0f}%)" for feature, value in pairs
    )


def nearest_centroid_assignment(x: np.ndarray, centroids: np.ndarray) -> tuple[int, np.ndarray]:
    distances = np.linalg.norm(centroids - x, axis=1)
    order = np.argsort(distances)
    return int(order[0]), distances


def cluster_balance_stats(labels: np.ndarray) -> dict:
    counts = pd.Series(labels).value_counts().sort_index()
    shares = (counts / len(labels) * 100).round(2)
    return {
        "counts": {int(key): int(value) for key, value in counts.items()},
        "shares_pct": {int(key): float(value) for key, value in shares.items()},
        "top2_share_pct": float(shares.sort_values(ascending=False).head(2).sum()),
        "singleton_clusters": int((counts == 1).sum()),
    }


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    sub_ratio, feature_cols = load_training_profiles()
    sub_ratio = add_profile_balance_keys(sub_ratio, feature_cols)
    balanced_sub_ratio, balance_report, balance_target_per_stratum = build_balanced_training_profiles(sub_ratio)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(balanced_sub_ratio[feature_cols].values)
    x_full = scaler.transform(sub_ratio[feature_cols].values)
    x_personal = scaler.transform(build_personal_feature_vector(feature_cols))

    results = []

    models = {
        "kmeans": KMeans(n_clusters=FIXED_K, random_state=BALANCE_RANDOM_STATE, n_init=20),
        "agglomerative": AgglomerativeClustering(n_clusters=FIXED_K, linkage="ward"),
    }

    for model_name, model in models.items():
        if model_name == "kmeans":
            model.fit(x_train)
            centroids = model.cluster_centers_
            full_labels = model.predict(x_full)
            personal_cluster, personal_distances = nearest_centroid_assignment(x_personal[0], centroids)
        else:
            train_labels = model.fit_predict(x_train)
            centroids = np.vstack([x_train[train_labels == cid].mean(axis=0) for cid in sorted(np.unique(train_labels))])
            full_labels = np.array([nearest_centroid_assignment(row, centroids)[0] for row in x_full])
            personal_cluster, personal_distances = nearest_centroid_assignment(x_personal[0], centroids)

        balance_stats = cluster_balance_stats(full_labels)
        silhouette = float(silhouette_score(x_full, full_labels))
        centroid_signatures = {
            int(cid): top_signature_from_centroid(
                scaler.inverse_transform(centroids[cid].reshape(1, -1))[0],
                feature_cols,
            )
            for cid in range(len(centroids))
        }
        personal_distance_ranks = np.argsort(personal_distances)[:3]
        results.append({
            "algorithm": model_name,
            "silhouette_score": round(silhouette, 4),
            "top2_share_pct": round(balance_stats["top2_share_pct"], 2),
            "singleton_clusters": balance_stats["singleton_clusters"],
            "cluster_counts": balance_stats["counts"],
            "cluster_shares_pct": balance_stats["shares_pct"],
            "personal_cluster": int(personal_cluster),
            "personal_signature": centroid_signatures[int(personal_cluster)],
            "personal_top3_candidates": [
                {
                    "cluster_id": int(cid),
                    "signature": centroid_signatures[int(cid)],
                    "distance": round(float(personal_distances[cid]), 4),
                }
                for cid in personal_distance_ranks
            ],
        })

    gmm_bundle = joblib.load(GMM_MODEL_PATH)
    gmm_assignments = pd.read_csv(os.path.join(MODEL_DIR, "cluster_assignments.csv"))
    gmm_stats = cluster_balance_stats(gmm_assignments["cluster"].values)
    results.insert(0, {
        "algorithm": "gmm_current",
        "silhouette_score": None,
        "top2_share_pct": round(gmm_stats["top2_share_pct"], 2),
        "singleton_clusters": gmm_stats["singleton_clusters"],
        "cluster_counts": gmm_stats["counts"],
        "cluster_shares_pct": gmm_stats["shares_pct"],
        "personal_cluster": None,
        "personal_signature": "See note.ipynb / gmm_predict.py for current dynamic interpretation",
        "personal_top3_candidates": [],
        "pca_n_components": gmm_bundle.get("pca_n_components"),
    })

    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    pd.DataFrame([
        {
            "algorithm": item["algorithm"],
            "silhouette_score": item.get("silhouette_score"),
            "top2_share_pct": item["top2_share_pct"],
            "singleton_clusters": item["singleton_clusters"],
            "personal_cluster": item.get("personal_cluster"),
            "personal_signature": item["personal_signature"],
        }
        for item in results
    ]).to_csv(RESULT_CSV_PATH, index=False, encoding="utf-8-sig")

    markdown_lines = [
        "# Alternative Cluster Comparison",
        "",
        f"- profile count: {len(sub_ratio)}",
        f"- balanced train profile count: {len(balanced_sub_ratio)}",
        f"- balance target per stratum: {balance_target_per_stratum}",
        f"- compared methods: {', '.join(item['algorithm'] for item in results)}",
        "",
    ]
    for item in results:
        markdown_lines.extend([
            f"## {item['algorithm']}",
            "",
            f"- top2 cluster share: {item['top2_share_pct']:.2f}%",
            f"- singleton clusters: {item['singleton_clusters']}",
            f"- silhouette score: {item['silhouette_score'] if item['silhouette_score'] is not None else 'n/a'}",
            f"- personal cluster: {item.get('personal_cluster', 'n/a')}",
            f"- personal signature: {item['personal_signature']}",
            "",
        ])
        for candidate in item.get("personal_top3_candidates", []):
            markdown_lines.append(
                f"  - cluster {candidate['cluster_id']}: {candidate['signature']} | distance {candidate['distance']:.4f}"
            )
        markdown_lines.append("")

    with open(RESULT_MD_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(markdown_lines))

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()