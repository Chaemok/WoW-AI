# Alternative Cluster Comparison

- profile count: 132
- balanced train profile count: 147
- balance target per stratum: 3
- compared methods: gmm_current, kmeans, agglomerative

## gmm_current

- top2 cluster share: 67.42%
- singleton clusters: 2
- silhouette score: n/a
- personal cluster: None
- personal signature: See note.ipynb / gmm_predict.py for current dynamic interpretation


## kmeans

- top2 cluster share: 68.18%
- singleton clusters: 3
- silhouette score: 0.1248
- personal cluster: 0
- personal signature: 외식(20%) · 커피/음료(9%) · 화장품소매(6%)

  - cluster 0: 외식(20%) · 커피/음료(9%) · 화장품소매(6%) | distance 5306.6791
  - cluster 1: 외식(19%) · 자동차/유지비(17%) · 병원/의료(17%) | distance 5306.9341
  - cluster 7: 자동차/유지비(18%) · 외식(15%) · 병원/의료(12%) | distance 5307.1605

## agglomerative

- top2 cluster share: 66.66%
- singleton clusters: 1
- silhouette score: 0.1391
- personal cluster: 2
- personal signature: 외식(20%) · 커피/음료(9%) · 자동차/유지비(7%)

  - cluster 2: 외식(20%) · 커피/음료(9%) · 자동차/유지비(7%) | distance 5306.7566
  - cluster 5: 외식(19%) · 병원/의료(17%) · 자동차/유지비(15%) | distance 5306.9053
  - cluster 3: 자동차/유지비(21%) · 외식(15%) · 병원/의료(12%) | distance 5307.1766
