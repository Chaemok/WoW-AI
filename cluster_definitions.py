"""고정 클러스터 이름/설명 정의"""

FINAL_CLUSTER_DEFINITIONS = {
    0: {
        'name': '장보기·의류형',
        'description': '음/식료품소매를 중심으로 의복/의류와 의약/의료품 지출이 함께 나타나는 생활구매형 유형',
    },
    1: {
        'name': '사무·서적형',
        'description': '사무/교육용품과 서적/도서 비중이 매우 높고 분식 소비가 함께 나타나는 학습 집중형 유형',
    },
    2: {
        'name': '화장품·커피형',
        'description': '화장품소매와 커피/음료 비중이 두드러지고 외식이 함께 나타나는 취향 소비형 유형',
    },
    3: {
        'name': '차량·생활이동형',
        'description': '자동차/유지비를 중심으로 외식, 병원/의료, 교육/학원 지출이 함께 붙는 이동 생활형 유형',
    },
    4: {
        'name': '외식·생활관리형',
        'description': '외식 비중이 높고 자동차/유지비, 음/식료품소매, 의약/의료품 지출이 함께 나타나는 생활관리형 유형',
    },
    5: {
        'name': '의료·의약품 중심형',
        'description': '병원/의료와 의약/의료품 비중이 높고 음/식료품소매가 함께 나타나는 건강관리형 유형',
    },
    6: {
        'name': '교육·선물 특화형',
        'description': '교육/학원, 선물/완구, 가전제품 소비가 매우 집중된 특화형 유형',
    },
    7: {
        'name': '커피·디저트형',
        'description': '커피/음료 비중이 매우 높고 화장품소매와 제과/제빵/떡/케익이 함께 나타나는 간식 중심형 유형',
    },
}

FINAL_CLUSTER_LABELS = {
    cluster_id: definition['name']
    for cluster_id, definition in FINAL_CLUSTER_DEFINITIONS.items()
}


def get_cluster_name(cluster_id: int, fallback: str | None = None) -> str:
    return FINAL_CLUSTER_LABELS.get(cluster_id, fallback or f'Cluster {cluster_id}')


def get_cluster_description(cluster_id: int, fallback: str | None = None) -> str:
    definition = FINAL_CLUSTER_DEFINITIONS.get(cluster_id)
    if definition:
        return definition['description']
    return fallback or ''
