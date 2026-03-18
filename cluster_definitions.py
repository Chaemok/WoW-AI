"""고정 클러스터 이름/설명 정의"""

FINAL_CLUSTER_DEFINITIONS = {
    0: {
        'name': '외식·생활잡화형',
        'description': '외식(19%)과 음/식료품소매(17%)가 주축이며, 의복/의류(13%)·의약/의료품(12%) 등 생활 전반에 걸쳐 고르게 지출하는 유형',
    },
    1: {
        'name': '사무·서적형',
        'description': '사무/교육용품(22%)과 서적/도서(11%) 비중이 압도적으로 높고 분식(18%) 소비가 함께 나타나는 학습 집중형 유형',
    },
    2: {
        'name': '외식·드라이브형',
        'description': '외식(20%)이 가장 많고 자동차/유지비(8.5%)·커피/음료(8.4%)가 함께 높은 이동 외향형 소비 유형',
    },
    3: {
        'name': '차량·의료관리형',
        'description': '자동차/유지비(23%)가 가장 높고 외식(17%)·병원/의료(11%)·교육/학원(7%)이 함께 나타나는 이동 중심 생활관리형 유형',
    },
    4: {
        'name': '외식·의료집중형',
        'description': '외식(26%)이 최다 지출이며, 자동차/유지비(17%)·음/식료품소매(16%)에 더해 의약/의료품(11%)·병원/의료(11%)까지 높은 생활밀착형 유형',
    },
    5: {
        'name': '의료·자동차형',
        'description': '병원/의료(17%)와 자동차/유지비(16%)가 핵심 지출이고, 외식(19%)·음/식료품소매(13%)가 함께 나타나는 건강·이동 관리형 유형',
    },
    6: {
        'name': '교육·선물 특화형',
        'description': '교육/학원(22%)·선물/완구(22%)·가전제품(20%)이 매우 집중된 특수 목적형 지출 유형',
    },
    7: {
        'name': '커피·디저트형',
        'description': '커피/음료(46%) 비중이 압도적으로 높고 화장품소매(10%)·분식(9%)·제과/제빵(9%)이 함께 나타나는 카페 중심형 유형',
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
