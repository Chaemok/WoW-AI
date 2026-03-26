"""
로컬 노트북이나 GPU 서버에서 내부 AI API를 호출하는 예제입니다.
사용 전: pip install requests
"""

from __future__ import annotations

import json
from pathlib import Path

import requests


SERVER_URL = "http://<서버IP>:8000"  # 실제 서버 주소로 교체
SAMPLE_ITEMS_PATH = Path(__file__).resolve().parent / "analysis" / "sample_monthly_items.json"


def _print_analysis(result: dict) -> None:
    """백엔드 계약에 맞춘 핵심 분석 결과를 출력합니다."""
    cluster = result.get("cluster") or {}
    summary = result.get("summary") or {}
    overspending = result.get("overspending") or []

    print(f"[클러스터] {cluster.get('id')} - {cluster.get('name')}")
    if cluster.get("description"):
        print(f"[설명] {cluster['description']}")

    print(f"[피드백]\n{result.get('feedback', '')}")
    print("\n[절감 권장]")
    for item in overspending:
        print(f"  {item['name']:18s}  {item['savableAmount']:>8,}원")

    if summary:
        print(f"  {'예상 총 절감':18s}  {summary.get('totalSavable', 0):>8,}원")
        print(f"  {'절감 후 예상 지출':18s}  {summary.get('expectedSpending', 0):>8,}원")


def analyze(csv_path: str | None = None, demo: bool = False, items: list[dict] | None = None) -> dict:
    """CSV 또는 카테고리 집계 배열로 소비 분석을 요청합니다."""
    if demo:
        payload = {"demo": True}
    elif items is not None:
        payload = {"items": items, "keepSession": False}
    else:
        if not csv_path:
            raise ValueError("csv_path 또는 items 중 하나를 제공해 주세요.")
        with open(csv_path, encoding="utf-8") as f:
            csv_text = f.read()
        payload = {"csvText": csv_text, "keepSession": False}

    response = requests.post(f"{SERVER_URL}/api/analyze", json=payload, timeout=180)
    response.raise_for_status()
    result = response.json()
    _print_analysis(result)
    return result


def analyze_sample_items() -> dict:
    """샘플 집계 데이터를 이용해 월간 리포트 호출을 테스트합니다."""
    with open(SAMPLE_ITEMS_PATH, encoding="utf-8") as f:
        items = json.load(f)
    return analyze(items=items)


def chat(message: str, session_id: str | None = None) -> dict:
    """세션 기반 대화 메시지를 전송합니다."""
    payload = {"message": message}
    if session_id:
        payload["sessionId"] = session_id

    response = requests.post(f"{SERVER_URL}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()

    print(f"[AI] {result['reply']}")
    return result


def health() -> None:
    """서버 상태를 확인합니다."""
    response = requests.get(f"{SERVER_URL}/health", timeout=5)
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    # 1) 서버 상태 확인
    health()

    # 2) 샘플 집계 데이터로 월간 분석 호출
    result = analyze_sample_items()

    # 3) 세션 기반 대화를 확인하고 싶다면 demo 분석을 따로 호출
    demo_result = analyze(demo=True)
    session_id = demo_result.get("sessionId")

    if session_id:
        chat("커피 외에 다른 절감 방법 알려줘", session_id=session_id)
