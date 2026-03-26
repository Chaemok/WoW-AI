"""
배포 직후 내부 AI API가 정상 동작하는지 빠르게 확인하는 스크립트입니다.

확인 항목:
- GET /health
- POST /api/analyze
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_ITEMS_PATH = BASE_DIR / "analysis" / "sample_monthly_items.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-url",
        default=os.environ.get("SERVER_URL", "http://127.0.0.1:8000"),
        help="내부 AI 서버 주소",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=240,
        help="analyze 요청 타임아웃(초)",
    )
    return parser.parse_args()


def load_sample_items() -> list[dict]:
    with open(SAMPLE_ITEMS_PATH, encoding="utf-8") as f:
        return json.load(f)


def check_health(server_url: str) -> dict:
    response = requests.get(f"{server_url}/health", timeout=10)
    response.raise_for_status()
    return response.json()


def check_analyze(server_url: str, timeout: int) -> dict:
    payload = {
        "items": load_sample_items(),
        "keepSession": False,
    }
    response = requests.post(
        f"{server_url}/api/analyze",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def print_health(data: dict) -> None:
    print("[health]")
    print(f"  status : {data.get('status')}")
    print(f"  model  : {data.get('model')}")
    print(f"  device : {data.get('device')}")
    print(f"  gpu    : {data.get('gpu')}")


def print_analysis(data: dict) -> None:
    cluster = data.get("cluster") or {}
    summary = data.get("summary") or {}
    print("[analyze]")
    print(f"  cluster : {cluster.get('id')} - {cluster.get('name')}")
    print(f"  savable : {summary.get('totalSavable', 0):,}원")
    print(f"  expected: {summary.get('expectedSpending', 0):,}원")
    print(f"  count   : {data.get('sourceTransactionCount', 0)}")

    overspending = data.get("overspending") or []
    if overspending:
        top = overspending[0]
        print(f"  top     : {top.get('name')} / {top.get('savableAmount', 0):,}원")


def main() -> int:
    args = parse_args()

    try:
        health_data = check_health(args.server_url)
        print_health(health_data)

        analyze_data = check_analyze(args.server_url, args.timeout)
        print_analysis(analyze_data)
        print("\n[OK] 내부 AI API 스모크 테스트가 완료되었습니다.")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] 스모크 테스트 실패: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
