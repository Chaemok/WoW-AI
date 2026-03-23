"""
로컬 노트북에서 챗봇 서버 사용하는 클라이언트 예제
사용 전: pip install requests
"""
import requests

SERVER_URL = "http://<서버IP>:8000"  # ← 실제 서버 IP로 교체


def analyze(csv_path: str = None, demo: bool = False) -> dict:
    """CSV 파일 소비 분석 요청"""
    if demo:
        payload = {"demo": True}
    else:
        with open(csv_path, encoding="utf-8") as f:
            csv_text = f.read()
        payload = {"csv_text": csv_text}

    r = requests.post(f"{SERVER_URL}/api/analyze", json=payload, timeout=120)
    r.raise_for_status()
    result = r.json()

    print(f"[클러스터] {result['cluster_id']} - {result['cluster_name']}")
    print(f"[피드백]\n{result['feedback']}")
    print(f"\n[절감 권장]")
    for cat, amt in result["reduction_summary"].items():
        print(f"  {cat:18s}  {amt:>8,}원")
    print(f"  {'예상 총 절감':18s}  {result['total_reduction']:>8,}원")

    return result


def chat(message: str, session_id: str = None) -> dict:
    """대화 메시지 전송"""
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id

    r = requests.post(f"{SERVER_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    result = r.json()

    print(f"[AI] {result['reply']}")
    return result


def health():
    """서버 상태 확인"""
    r = requests.get(f"{SERVER_URL}/health", timeout=5)
    r.raise_for_status()
    print(r.json())


# ─── 사용 예시 ───────────────────────────────────────────────
if __name__ == "__main__":
    # 1) 서버 상태 확인
    health()

    # 2) 더미 데이터로 분석
    result = analyze(demo=True)
    session_id = result["session_id"]

    # 3) 분석 결과 기반으로 이어서 대화
    chat("커피 외에 다른 절감 방법 알려줘", session_id=session_id)
    chat("외식비 줄이는 실천 방법 3가지만", session_id=session_id)

    # 4) 내 CSV로 분석하려면:
    # result = analyze(csv_path="my_transactions.csv")
