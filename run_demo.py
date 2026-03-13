"""
run_demo.py
────────────────────────────────────────────────────────────
[역할] 한 번의 실행으로 HTML 데모 서버를 띄우고 브라우저를 연다
[실행] python run_demo.py
────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import threading
import time
import webbrowser

from analysis.demo_app import run_demo_server

HOST = '127.0.0.1'
PORT = 5000
URL = f'http://{HOST}:{PORT}'


def open_browser_after_delay(delay: float = 1.0):
    """서버가 뜬 뒤 브라우저 열기"""
    time.sleep(delay)
    webbrowser.open(URL)


if __name__ == '__main__':
    print('소비 유형 HTML 데모를 실행합니다.')
    print(f'브라우저 주소: {URL}')
    threading.Thread(target=open_browser_after_delay, daemon=True).start()
    run_demo_server(host=HOST, port=PORT, debug=False)
