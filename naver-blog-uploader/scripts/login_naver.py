"""
네이버 로그인 후 쿠키를 저장하는 스크립트.
사용자가 직접 로그인하면 쿠키를 config/naver_cookies.json에 저장한다.
"""

import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

_SKILL_DIR = Path(__file__).parent.parent
COOKIE_PATH = _SKILL_DIR / "config" / "naver_cookies.json"
NAVER_HOME = "https://www.naver.com/"


def main():
    COOKIE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        print("브라우저를 열고 네이버 로그인 페이지로 이동합니다.")
        print("로그인을 완료하면 쿠키가 자동으로 저장됩니다. (최대 5분 대기)")
        page.goto("https://nid.naver.com/nidlogin.login")

        # NID_AUT 쿠키가 생길 때까지 폴링 (로그인 완료 감지)
        import time
        print("로그인 완료를 기다리는 중...")
        for _ in range(300):  # 최대 5분
            cookies = context.cookies()
            auth_cookies = [c for c in cookies if c["name"] in ("NID_AUT", "NID_SES")]
            if len(auth_cookies) >= 2:
                break
            time.sleep(1)
        else:
            print("오류: 5분 내에 로그인이 완료되지 않았습니다.")
            browser.close()
            return

        with open(COOKIE_PATH, "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)

        print("쿠키 저장 완료!")
        print(f"저장 경로: {COOKIE_PATH}")
        print(f"저장된 쿠키 수: {len(cookies)}개")

        browser.close()


if __name__ == "__main__":
    main()
