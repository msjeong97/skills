#!/usr/bin/env python3
"""
네이버 블로그 테스트 글 삭제 스크립트.

안전 규칙:
- 반드시 --log-no 명시 필요
- 삭제 대상 글 제목이 "[TEST]"로 시작해야 함
- 조건 불충족 시 삭제 거부

사용법:
    python delete_post.py --log-no 123456789
    python delete_post.py --log-no 123456789 --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

_SKILL_DIR = Path(__file__).parent.parent
COOKIE_PATH = _SKILL_DIR / "config" / "naver_cookies.json"
CONFIG_PATH = _SKILL_DIR / "config" / "settings.json"

TEST_TITLE_PREFIX = "[TEST]"


def load_settings() -> dict:
    if not CONFIG_PATH.exists():
        print("오류: config/settings.json 없음. blog_id를 포함한 파일을 생성해 주세요.")
        sys.exit(1)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        settings = json.load(f)
    if "blog_id" not in settings:
        print("오류: config/settings.json에 blog_id가 없습니다.")
        sys.exit(1)
    return settings


def load_cookies() -> list[dict]:
    if not COOKIE_PATH.exists():
        print("오류: 쿠키 파일이 없습니다. 먼저 login_naver.py를 실행해 주세요.")
        sys.exit(1)
    with open(COOKIE_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_post_title(page, blog_id: str, log_no: str) -> str | None:
    """글 보기 페이지에서 제목 추출."""
    url = f"https://blog.naver.com/{blog_id}/{log_no}"
    page.goto(url, wait_until="networkidle")
    time.sleep(2)
    title = page.evaluate("""() => {
        return document.querySelector('.se-title-text')?.innerText?.trim()
            || document.querySelector('h3.se_textarea')?.innerText?.trim()
            || null;
    }""")
    return title


def delete_post(page, blog_id: str, log_no: str) -> bool:
    """블로그 관리 페이지에서 특정 logNo 글 삭제."""
    manage_url = f"https://blog.naver.com/PostDelete.naver?blogId={blog_id}&logNo={log_no}"
    page.goto(manage_url, wait_until="networkidle")
    time.sleep(2)

    # 삭제 확인 팝업 처리
    try:
        confirm_btn = page.locator("button:has-text('삭제'), input[value='확인']")
        if confirm_btn.count() > 0:
            confirm_btn.first.click(force=True, timeout=3000)
            time.sleep(2)
    except Exception as e:
        print(f"  ! 팝업 처리 중 오류 (무시하고 계속): {e}")

    # 삭제 후 리다이렉트 확인
    return "PostDelete" not in page.url or "complete" in page.url.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="네이버 블로그 테스트 글 삭제")
    parser.add_argument("--log-no", required=True, help="삭제할 글 logNo (숫자)")
    parser.add_argument("--dry-run", action="store_true", help="실제 삭제 없이 검증만")
    args = parser.parse_args()

    # logNo 숫자 검증
    if not re.fullmatch(r"\d+", args.log_no):
        print(f"오류: --log-no는 숫자여야 합니다. 입력값: {args.log_no!r}")
        sys.exit(1)

    settings = load_settings()
    blog_id = settings["blog_id"]
    cookies = load_cookies()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        ctx.add_cookies(cookies)
        page = ctx.new_page()

        print(f"글 제목 확인 중... (logNo: {args.log_no})")
        title = get_post_title(page, blog_id, args.log_no)

        if title is None:
            print(f"오류: 글 제목을 읽을 수 없습니다. logNo={args.log_no}")
            browser.close()
            sys.exit(1)

        print(f"  글 제목: {title!r}")

        # 안전 조건: [TEST]로 시작해야만 삭제 허용
        if not title.strip().startswith(TEST_TITLE_PREFIX):
            print(f"거부: 제목이 {TEST_TITLE_PREFIX!r}로 시작하지 않습니다.")
            print("  → 테스트 글만 자동삭제 가능합니다.")
            browser.close()
            sys.exit(1)

        if args.dry_run:
            print(f"[dry-run] 삭제 대상 확인됨: {title!r} (logNo={args.log_no})")
            print("  → --dry-run이므로 실제 삭제하지 않음.")
            browser.close()
            return

        print(f"삭제 진행: {title!r} (logNo={args.log_no})")
        success = delete_post(page, blog_id, args.log_no)

        if success:
            print(f"✅ 삭제 완료: logNo={args.log_no}")
        else:
            print(f"⚠️  삭제 결과 불명확. 블로그에서 직접 확인 필요.")

        browser.close()


if __name__ == "__main__":
    main()
