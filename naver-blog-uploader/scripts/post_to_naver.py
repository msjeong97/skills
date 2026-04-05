"""
네이버 블로그 자동 포스팅 스크립트 (SmartEditor 3 기준)

사용법:
    python post_to_naver.py \
        --blog-id "blog10_04" \
        --title "제목" \
        --content "본문 [그림1] 계속 [장소] [그림2]" \
        --images "img1.jpg,img2.jpg" \
        --tags "태그1,태그2" \
        --map-url "https://naver.me/xxxx" \
        --caller-chat-id "telegram:8279476113"
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


def check_authorization(caller_chat_id: str | None) -> None:
    """발신자 chat ID가 화이트리스트에 있는지 확인. 없으면 종료."""
    if not CONFIG_PATH.exists():
        print("오류: config.json 없음. 권한 확인 불가.")
        sys.exit(1)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)

    authorized = config.get("authorized_chat_ids", [])

    if not caller_chat_id:
        print("오류: --caller-chat-id 미전달. 권한 없음.")
        sys.exit(1)

    if caller_chat_id not in authorized:
        print(f"오류: 미인가 채널 ({caller_chat_id}). 발행 거부.")
        sys.exit(1)

    print(f"  ✓ 인증 완료: {caller_chat_id}")

import pyperclip
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

COOKIE_PATH = Path.home() / ".openclaw" / "naver_cookies.json"


def load_cookies() -> list[dict]:
    if not COOKIE_PATH.exists():
        print("오류: 쿠키 파일이 없습니다. 먼저 login_naver.py를 실행해 주세요.")
        sys.exit(1)
    with open(COOKIE_PATH, encoding="utf-8") as f:
        return json.load(f)


def paste_text(page, text: str):
    """클립보드를 통해 텍스트 붙여넣기 (IME 깨짐 방지)."""
    pyperclip.copy(text)
    page.keyboard.press("Meta+v")
    time.sleep(0.5)


def get_place_name_from_map_url(map_url: str) -> str | None:
    """네이버 지도 URL(단축 포함)에서 장소명 추출."""
    try:
        # 단축 URL → 실제 URL 리다이렉트 따라가기
        result = subprocess.run(
            ["curl", "-sI", map_url],
            capture_output=True, text=True, timeout=10
        )
        location = ""
        for line in result.stdout.splitlines():
            if line.lower().startswith("location:"):
                location = line.split(":", 1)[1].strip()
                break

        target_url = location if location else map_url

        # Playwright로 og:title 추출
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(target_url, wait_until="domcontentloaded")
            time.sleep(3)
            name = page.evaluate("""() => {
                const og = document.querySelector('meta[property="og:title"]');
                if (og && og.content) return og.content.replace(' - 네이버지도', '').trim();
                return document.title.replace(' - 네이버지도', '').trim();
            }""")
            browser.close()
            print(f"  장소명 추출: {name}")
            return name if name else None
    except Exception as e:
        print(f"  ! 장소명 추출 실패: {e}")
        return None


def split_content(content: str, images: list[str]) -> list[tuple[str, str | None, bool]]:
    """
    본문을 [그림N] / [장소] 마커 기준으로 분리.
    반환값: [(텍스트, 이미지경로_또는_None, is_place_marker), ...]
    """
    # [그림N] 과 [장소] 를 모두 분리
    parts = re.split(r"(\[그림\d+\]|\[장소\])", content)
    result = []
    img_counter = [0]

    i = 0
    while i < len(parts):
        text = parts[i].strip()
        i += 1
        if i < len(parts):
            marker = parts[i]
            i += 1
            if marker == "[장소]":
                result.append((text, None, True))
            else:
                img_idx = int(re.search(r"\d+", marker).group()) - 1
                img_path = images[img_idx] if img_idx < len(images) else None
                result.append((text, img_path, False))
        else:
            result.append((text, None, False))
    return result


def upload_image(page, image_path: str) -> bool:
    """사진 추가 버튼 클릭 → file input → 이미지 업로드."""
    abs_path = str(Path(image_path).resolve())
    try:
        photo_btn = page.locator("button:has-text('사진')").first
        photo_btn.click(timeout=5000)
        time.sleep(1)
        file_input = page.locator("input[type='file'][accept*='jpg']").first
        file_input.set_input_files(abs_path, timeout=5000)
        time.sleep(3)
        print(f"  ✓ 이미지 업로드 완료: {Path(image_path).name}")
        return True
    except Exception as e:
        print(f"  ✗ 이미지 업로드 실패: {e}")
        return False


def add_place(page, place_name: str) -> bool:
    """
    장소 탭에서 장소명 검색 → 추가 → 지도에 직접 표시 체크 → 확인.

    흐름:
    1. "장소" 툴바 버튼 클릭
    2. 장소명 입력 (자동완성 선택 or 검색)
    3. "추가" 버튼 — dispatchEvent (숨겨진 버튼이라 force 불가)
    4. "지도에 직접 표시" 체크 — label 클릭
    5. "확인" 버튼 — dispatchEvent
    6. 본문에 지도 카드 삽입 확인
    """
    try:
        print(f"  장소 추가 중: {place_name}")

        # ── 1. "장소" 툴바 버튼 클릭 ────────────────────────
        page.evaluate("""() => {
            // se-toolbar-item 중 장소 아이템의 버튼 클릭
            const mapItem = document.querySelector('.se-toolbar-item-map button');
            if (mapItem) { mapItem.click(); return; }
            // fallback: 텍스트로 찾기
            Array.from(document.querySelectorAll('button'))
                .find(b => b.innerText.includes('장소'))?.click();
        }""")
        time.sleep(2)

        # ── 2. 장소명 입력 ────────────────────────────────────
        place_input = page.locator(".react-autosuggest__input").first
        place_input.wait_for(timeout=5000)
        place_input.click()
        place_input.fill(place_name)
        time.sleep(2)

        # ── 3. 자동완성 결과 선택 ────────────────────────────
        clicked = page.evaluate("""() => {
            const items = document.querySelectorAll('.react-autosuggest__suggestions-list li');
            for (const item of items) {
                const text = item.innerText.trim();
                if (text !== '국내' && text !== '해외' && text.length > 0) {
                    item.click();
                    return text;
                }
            }
            return null;
        }""")

        if not clicked:
            # fallback: 검색 버튼 클릭 후 결과 선택
            page.evaluate("""() => {
                Array.from(document.querySelectorAll('button'))
                    .find(b => b.innerText.trim() === '검색')?.click();
            }""")
            time.sleep(2)
            clicked = page.evaluate("""() => {
                const items = document.querySelectorAll('.react-autosuggest__suggestions-list li');
                for (const item of items) {
                    const text = item.innerText.trim();
                    if (text !== '국내' && text !== '해외' && text.length > 0) {
                        item.click();
                        return text;
                    }
                }
                return null;
            }""")

        if not clicked:
            print(f"  ! 장소 검색 결과 없음: {place_name}")
            page.keyboard.press("Escape")
            return False

        time.sleep(1.5)

        # ── 4. "추가" 버튼 — dispatchEvent ──────────────────
        # 버튼이 hidden 상태라 Playwright native click 불가 → dispatchEvent 사용
        add_result = page.evaluate("""() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const addBtn = btns.find(b => b.innerText.trim() === '추가');
            if (!addBtn) return 'not_found';
            addBtn.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
            return 'dispatched';
        }""")
        print(f"  추가 버튼: {add_result}")
        time.sleep(1.5)

        # ── 5. "지도에 직접 표시" 체크 ──────────────────────
        try:
            is_checked = page.evaluate(
                "() => document.querySelector('#place-custom-drop-pin')?.checked"
            )
            if not is_checked:
                # label 클릭 우선 (React 이벤트 호환)
                label_clicked = page.evaluate("""() => {
                    const label = document.querySelector('label[for="place-custom-drop-pin"]');
                    if (label) { label.click(); return true; }
                    return false;
                }""")
                if not label_clicked:
                    page.evaluate("""() => {
                        const cb = document.querySelector('#place-custom-drop-pin');
                        if (cb) cb.dispatchEvent(
                            new MouseEvent('click', {bubbles: true, cancelable: true})
                        );
                    }""")
                time.sleep(0.8)
                checked_now = page.evaluate(
                    "() => document.querySelector('#place-custom-drop-pin')?.checked"
                )
                print(f"  지도에 직접 표시: {'✓' if checked_now else '✗'}")
        except Exception as e:
            print(f"  ! 지도에 직접 표시 체크 실패: {e}")

        # ── 6. "확인" 버튼 — dispatchEvent ──────────────────
        time.sleep(0.5)
        confirm_result = page.evaluate("""() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const confirmBtn = btns.find(b => b.innerText.trim() === '확인');
            if (!confirmBtn) return 'not_found';
            if (confirmBtn.disabled) return 'disabled';
            confirmBtn.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
            return 'dispatched';
        }""")
        print(f"  확인 버튼: {confirm_result}")
        time.sleep(2)

        # ── 7. 본문에 지도 카드 삽입 확인 ────────────────────
        has_map = page.evaluate("""() => {
            return document.querySelector(
                '.se-map-component, .se-module-map, [class*="se-map"]'
            ) !== null;
        }""")
        print(f"  본문 지도 카드: {'✓ 삽입됨' if has_map else '✗ 미삽입'}")

        print(f"  ✓ 장소 추가 완료: {clicked}")
        page.keyboard.press("End")
        page.keyboard.press("Enter")
        time.sleep(0.5)
        return True

    except Exception as e:
        print(f"  ✗ 장소 추가 실패: {e}")
        try:
            page.keyboard.press("Escape")
        except Exception:
            pass
        return False


def main():
    parser = argparse.ArgumentParser(description="네이버 블로그 자동 포스팅")
    parser.add_argument("--blog-id", default="blog10_04", help="네이버 블로그 ID")
    parser.add_argument("--title", required=True, help="포스트 제목")
    parser.add_argument("--content", required=True, help="본문 ([그림N], [장소] 마커 포함)")
    parser.add_argument("--images", default="", help="이미지 경로 (콤마 구분)")
    parser.add_argument("--tags", default="", help="태그 (콤마 구분)")
    parser.add_argument("--map-url", default="", help="네이버 지도 링크 (장소 자동 추가)")
    parser.add_argument("--no-publish", action="store_true", help="발행 안 하고 확인만")
    parser.add_argument("--caller-chat-id", default="", help="발신자 채널 ID (인증용, 예: telegram:8279476113)")
    args = parser.parse_args()

    # ── 인증 확인 ─────────────────────────────────────────────
    check_authorization(args.caller_chat_id or None)

    images = [p.strip() for p in args.images.split(",") if p.strip()]
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    # 지도 링크에서 장소명 추출
    place_name = None
    if args.map_url:
        print("장소명 추출 중...")
        place_name = get_place_name_from_map_url(args.map_url)

    # [장소] 마커가 없으면 content 끝에 자동 추가
    content = args.content
    if place_name and "[장소]" not in content:
        content = content.rstrip() + "\n\n[장소]"

    content_parts = split_content(content, images)
    cookies = load_cookies()
    write_url = f"https://blog.naver.com/PostWriteForm.naver?blogId={args.blog_id}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        ctx.add_cookies(cookies)
        page = ctx.new_page()

        print("블로그 에디터 열기...")
        page.goto(write_url, wait_until="networkidle")
        time.sleep(4)

        if "login" in page.url or "nid.naver.com" in page.url:
            print("오류: 쿠키 만료. login_naver.py 재실행 필요.")
            browser.close()
            sys.exit(1)

        # ── 0. 초기 팝업 처리 ─────────────────────────────────
        try:
            if page.locator(".se-popup-alert-confirm").count() > 0:
                print("  초기 팝업 → 취소 (새 글 시작)")
                page.locator("button.se-popup-button-cancel").click(force=True, timeout=3000)
                time.sleep(1)
        except Exception:
            pass

        # ── 1. 제목 입력 ──────────────────────────────────────
        print("제목 입력 중...")
        try:
            page.locator(".se-title-text").first.click(timeout=8000)
            time.sleep(0.5)
            paste_text(page, args.title)
            print("  ✓ 제목 완료")
        except Exception as e:
            print(f"  ✗ 제목 실패: {e}")

        # ── 2. 본문 포커스 ────────────────────────────────────
        try:
            page.locator(".se-component.se-text .se-module-text").first.click(timeout=8000)
            time.sleep(0.5)
        except Exception:
            pass

        # ── 3. 본문 + 이미지 + 장소 순서대로 삽입 ───────────
        print("본문 입력 중...")
        for idx, (text, img_path, is_place) in enumerate(content_parts):
            if text:
                paste_text(page, text)
                page.keyboard.press("Enter")
                time.sleep(0.3)

            if is_place and place_name:
                add_place(page, place_name)

            if img_path:
                print(f"  이미지 {idx + 1} 업로드: {Path(img_path).name}")
                upload_image(page, img_path)
                page.keyboard.press("End")
                page.keyboard.press("Enter")
                time.sleep(0.5)

        print("  ✓ 본문 완료")

        # ── 4. 발행 패널 열기 ─────────────────────────────────
        print("발행 패널 열기...")
        page.evaluate("""() => {
            Array.from(document.querySelectorAll('button'))
                .find(b => b.innerText.trim() === '발행')?.click();
        }""")
        time.sleep(3)

        # ── 5. 태그 입력 ─────────────────────────────────────
        if tags:
            print(f"태그 입력: {tags}")
            try:
                page.locator("#tag-input").wait_for(timeout=8000)
                page.evaluate("document.querySelector('#tag-input').focus()")
                time.sleep(0.3)
                for tag in tags:
                    pyperclip.copy(tag)
                    page.keyboard.press("Meta+v")
                    time.sleep(0.2)
                    page.keyboard.press("Enter")
                    time.sleep(0.3)
                print("  ✓ 태그 완료")
            except Exception as e:
                print(f"  ✗ 태그 실패: {e}")

        # ── 6. 최종 발행 ─────────────────────────────────────
        if args.no_publish:
            print("\n[--no-publish] 10초 후 닫힘.")
            time.sleep(10)
        else:
            print("최종 발행 중...")
            try:
                page.evaluate("""() => {
                    const btns = Array.from(document.querySelectorAll('button'))
                        .filter(b => b.innerText.trim() === '발행');
                    if (btns.length >= 2) btns[btns.length - 1].click();
                    else if (btns.length === 1) btns[0].click();
                }""")
                try:
                    page.wait_for_url(re.compile(r"blog\.naver\.com/\w+/\d+"), timeout=15000)
                    print(f"\n✅ 발행 완료! URL: {page.url}")
                except PlaywrightTimeoutError:
                    print(f"\n발행 시도 완료 (현재: {page.url})")
            except Exception as e:
                print(f"발행 실패: {e}")

        browser.close()


if __name__ == "__main__":
    main()
