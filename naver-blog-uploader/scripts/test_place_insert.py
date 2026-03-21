"""
장소 삽입 단독 테스트 스크립트
흐름: 장소 클릭 → 장소명 입력 → 추가(dispatchEvent) → 지도에 직접 표시 → 확인(dispatchEvent) → 지도 카드 확인
"""
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

COOKIE_PATH = Path.home() / ".openclaw" / "naver_cookies.json"
BLOG_ID = "blog10_04"
PLACE_NAME = "위트앤미트 강남점"


def main():
    with open(COOKIE_PATH) as f:
        cookies = json.load(f)

    write_url = f"https://blog.naver.com/PostWriteForm.naver?blogId={BLOG_ID}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=300)
        ctx = browser.new_context()
        ctx.add_cookies(cookies)
        page = ctx.new_page()

        print("에디터 로딩...")
        page.goto(write_url, wait_until="networkidle")
        time.sleep(4)

        if "login" in page.url:
            print("쿠키 만료!")
            browser.close()
            return

        # 초기 팝업 처리
        try:
            cancel = page.locator("button.se-popup-button-cancel").first
            if cancel.is_visible():
                cancel.click()
                time.sleep(1)
        except Exception:
            pass

        print(f"\n[테스트] 장소 삽입: {PLACE_NAME}")

        # ── 1. 장소 툴바 버튼 클릭 ──
        print("1. 장소 버튼 클릭...")
        result = page.evaluate("""() => {
            const mapBtn = document.querySelector('.se-toolbar-item-map button');
            if (mapBtn) { mapBtn.click(); return 'map-toolbar'; }
            const allBtns = Array.from(document.querySelectorAll('button'));
            const placeBtn = allBtns.find(b => b.innerText.includes('장소'));
            if (placeBtn) { placeBtn.click(); return placeBtn.innerText.trim(); }
            return 'not_found';
        }""")
        print(f"   → {result}")
        time.sleep(2)

        # ── 2. 장소명 입력 ──
        print("2. 장소명 입력...")
        try:
            inp = page.locator(".react-autosuggest__input").first
            inp.wait_for(timeout=5000)
            inp.click()
            inp.fill(PLACE_NAME)
            print(f"   → '{PLACE_NAME}' 입력 완료")
            time.sleep(2)
        except Exception as e:
            print(f"   → 실패: {e}")
            browser.close()
            return

        # ── 3. 자동완성 선택 ──
        print("3. 자동완성 선택...")
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
            print("   → 자동완성 없음, 검색 버튼 클릭...")
            page.evaluate("() => Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === '검색')?.click()")
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
        print(f"   → 선택됨: {clicked}")
        if not clicked:
            print("   → 결과 없음, 종료")
            browser.close()
            return
        time.sleep(1.5)

        # ── 4. "추가" 버튼 dispatchEvent ──
        print("4. 추가 버튼 (dispatchEvent)...")
        add_state = page.evaluate("""() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const addBtn = btns.find(b => b.innerText.trim() === '추가');
            if (!addBtn) return {found: false};
            const before = {disabled: addBtn.disabled, display: getComputedStyle(addBtn).display, visibility: getComputedStyle(addBtn).visibility};
            addBtn.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
            return {found: true, before};
        }""")
        print(f"   → {add_state}")
        time.sleep(1.5)

        # ── 5. 확인 버튼 상태 확인 ──
        print("5. 확인 버튼 상태 확인...")
        confirm_state = page.evaluate("""() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const confirmBtn = btns.find(b => b.innerText.trim() === '확인');
            if (!confirmBtn) return {found: false};
            return {found: true, disabled: confirmBtn.disabled, text: confirmBtn.innerText.trim()};
        }""")
        print(f"   → {confirm_state}")

        # ── 6. "지도에 직접 표시" 체크 ──
        print("6. 지도에 직접 표시 체크...")
        check_state = page.evaluate("""() => {
            const cb = document.querySelector('#place-custom-drop-pin');
            if (!cb) return {found: false};
            const label = document.querySelector('label[for="place-custom-drop-pin"]');
            const was_checked = cb.checked;
            if (!was_checked) {
                if (label) label.click();
                else cb.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
            }
            return {found: true, was_checked, now_checked: cb.checked};
        }""")
        print(f"   → {check_state}")
        time.sleep(0.8)

        # ── 7. 확인 버튼 상태 재확인 ──
        print("7. 확인 버튼 재확인...")
        confirm_state2 = page.evaluate("""() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const confirmBtn = btns.find(b => b.innerText.trim() === '확인');
            if (!confirmBtn) return {found: false};
            return {found: true, disabled: confirmBtn.disabled};
        }""")
        print(f"   → {confirm_state2}")

        # ── 8. 확인 버튼 클릭 (dispatchEvent) ──
        print("8. 확인 버튼 클릭 (dispatchEvent)...")
        confirm_result = page.evaluate("""() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const confirmBtn = btns.find(b => b.innerText.trim() === '확인');
            if (!confirmBtn) return 'not_found';
            if (confirmBtn.disabled) return 'disabled';
            confirmBtn.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
            return 'dispatched';
        }""")
        print(f"   → {confirm_result}")
        time.sleep(2.5)

        # ── 9. 팝업 닫혔는지 + 본문 지도 카드 확인 ──
        print("9. 결과 확인...")
        result = page.evaluate("""() => {
            const popup = document.querySelector('.se-popup-map, .se-popup[class*="map"]');
            const popupVisible = popup ? getComputedStyle(popup).display !== 'none' : false;
            const mapComp = document.querySelector('.se-map-component, .se-module-map, [class*="se-map"]');
            return {
                popupOpen: popupVisible,
                mapInContent: mapComp !== null,
                mapClass: mapComp ? mapComp.className : null
            };
        }""")
        print(f"   → 팝업 열림: {result['popupOpen']}")
        print(f"   → 본문 지도 카드: {'✓ 삽입됨' if result['mapInContent'] else '✗ 미삽입'}")
        if result['mapClass']:
            print(f"   → 클래스: {result['mapClass']}")

        # 스크린샷
        page.screenshot(path="/tmp/test_place_result.png", full_page=False)
        print("\n스크린샷 저장: /tmp/test_place_result.png")

        print("\n5초 후 닫힘...")
        time.sleep(5)
        browser.close()


if __name__ == "__main__":
    main()
