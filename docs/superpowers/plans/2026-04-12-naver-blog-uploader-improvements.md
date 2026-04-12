# Naver Blog Uploader — Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 이번 세션 디버깅 과정에서 발견된 문제들을 수정하고, 불필요한 인증을 제거하며, 테스트 자동화를 추가한다.

**Architecture:** 모든 변경 대상은 `/Users/minseop/.openclaw/skills/naver-blog-uploader/scripts/` 이하 파일이다. 쿠키/설정 파일을 skill 디렉토리 내 `config/` 폴더로 이전하고, 텔레그램 인증 레이어를 제거하며, 테스트 + 자동삭제 스크립트를 신규 추가한다.

**Tech Stack:** Python 3, Playwright (sync), pyperclip, requests

---

## 배경: 이번 세션에서 발견된 문제들

| # | 문제 | 원인 | 상태 |
|---|------|------|------|
| 1 | 최종 발행 버튼 클릭 불가 | `container__HW_tc` 도움말 패널이 pointer-events로 가림 | ✅ 이미 수정됨 |
| 2 | 쿠키 경로가 `~/.openclaw`에 하드코딩 | Telegram 봇 환경 기반 설계 | 이번 플랜에서 수정 |
| 3 | `--caller-chat-id` 텔레그램 인증 필수 | 멀티유저 Telegram 봇 보호용 — 로컬 Claude Code엔 불필요 | 이번 플랜에서 제거 |
| 4 | 디버깅 흔적 코드 잔존 | 세션 중 추가한 screenshot, print 코드 | 이번 플랜에서 정리 |
| 5 | 테스트 글 수동 삭제 필요 | 삭제 스크립트 없음 | 이번 플랜에서 추가 |

---

## 파일 구조 (변경 후)

```
/Users/minseop/.openclaw/skills/naver-blog-uploader/
├── config/
│   ├── .gitignore          # NEW: naver_cookies.json 제외
│   ├── naver_cookies.json  # MOVED: ~/.openclaw/naver_cookies.json → 여기로
│   └── settings.json       # NEW: blog_id 등 설정 (config.json 대체)
├── scripts/
│   ├── post_to_naver.py    # MODIFY: 인증 제거, 경로 수정, 코드 정리
│   ├── login_naver.py      # MODIFY: 쿠키 저장 경로 변경
│   ├── delete_post.py      # NEW: 테스트 글 자동삭제 (엄격한 조건)
│   ├── test_upload.py      # NEW: 업로드 → 검증 → 삭제 E2E 테스트
│   └── requirements.txt    # 그대로
└── SKILL.md
```

---

## Task 1: config/ 디렉토리 생성 + 쿠키 경로 이전

**Files:**
- Create: `config/.gitignore`
- Create: `config/settings.json`
- Modify: `scripts/login_naver.py`
- Modify: `scripts/post_to_naver.py` — COOKIE_PATH, CONFIG_PATH 상수

- [ ] **Step 1: config/ 디렉토리와 .gitignore 생성**

```bash
mkdir -p /Users/minseop/.openclaw/skills/naver-blog-uploader/config
```

`config/.gitignore` 내용:
```
naver_cookies.json
```

- [ ] **Step 2: 기존 config.json → settings.json으로 이전 (불필요 필드 제거)**

기존 `config.json`:
```json
{
  "authorized_chat_ids": ["telegram:8279476113", "telegram:8756670648"],
  "blog_id": "blog10_04"
}
```

새 `config/settings.json` (텔레그램 인증 필드 제거):
```json
{
  "blog_id": "blog10_04"
}
```

```bash
# config/settings.json 작성
cat > /Users/minseop/.openclaw/skills/naver-blog-uploader/config/settings.json << 'EOF'
{
  "blog_id": "blog10_04"
}
EOF
```

- [ ] **Step 3: 기존 쿠키 파일 복사**

```bash
cp ~/.openclaw/naver_cookies.json \
   /Users/minseop/.openclaw/skills/naver-blog-uploader/config/naver_cookies.json
```

- [ ] **Step 4: `login_naver.py` 쿠키 저장 경로 변경**

`scripts/login_naver.py`에서:
```python
# 변경 전
COOKIE_PATH = Path.home() / ".openclaw" / "naver_cookies.json"

# 변경 후
COOKIE_PATH = Path(__file__).parent.parent / "config" / "naver_cookies.json"
```

- [ ] **Step 5: `post_to_naver.py` 경로 상수 변경**

`scripts/post_to_naver.py`에서:
```python
# 변경 전
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
# ...
COOKIE_PATH = Path.home() / ".openclaw" / "naver_cookies.json"

# 변경 후
_SKILL_DIR = Path(__file__).parent.parent
CONFIG_PATH = _SKILL_DIR / "config" / "settings.json"
COOKIE_PATH = _SKILL_DIR / "config" / "naver_cookies.json"
```

- [ ] **Step 6: 동작 검증**

```bash
cd /Users/minseop/.openclaw/skills/naver-blog-uploader
python3 scripts/login_naver.py --help 2>/dev/null || python3 -c "
from pathlib import Path
import json
p = Path('config/settings.json')
print('settings.json exists:', p.exists())
print('blog_id:', json.loads(p.read_text()).get('blog_id'))
c = Path('config/naver_cookies.json')
print('cookies exist:', c.exists())
"
```

기대 출력:
```
settings.json exists: True
blog_id: blog10_04
cookies exist: True
```

- [ ] **Step 7: `test_place_insert.py` 경로도 업데이트**

`scripts/test_place_insert.py`에서:
```python
# 변경 전
COOKIE_PATH = Path.home() / ".openclaw" / "naver_cookies.json"
BLOG_ID = "blog10_04"

# 변경 후
_SKILL_DIR = Path(__file__).parent.parent
COOKIE_PATH = _SKILL_DIR / "config" / "naver_cookies.json"
_settings = json.loads((_SKILL_DIR / "config" / "settings.json").read_text())
BLOG_ID = _settings["blog_id"]
```

---

## Task 2: 텔레그램 인증 제거 + `blog_id` 자동 로드

**Files:**
- Modify: `scripts/post_to_naver.py`

**텔레그램 인증이 있었던 이유:** 원래 이 스크립트는 Telegram 봇이 여러 사용자 메시지를 받아 실행하는 환경을 위해 설계됐다. 로컬 Claude Code 환경에서는 호출 자체가 이미 인증된 환경이므로 불필요하다.

- [ ] **Step 1: `check_authorization()` 함수 전체 제거**

`scripts/post_to_naver.py`에서 아래 코드 블록 전체 삭제:
```python
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
```

- [ ] **Step 2: `main()` 내 argparse 정리**

제거할 인자:
```python
# 삭제
parser.add_argument("--blog-id", ...)
parser.add_argument("--caller-chat-id", ...)
```

`blog_id`는 settings.json에서 읽도록 변경:
```python
def load_settings() -> dict:
    if not CONFIG_PATH.exists():
        print("오류: config/settings.json 없음.")
        sys.exit(1)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)
```

`main()` 상단:
```python
settings = load_settings()
blog_id = settings["blog_id"]
write_url = f"https://blog.naver.com/PostWriteForm.naver?blogId={blog_id}"
```

- [ ] **Step 3: 인증 호출 라인 제거**

```python
# 삭제
check_authorization(args.caller_chat_id or None)
```

- [ ] **Step 4: 실행 확인 (--help 출력)**

```bash
cd /Users/minseop/.openclaw/skills/naver-blog-uploader
python3 scripts/post_to_naver.py --help
```

기대 출력 (`--caller-chat-id` 없음):
```
usage: post_to_naver.py [-h] --title TITLE --content CONTENT
                        [--images IMAGES] [--tags TAGS]
                        [--map-url MAP_URL] [--no-publish]
```

---

## Task 3: 코드 정리 (디버깅 흔적 제거)

**Files:**
- Modify: `scripts/post_to_naver.py`

- [ ] **Step 1: `--no-publish` 경로 정리**

현재 `--no-publish`는 `/tmp/naver_before_publish.png`를 저장함. 이 side-effect 제거:

```python
# 변경 전
if args.no_publish:
    print("\n[--no-publish] 스크린샷 저장 후 닫힘.")
    page.screenshot(path="/tmp/naver_before_publish.png", full_page=False)
    print("  스크린샷: /tmp/naver_before_publish.png")
    time.sleep(3)

# 변경 후
if args.no_publish:
    print("\n[--no-publish] 발행 없이 종료.")
```

- [ ] **Step 2: 발행 완료 URL 정규식 개선**

실제 발행 URL이 `PostView.naver?...` 형태임을 확인. 정규식 업데이트:

```python
# 변경 전
page.wait_for_url(re.compile(r"blog\.naver\.com/(?!PostWrite)\w"), timeout=15000)

# 변경 후 (더 명확하게)
page.wait_for_url(re.compile(r"blog\.naver\.com/(PostView|[a-zA-Z0-9_]+/\d+)"), timeout=15000)
```

- [ ] **Step 3: logNo 추출 + 출력 추가**

발행 완료 후 logNo를 파싱해서 출력:

```python
url = page.url
log_no_match = re.search(r"logNo=(\d+)", url)
log_no = log_no_match.group(1) if log_no_match else "unknown"
print(f"\n✅ 발행 완료!")
print(f"  logNo: {log_no}")
print(f"  URL: https://blog.naver.com/{blog_id}/{log_no}")
```

- [ ] **Step 4: viewport 원복 — headless=True + 적정 viewport**

```python
# 그대로 유지 (이미 이번 세션에서 수정됨)
browser = p.chromium.launch(headless=True)
ctx = browser.new_context(viewport={"width": 1440, "height": 900})
```

---

## Task 4: `delete_post.py` — 테스트 글 자동삭제 스크립트

**Files:**
- Create: `scripts/delete_post.py`

**안전 규칙 (엄격):**
1. `--log-no` 인자가 반드시 필요 (숫자만 허용)
2. 삭제 전 해당 글의 제목을 읽어서 `[TEST]`로 시작하는지 확인
3. 조건 미충족 시 삭제 거부 + 오류 메시지
4. `--dry-run` 플래그로 실제 삭제 없이 검증만 가능

- [ ] **Step 1: `delete_post.py` 작성**

```python
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
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_cookies() -> list[dict]:
    if not COOKIE_PATH.exists():
        print("오류: 쿠키 없음. login_naver.py 재실행 필요.")
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
    except Exception:
        pass

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
        if not title.startswith(TEST_TITLE_PREFIX):
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
```

- [ ] **Step 2: dry-run으로 안전 조건 검증**

테스트 글이 있다면:
```bash
cd /Users/minseop/.openclaw/skills/naver-blog-uploader
python3 scripts/delete_post.py --log-no 224249538351 --dry-run
```

일반 글에 대해 거부 테스트:
```bash
# 실제 글 logNo로 시도 → 거부돼야 함
python3 scripts/delete_post.py --log-no 224249539694 --dry-run
```

기대 출력:
```
글 제목 확인 중... (logNo: 224249539694)
  글 제목: '광진구 닭갈비 맛집 산골닭갈비 | ...'
거부: 제목이 '[TEST]'로 시작하지 않습니다.
  → 테스트 글만 자동삭제 가능합니다.
```

---

## Task 5: `test_upload.py` — E2E 테스트 + 자동삭제

**Files:**
- Create: `scripts/test_upload.py`

테스트 포스트는 반드시 `[TEST]`로 시작하는 제목을 사용. 발행 성공 후 logNo를 받아 `delete_post.py`를 호출해 즉시 삭제.

- [ ] **Step 1: `test_upload.py` 작성**

```python
#!/usr/bin/env python3
"""
네이버 블로그 업로더 E2E 테스트.

1. 테스트 글 발행 (제목: "[TEST] 자동화 테스트 - {timestamp}")
2. logNo 추출 + URL 검증
3. delete_post.py 호출로 자동삭제
4. 삭제 확인

사용법:
    python test_upload.py
    python test_upload.py --no-delete   # 삭제 건너뜀 (수동 확인용)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

POST_SCRIPT = Path(__file__).parent / "post_to_naver.py"
DELETE_SCRIPT = Path(__file__).parent / "delete_post.py"

TEST_TITLE_TEMPLATE = "[TEST] 자동화 테스트 - {ts}"
TEST_CONTENT = "이 글은 자동화 테스트용입니다. 자동 삭제됩니다."
TEST_TAGS = "자동화테스트,삭제예정"
TEST_IMAGE = str(_SKILL_DIR / "config" / "test_image.jpg")  # 없으면 이미지 없이 진행


def run_upload(title: str) -> str | None:
    """post_to_naver.py 실행 → logNo 반환. 실패 시 None."""
    cmd = [
        sys.executable, str(POST_SCRIPT),
        "--title", title,
        "--content", TEST_CONTENT,
        "--tags", TEST_TAGS,
    ]
    # 테스트 이미지가 있으면 추가
    if Path(TEST_IMAGE).exists():
        cmd += ["--images", TEST_IMAGE]

    print(f"업로드 실행 중...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    output = result.stdout + result.stderr
    print(output)

    if result.returncode != 0:
        print(f"❌ 업로드 실패 (returncode={result.returncode})")
        return None

    # logNo 파싱
    match = re.search(r"logNo[=:]?\s*(\d+)", output)
    if not match:
        print("❌ logNo를 출력에서 찾지 못했습니다.")
        return None

    return match.group(1)


def run_delete(log_no: str) -> bool:
    """delete_post.py 실행 → 성공 여부 반환."""
    cmd = [sys.executable, str(DELETE_SCRIPT), "--log-no", log_no]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    output = result.stdout + result.stderr
    print(output)
    return result.returncode == 0 and "삭제 완료" in output


def main() -> None:
    parser = argparse.ArgumentParser(description="블로그 업로더 E2E 테스트")
    parser.add_argument("--no-delete", action="store_true", help="삭제 단계 건너뜀")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    title = TEST_TITLE_TEMPLATE.format(ts=ts)

    print("=" * 50)
    print(f"테스트 시작: {title}")
    print("=" * 50)

    # Step 1: 업로드
    log_no = run_upload(title)
    if not log_no:
        print("\n❌ 테스트 실패: 업로드 단계")
        sys.exit(1)

    print(f"\n✅ 업로드 성공: logNo={log_no}")

    if args.no_delete:
        print(f"\n[--no-delete] 삭제 건너뜀. 수동으로 확인 후 삭제하세요.")
        print(f"  삭제 명령: python delete_post.py --log-no {log_no}")
        return

    # Step 2: 삭제
    print(f"\n삭제 진행 중... (logNo={log_no})")
    time.sleep(2)  # 발행 직후 바로 삭제 시 API 반영 대기
    success = run_delete(log_no)

    if success:
        print(f"\n✅ 테스트 완료: 업로드 + 삭제 모두 성공")
    else:
        print(f"\n⚠️  삭제 실패. 수동 삭제 필요: logNo={log_no}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: E2E 테스트 실행**

```bash
cd /Users/minseop/.openclaw/skills/naver-blog-uploader
python3 scripts/test_upload.py
```

기대 흐름:
```
==================================================
테스트 시작: [TEST] 자동화 테스트 - 20260412-143000
==================================================
업로드 실행 중...
  ✓ 제목 완료
  ✓ 본문 완료
  발행 패널 열기...
  ✓ 태그 완료
  최종 발행 중...
  ✅ 발행 완료!
  logNo: 224249XXXXXX

✅ 업로드 성공: logNo=224249XXXXXX

삭제 진행 중...
글 제목 확인 중... (logNo: 224249XXXXXX)
  글 제목: '[TEST] 자동화 테스트 - 20260412-143000'
삭제 진행: '[TEST] 자동화 테스트 - 20260412-143000'
✅ 삭제 완료: logNo=224249XXXXXX

✅ 테스트 완료: 업로드 + 삭제 모두 성공
```

- [ ] **Step 3: `--no-delete` 플래그로 삭제 안전성 확인**

```bash
python3 scripts/test_upload.py --no-delete
```

출력에 logNo가 나오면, 해당 logNo로 delete_post.py --dry-run 실행:
```bash
python3 scripts/delete_post.py --log-no <logNo> --dry-run
# → 제목이 [TEST]로 시작 → 삭제 허용 메시지
```

실제 삭제:
```bash
python3 scripts/delete_post.py --log-no <logNo>
# → ✅ 삭제 완료
```

---

## Self-Review

**Spec coverage 체크:**
- [x] 1. 에러 수정 — Task 3에서 pointer-events fix 유지, URL 정규식 개선
- [x] 2. ~/.openclaw → skill dir 이전 — Task 1에서 config/ 폴더로 이전, .gitignore 추가
- [x] 3. 불필요한 코드/인자 간소화 — Task 2에서 텔레그램 인증 제거, blog_id 자동 로드
- [x] 4. 테스트 코드 + 자동삭제 — Task 4 (delete_post.py) + Task 5 (test_upload.py)

**Placeholder scan:** 없음 — 모든 코드 블록 완성됨

**Type consistency:** `log_no`는 str로 일관되게 처리됨 (정규식, subprocess 인자 모두 str)

**추가 고려사항:**
- `delete_post.py`의 `PostDelete.naver` 엔드포인트가 실제 작동하는지 확인 필요 (Task 4 Step 2에서 검증)
- 테스트 이미지가 없어도 업로드는 진행됨 (이미지 없는 경우 처리됨)
- 기존 `test_place_insert.py`는 건드리지 않음 (별도 목적의 스크립트)
