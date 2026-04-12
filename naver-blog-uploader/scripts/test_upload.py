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

POST_SCRIPT = Path(__file__).parent / "post_to_naver.py"
DELETE_SCRIPT = Path(__file__).parent / "delete_post.py"

TEST_TITLE_TEMPLATE = "[TEST] 자동화 테스트 - {ts}"
TEST_CONTENT = "이 글은 자동화 테스트용입니다. 자동 삭제됩니다."
TEST_TAGS = "자동화테스트,삭제예정"


def run_upload(title: str) -> str | None:
    """post_to_naver.py 실행 → logNo 반환. 실패 시 None."""
    cmd = [
        sys.executable, str(POST_SCRIPT),
        "--title", title,
        "--content", TEST_CONTENT,
        "--tags", TEST_TAGS,
    ]

    print("업로드 실행 중...")
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
    return result.returncode == 0


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
        print("\n[--no-delete] 삭제 건너뜀. 수동으로 확인 후 삭제하세요.")
        print(f"  삭제 명령: python {DELETE_SCRIPT} --log-no {log_no}")
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
