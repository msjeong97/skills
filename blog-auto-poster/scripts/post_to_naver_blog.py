#!/usr/bin/env python3
# pyenv virtualenv: py3.14-openclaw
# 실행: PYENV_VERSION=py3.14-openclaw python post_to_naver_blog.py [args]
"""
Playwright 기반 네이버 블로그 글쓰기 자동화 스크립트.

핵심 동작:
- 기존 로그인된 Chrome 프로필 사용 (재로그인 불필요)
- 제목: .se-title-text → iframe body ClipboardEvent paste (text/plain만)
- 본문: iframe body ClipboardEvent paste (text/plain만, text/html 제외 → 한글 깨짐 방지)
- 이미지: expect_file_chooser로 안정적 업로드 → 본문 아래에 순서대로 배치
- 태그: 발행 패널 열고 type + Enter

Usage:
  python3 post_to_naver_blog.py \
    --title "제목" \
    --content "본문 내용" \
    --images "/path/img1.jpg,/path/img2.jpg" \
    --tags "태그1,태그2" \
    --blog-id blog10_04
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from playwright.async_api import async_playwright, Page


CHROME_PROFILE = str(Path.home() / "Library/Application Support/Google/Chrome/Default")
WRITE_URL_TEMPLATE = "https://blog.naver.com/{blog_id}/postwrite"


async def paste_to_iframe(page: Page, text: str):
    """iframe body에 ClipboardEvent paste (text/plain만 — 한글 깨짐 방지)"""
    safe_text = text.replace("\\", "\\\\").replace("`", "\\`")
    await page.evaluate(f"""
        async () => {{
            const iframe = document.querySelector('iframe');
            if (!iframe || !iframe.contentDocument) throw new Error('iframe 없음');
            const dt = new DataTransfer();
            dt.setData('text/plain', `{safe_text}`);
            const ev = new ClipboardEvent('paste', {{
                bubbles: true, cancelable: true, clipboardData: dt
            }});
            iframe.contentDocument.body.dispatchEvent(ev);
            await new Promise(r => setTimeout(r, 400));
        }}
    """)


async def input_title(page: Page, title: str):
    """제목 영역에 텍스트 입력"""
    print(f"[제목] 입력 중: {title}", file=sys.stderr)
    safe_title = title.replace("\\", "\\\\").replace("`", "\\`")
    await page.evaluate(f"""
        async () => {{
            const titleEl = document.querySelector('.se-title-text');
            if (!titleEl) throw new Error('제목 요소 없음');
            titleEl.click();
            await new Promise(r => setTimeout(r, 300));
            const iframe = document.querySelector('iframe');
            const dt = new DataTransfer();
            dt.setData('text/plain', `{safe_title}`);
            const ev = new ClipboardEvent('paste', {{
                bubbles: true, cancelable: true, clipboardData: dt
            }});
            if (iframe && iframe.contentDocument) {{
                iframe.contentDocument.body.dispatchEvent(ev);
            }}
            await new Promise(r => setTimeout(r, 300));
        }}
    """)


async def input_body(page: Page, content: str):
    """본문 클릭 후 텍스트 붙여넣기"""
    print(f"[본문] {len(content)}자 입력 중...", file=sys.stderr)
    # 본문 영역으로 커서 이동
    try:
        await page.locator('.se-component.se-text').first.click(timeout=3000)
    except Exception:
        pass
    await page.wait_for_timeout(300)
    await paste_to_iframe(page, content)


async def upload_image(page: Page, image_path: str):
    """
    사진 추가 버튼 클릭 → expect_file_chooser로 파일 선택.
    Playwright가 OS 파일 다이얼로그를 안정적으로 처리함.
    """
    abs_path = str(Path(image_path).resolve())
    if not Path(abs_path).exists():
        print(f"[경고] 파일 없음, 건너뜀: {abs_path}", file=sys.stderr)
        return False

    print(f"[이미지] 업로드: {Path(abs_path).name}", file=sys.stderr)

    try:
        # 파일 선택 다이얼로그를 기다리면서 버튼 클릭
        async with page.expect_file_chooser(timeout=10000) as fc_info:
            await page.locator(
                'button:has-text("사진 추가"), '
                'button[aria-label="사진 추가"], '
                'button[data-type="image"]'
            ).first.click()
        file_chooser = await fc_info.value
        await file_chooser.set_files(abs_path)
        # 업로드 완료 대기 (썸네일 생성 등)
        await page.wait_for_timeout(3000)
        print(f"[이미지] 완료: {Path(abs_path).name}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"[경고] 이미지 업로드 실패 ({Path(abs_path).name}): {e}", file=sys.stderr)
        return False


async def upload_all_images(page: Page, image_paths: list[str]):
    """모든 이미지를 순서대로 업로드 (본문 커서 위치에 순서대로 삽입됨)"""
    if not image_paths:
        return
    print(f"[이미지] 총 {len(image_paths)}장 업로드 시작", file=sys.stderr)
    for path in image_paths:
        await upload_image(page, path)
        await page.wait_for_timeout(500)


async def add_tags_and_open_publish(page: Page, tags: list[str]):
    """발행 버튼 클릭 → 발행 패널에서 태그 입력"""
    # 발행 패널 열기
    await page.locator('button:has-text("발행")').first.click(timeout=5000)
    await page.wait_for_timeout(1000)

    if not tags:
        return

    tag_input = page.locator('input[placeholder*="태그"]').first
    for tag in tags:
        clean_tag = tag.strip().lstrip("#")
        if not clean_tag:
            continue
        try:
            await tag_input.click()
            await tag_input.type(clean_tag, delay=60)
            await page.keyboard.press("Enter")
            await page.wait_for_timeout(300)
            print(f"[태그] #{clean_tag}", file=sys.stderr)
        except Exception as e:
            print(f"[경고] 태그 실패 ({clean_tag}): {e}", file=sys.stderr)


async def publish(page: Page):
    """발행 패널의 최종 발행 버튼 클릭"""
    try:
        # 발행 패널 안의 '발행' 버튼 (툴바의 발행이 아닌 패널 내 버튼)
        await page.locator('button:has-text("발행")').last.click(timeout=5000)
        await page.wait_for_timeout(3000)
        current_url = page.url
        print(f"[완료] 포스팅 발행 성공! URL: {current_url}", file=sys.stderr)
        return current_url
    except Exception as e:
        print(f"[오류] 발행 실패: {e}", file=sys.stderr)
        raise


async def post_blog(
    blog_id: str,
    title: str,
    content: str,
    images: list[str],
    tags: list[str],
):
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=CHROME_PROFILE,
            headless=False,
            channel="chrome",
            args=["--start-maximized"],
        )
        page = browser.pages[0] if browser.pages else await browser.new_page()

        write_url = WRITE_URL_TEMPLATE.format(blog_id=blog_id)
        print(f"[시작] {write_url}", file=sys.stderr)
        await page.goto(write_url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(2000)

        # 1. 제목 입력
        await input_title(page, title)
        await page.wait_for_timeout(400)

        # 2. 본문 입력
        await input_body(page, content)
        await page.wait_for_timeout(400)

        # 3. 이미지 업로드 (본문 아래 커서 위치에 순서대로 삽입)
        if images:
            # 본문 끝으로 커서 이동
            try:
                await page.locator('.se-component.se-text').last.click(timeout=3000)
                await page.keyboard.press("End")
                await page.wait_for_timeout(300)
            except Exception:
                pass
            await upload_all_images(page, images)

        # 4. 태그 입력 + 발행 패널 열기
        await add_tags_and_open_publish(page, tags)

        # 5. 최종 발행
        url = await publish(page)

        await browser.close()
        return url


def main():
    parser = argparse.ArgumentParser(description="네이버 블로그 자동 포스터")
    parser.add_argument("--title",    required=True, help="블로그 글 제목")
    parser.add_argument("--content",  required=True, help="본문 텍스트 (또는 파일 경로)")
    parser.add_argument("--images",   default="",    help="이미지 절대경로 (쉼표 구분)")
    parser.add_argument("--tags",     default="",    help="태그 (쉼표 구분)")
    parser.add_argument("--blog-id",  default="blog10_04")
    args = parser.parse_args()

    # content가 파일 경로면 읽기
    content = args.content
    if os.path.exists(content):
        with open(content, encoding="utf-8") as f:
            content = f.read()

    images = [i.strip() for i in args.images.split(",") if i.strip()]
    tags   = [t.strip() for t in args.tags.split(",")   if t.strip()]

    asyncio.run(post_blog(
        blog_id=args.blog_id,
        title=title := args.title,
        content=content,
        images=images,
        tags=tags,
    ))


if __name__ == "__main__":
    main()
