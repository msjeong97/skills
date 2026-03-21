---
name: naver-blog-uploader
description: >
  Playwright를 이용해 네이버 블로그에 글과 사진을 자동으로 업로드한다.
  blog-auto-poster 스킬이 작성한 초안 + [그림N] 마커 + 사진 파일들을 받아
  네이버 SmartEditor에 텍스트 붙여넣기 + 사진 자동 업로드 + 발행까지 처리.
  사용 시점: 블로그 초안이 확정되어 네이버에 올릴 준비가 됐을 때.
---

# 네이버 블로그 자동 업로더

## 전체 워크플로우

```
[초안(텍스트) + 사진파일 목록([그림N] 순서대로) 입력]
  → [scripts/post_to_naver.py 실행]
  → [Playwright: 네이버 로그인 (쿠키 캐시)]
  → [글쓰기 페이지 진입]
  → [제목 입력]
  → [본문 클립보드 붙여넣기]
  → [[그림N] 마커 위치마다 사진 업로드]
  → [태그 입력]
  → [발행 클릭]
```

## 사용법

```bash
python3 scripts/post_to_naver.py \
  --title "성수 짜이 카페 높은산 | ..." \
  --content "본문 텍스트..." \
  --images "img1.jpg,img2.jpg,img3.jpg" \
  --tags "성수카페,짜이,마살라짜이" \
  --markers "[그림1],[그림2],[그림3]"
```

## 스크립트 파일

- `scripts/post_to_naver.py` — 메인 업로드 스크립트 (Playwright)
- `scripts/login_naver.py` — 최초 로그인 + 쿠키 저장
- `scripts/requirements.txt` — Python 의존성

## 설정

- 쿠키 파일: `~/.openclaw/naver_cookies.json` (최초 1회 로그인 후 자동 저장)
- 로그인 만료 시: `python3 scripts/login_naver.py` 재실행

## 참고 파일

- `../blog-auto-poster/` — 글 작성 스킬 (본 스킬과 함께 사용)
