#!/usr/bin/env python3
# pyenv virtualenv: py3.14-openclaw
# 실행: PYENV_VERSION=py3.14-openclaw python fetch_place_info.py <url>
"""
네이버 지도 링크에서 장소 정보를 추출하는 스크립트.
Usage: python3 fetch_place_info.py <naver_map_url>
"""

import sys
import json
import re
import requests
from bs4 import BeautifulSoup


def resolve_short_url(url: str) -> str:
    """naver.me 단축 URL을 실제 URL로 변환"""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=10)
        return resp.url
    except Exception:
        return url


def extract_place_id(url: str) -> str | None:
    """URL에서 네이버 지도 place ID 추출"""
    patterns = [
        r'/place/(\d+)',
        r'entry/place/(\d+)',
        r'placeid=(\d+)',
        r'/(\d{10,})',
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def fetch_place_info(place_id: str) -> dict:
    """네이버 지도 API로 장소 정보 가져오기"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://map.naver.com/",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }

    result = {
        "place_id": place_id,
        "name": "",
        "category": "",
        "address": "",
        "phone": "",
        "hours": "",
        "menu": [],
        "review_keywords": [],
        "description": "",
        "url": f"https://map.naver.com/v5/entry/place/{place_id}",
    }

    # 기본 정보
    api_url = f"https://map.naver.com/v5/api/sites/summary/{place_id}?lang=ko"
    try:
        resp = requests.get(api_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            result["name"] = data.get("name", "")
            result["category"] = data.get("category", "")
            result["address"] = data.get("roadAddress", data.get("address", ""))
            result["phone"] = data.get("phone", "")
            result["description"] = data.get("description", "")

            # 영업시간
            business_hours = data.get("businessHours", {})
            if business_hours:
                today_hours = business_hours.get("businessHours", [])
                if today_hours:
                    result["hours"] = ", ".join(today_hours)
    except Exception as e:
        result["error_summary"] = str(e)

    # 메뉴 정보
    menu_url = f"https://map.naver.com/v5/api/sites/{place_id}/menus?lang=ko"
    try:
        resp = requests.get(menu_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            menus = resp.json()
            result["menu"] = [
                {"name": m.get("name", ""), "price": m.get("price", "")}
                for m in menus[:10]
            ]
    except Exception:
        pass

    # 리뷰 키워드 (방문자 리뷰 태그)
    review_url = f"https://map.naver.com/v5/api/sites/{place_id}/reviews/highlights?lang=ko"
    try:
        resp = requests.get(review_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            highlights = data.get("result", {}).get("highlights", [])
            result["review_keywords"] = [h.get("text", "") for h in highlights[:10]]
    except Exception:
        pass

    # HTML 스크래핑으로 보완 (API 실패 시)
    if not result["name"]:
        try:
            page_url = f"https://map.naver.com/v5/entry/place/{place_id}"
            resp = requests.get(page_url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, "lxml")
            title_tag = soup.find("meta", property="og:title")
            if title_tag:
                result["name"] = title_tag.get("content", "")
        except Exception:
            pass

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 fetch_place_info.py <naver_map_url>")
        sys.exit(1)

    url = sys.argv[1]

    # 단축 URL 처리
    if "naver.me" in url or "me.naver" in url:
        print(f"단축 URL 해석 중: {url}", file=sys.stderr)
        url = resolve_short_url(url)
        print(f"실제 URL: {url}", file=sys.stderr)

    place_id = extract_place_id(url)
    if not place_id:
        print(f"Error: place ID를 찾을 수 없습니다. URL: {url}", file=sys.stderr)
        sys.exit(1)

    print(f"Place ID: {place_id}", file=sys.stderr)
    info = fetch_place_info(place_id)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
