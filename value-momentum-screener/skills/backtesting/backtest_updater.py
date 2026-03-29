#!/usr/bin/env python3
"""
Backtesting price updater for value-momentum-screener.
Usage: python backtest_updater.py [--results-dir PATH]
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path


# ── 헬퍼 함수 ────────────────────────────────────────────────

def add_calendar_days(date_str: str, days: int) -> str:
    """'YYYY-MM-DD' 문자열에 days를 더한 날짜 문자열 반환."""
    d = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)
    return d.strftime("%Y-%m-%d")


def extract_top_picks(data: dict) -> list:
    """JSON 데이터에서 top10 또는 top20 키로 픽 목록 반환."""
    return data.get("top10") or data.get("top20") or []


def parse_catalyst_score(text: str) -> int | None:
    """'촉매 15/30' 패턴에서 촉매 점수 추출."""
    m = re.search(r"촉매\s+(\d+)/30", text)
    return int(m.group(1)) if m else None


def update_md_price_table(
    md_path: Path,
    base_price: float,
    prices: dict,
) -> bool:
    """
    MD 파일의 빈 가격 추적 행을 채운다.
    prices: {"1w": float|None, "2w": float|None, "4w": float|None}
    이미 채워진 행은 건드리지 않는다.
    반환: 실제 변경이 발생했으면 True.
    """
    content = md_path.read_text(encoding="utf-8")
    base_str = f"${base_price:.2f}"

    # 이미 채워진 행 감지: 기준가 뒤에 숫자가 있으면 skip
    filled_pattern = re.compile(
        r"\|\s*" + re.escape(base_str) + r"\s*\|\s*\$[\d.]"
    )
    if filled_pattern.search(content):
        return False

    # 빈 행 패턴: | $XXX.XX | | | |  (공백 허용)
    empty_pattern = re.compile(
        r"(\|\s*" + re.escape(base_str) + r"\s*\|)(\s*\|\s*){3}"
    )
    if not empty_pattern.search(content):
        return False

    def cell(price, base):
        if price is None:
            return " "
        pct = (price - base) / base * 100
        sign = "+" if pct >= 0 else ""
        return f" ${price:.2f} ({sign}{pct:.1f}%) "

    new_row = (
        f"| {base_str} |"
        + cell(prices.get("1w"), base_price) + "|"
        + cell(prices.get("2w"), base_price) + "|"
        + cell(prices.get("4w"), base_price) + "|"
    )
    new_content = empty_pattern.sub(new_row, content)
    if new_content == content:
        return False
    md_path.write_text(new_content, encoding="utf-8")
    return True
