#!/usr/bin/env python3
"""
Telegram notifier for value-momentum-screener results.
Usage: python telegram_notify.py [--date YYYY-MM-DD] [--results-dir PATH]
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "telegram_config.json"


# ── 파싱 ──────────────────────────────────────────────────────────────────────

def parse_summary_table(md_text: str) -> list[dict]:
    """
    '## Top 5 추천 종목' 아래 마크다운 테이블 파싱.
    반환: [{"rank", "ticker", "price", "total", "quant", "catalyst", "summary"}, ...]
    """
    table_pattern = re.compile(
        r'## Top 5 추천 종목.*?\n(?:.*?\n)*?\|[-| ]+\|\n((?:\|.*?\n)+)',
        re.DOTALL,
    )
    m = table_pattern.search(md_text)
    if not m:
        return []

    rows = []
    for line in m.group(1).strip().splitlines():
        cols = [c.strip() for c in line.strip('|').split('|')]
        if len(cols) < 7:
            continue
        ticker = re.sub(r'\s*⚠️.*', '', cols[1]).strip()
        rows.append({
            "rank": cols[0].strip(),
            "ticker": ticker,
            "price": cols[2].strip(),
            "total": cols[3].strip(),
            "quant": cols[4].strip(),
            "catalyst": cols[5].strip(),
            "summary": cols[6].strip(),
        })
    return rows


def parse_stock_sections(md_text: str) -> list[dict]:
    """
    '### N. TICKER — 회사명' 섹션들 파싱.
    반환: [{"rank", "ticker", "title", "total", "pe_line", "tech_line", "catalyst", "risk"}, ...]
    """
    section_pat = re.compile(
        r'### (\d+)\. (.+?)\n([\s\S]*?)(?=\n### \d+\.|\Z)'
    )
    stocks = []
    for m in section_pat.finditer(md_text):
        rank = m.group(1)
        title = m.group(2).strip()
        body = m.group(3)

        ticker_m = re.match(r'(\w+)', title)
        ticker = ticker_m.group(1) if ticker_m else title

        score_m = re.search(r'\*\*점수 내역\*\*([\s\S]*?)(?=\n\*\*|\n---|\Z)', body)
        score_text = score_m.group(1) if score_m else body

        def _extract(pattern):
            hit = re.search(pattern + r'(.+?)(?:\n|$)', score_text)
            return hit.group(1).strip() if hit else ""

        stocks.append({
            "rank": rank,
            "ticker": ticker,
            "title": title,
            "total": _extract(r'- 총점: '),
            "pe_line": _extract(r'- PE: '),
            "tech_line": _extract(r'- RSI: '),
            "catalyst": _extract(r'- 촉매: '),
            "risk": _extract(r'- ⚠️ 리스크: '),
        })
    return stocks
