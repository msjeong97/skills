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
        r'## Top 5 추천 종목[^\n]*\n(?:[^\n]*\n)*?\|[-| ]+\|\n((?:\|.*?\n)+)',
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

        def _extract(pattern, _st=score_text):
            hit = re.search(pattern + r'([^\r\n]+)', _st)
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


# ── 포맷 ──────────────────────────────────────────────────────────────────────

_ESCAPE_CHARS = r'_*[]()~`>#+-=|{}.!$'


def escape_md(text: str) -> str:
    """Telegram MarkdownV2 특수문자 이스케이프."""
    for ch in _ESCAPE_CHARS:
        text = text.replace(ch, f'\\{ch}')
    return text


def format_header_message(date_str: str, summary_rows: list[dict]) -> str:
    """헤더 요약 메시지 생성."""
    lines = [
        "📊 *Value Momentum Screener*",
        f"{escape_md(date_str)} \\| 유니버스 150종목",
        "",
        "🏆 *Top 5*",
    ]
    for row in summary_rows:
        lines.append(
            f"{escape_md(row['rank'])}\\. *{escape_md(row['ticker'])}* "
            f"— {escape_md(row['total'])} "
            f"\\[계량 {escape_md(row['quant'])} \\+ 촉매 {escape_md(row['catalyst'])}\\]"
        )
        lines.append(f"  💬 {escape_md(row['summary'])}")
    lines += ["", "⚠️ 투자 조언 아님\\."]
    return "\n".join(lines)


def format_stock_message(summary: dict, detail: dict) -> str:
    """종목별 상세 메시지 생성."""
    lines = [
        f"*{escape_md(detail['rank'])}\\. {escape_md(detail['title'])}*",
        "",
        f"💰 현재가: {escape_md(summary['price'])}",
        f"📊 {escape_md(detail['total'])}",
        "",
    ]
    if detail["pe_line"]:
        lines.append(escape_md(detail["pe_line"]))
    if detail["tech_line"]:
        lines.append(escape_md(detail["tech_line"]))
    if detail["catalyst"]:
        lines += ["", f"💬 {escape_md(detail['catalyst'])}"]
    if detail["risk"]:
        lines += ["", f"⚠️ {escape_md(detail['risk'])}"]
    return "\n".join(lines)
