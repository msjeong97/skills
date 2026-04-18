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

_ESCAPE_CHARS = '\\' + r'_*[]()~`>#+-=|{}.!$'


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
        f"{escape_md(detail['rank'])}\\. *{escape_md(detail['title'])}*",
        "",
        f"💰 현재가: {escape_md(summary['price'])}",
        f"📊 {escape_md(detail['total'])}",
        "",
    ]
    if detail["pe_line"]:
        lines.append(f"📈 {escape_md(detail['pe_line'])}")
    if detail["tech_line"]:
        lines.append(f"📉 {escape_md(detail['tech_line'])}")
    if detail["catalyst"]:
        lines += ["", f"💬 {escape_md(detail['catalyst'])}"]
    if detail["risk"]:
        lines += ["", f"⚠️ {escape_md(detail['risk'])}"]
    return "\n".join(lines)


# ── 전송 ──────────────────────────────────────────────────────────────────────

def load_config() -> dict | None:
    """telegram_config.json 로딩. 없으면 None 반환."""
    if not CONFIG_PATH.exists():
        print(f"[telegram] 설정 파일 없음: {CONFIG_PATH}", file=sys.stderr)
        return None
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def send_message(bot_token: str, chat_id: str, text: str) -> bool:
    """Telegram Bot API로 메시지 전송. 성공 시 True."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    resp = requests.post(
        url,
        json={"chat_id": chat_id, "text": text, "parse_mode": "MarkdownV2"},
        timeout=10,
    )
    if not resp.ok:
        print(f"[telegram] 전송 실패: {resp.status_code} {resp.text}", file=sys.stderr)
        return False
    return True


def run(results_dir: Path, date_str: str) -> int:
    """전체 파이프라인: config 로딩 → MD 파싱 → 메시지 전송."""
    config = load_config()
    if config is None:
        return 0

    md_path = results_dir / f"{date_str}.md"
    if not md_path.exists():
        print(f"[telegram] MD 파일 없음: {md_path}", file=sys.stderr)
        return 0

    md_text = md_path.read_text(encoding="utf-8")
    summary_rows = parse_summary_table(md_text)
    stock_details = parse_stock_sections(md_text)

    if not summary_rows:
        print("[telegram] 요약 테이블 파싱 실패", file=sys.stderr)
        return 0

    bot_token = config["bot_token"]
    chat_id = config["chat_id"]

    send_message(bot_token, chat_id, format_header_message(date_str, summary_rows))
    time.sleep(0.5)

    detail_by_rank = {d["rank"]: d for d in stock_details}
    for row in summary_rows:
        detail = detail_by_rank.get(row["rank"])
        if detail is None:
            continue
        send_message(bot_token, chat_id, format_stock_message(row, detail))
        time.sleep(0.5)

    print(f"[telegram] 전송 완료: {len(summary_rows) + 1}개 메시지")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Telegram notifier for screener results")
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="결과 날짜 (기본: 오늘, 형식: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="results 디렉토리 경로",
    )
    args = parser.parse_args()
    sys.exit(run(args.results_dir, args.date))


if __name__ == "__main__":
    main()
