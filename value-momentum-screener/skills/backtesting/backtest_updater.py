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


# ── 데이터 로딩 ───────────────────────────────────────────────

def load_scan_results(results_dir: Path) -> list[dict]:
    """
    results_dir의 *-raw.json 파일을 모두 읽어 스캔 기록 목록 반환.
    반환 형식: [{"scan_date": "YYYY-MM-DD", "picks": [...], "json_file": Path}]
    """
    scans = []
    for json_file in sorted(results_dir.glob("*-raw.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        scan_date = data["scan_date"].split(" ")[0]
        picks = extract_top_picks(data)
        scans.append({
            "scan_date": scan_date,
            "picks": picks,
            "json_file": json_file,
        })
    return scans


# ── 가격 조회 ─────────────────────────────────────────────────

def get_price_on_date(ticker: str, target_date: str) -> float | None:
    """
    target_date(YYYY-MM-DD) 당일 또는 이후 첫 거래일 종가 반환.
    미래 날짜이면 None 반환.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance 미설치. pip install yfinance", file=sys.stderr)
        return None

    today = datetime.today().date()
    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    if target > today:
        return None

    end = target + timedelta(days=5)
    hist = yf.Ticker(ticker).history(
        start=target.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    if hist.empty:
        return None
    return round(float(hist["Close"].iloc[0]), 2)
