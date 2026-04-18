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
    ticker: str,
    base_price: float,
    prices: dict,
) -> bool:
    """
    MD 파일에서 ticker 섹션의 빈 가격 추적 행을 채운다.
    prices: {"1w": float|None, "2w": float|None, "4w": float|None}
    이미 채워진 행(비어있지 않은 행)은 건드리지 않는다.
    반환: 실제 변경이 발생했으면 True.
    """
    content = md_path.read_text(encoding="utf-8")
    base_str = f"${base_price:.2f}"

    # ticker 섹션 범위만 추출 (다음 ### 섹션 또는 문서 끝까지)
    section_pattern = re.compile(
        rf"(### \d+\. {re.escape(ticker)}[\s\S]*?)(?=\n### \d+\.|\Z)",
    )
    section_match = section_pattern.search(content)
    if not section_match:
        return False

    section_text = section_match.group(1)
    section_start = section_match.start(1)
    section_end = section_match.end(1)

    # 빈 행 패턴: | $XXX.XX | | | |
    empty_pattern = re.compile(
        r"\|\s*" + re.escape(base_str) + r"\s*\|(\s*\|\s*){3}"
    )
    if not empty_pattern.search(section_text):
        return False  # 행이 없거나 이미 채워짐

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
    new_section = empty_pattern.sub(new_row, section_text, count=1)
    if new_section == section_text:
        return False

    new_content = content[:section_start] + new_section + content[section_end:]
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


# ── 촉매 점수 파싱 ────────────────────────────────────────────

def parse_catalyst_score_from_md(md_path: Path, ticker: str) -> int | None:
    """MD 파일에서 특정 ticker의 촉매 점수 파싱."""
    if not md_path.exists():
        return None
    content = md_path.read_text(encoding="utf-8")
    # ticker 섹션 찾기
    ticker_section = re.search(
        rf"### \d+\. {re.escape(ticker)}.*?(?=### \d+\.|$)",
        content,
        re.DOTALL,
    )
    if not ticker_section:
        return None
    return parse_catalyst_score(ticker_section.group())


# ── 메인 파이프라인 ───────────────────────────────────────────

def run(results_dir: Path) -> dict:
    """
    전체 파이프라인 실행.
    1. 스캔 결과 로딩
    2. 각 픽에 대해 +7/+14/+28일 가격 조회
    3. MD 파일 업데이트
    4. 분석용 데이터 반환
    """
    scans = load_scan_results(results_dir)
    analysis_picks = []
    updated_count = 0

    for scan in scans:
        scan_date = scan["scan_date"]
        md_path = results_dir / f"{scan_date}.md"

        date_1w = add_calendar_days(scan_date, 7)
        date_2w = add_calendar_days(scan_date, 14)
        date_4w = add_calendar_days(scan_date, 28)

        for pick in scan["picks"]:
            ticker = pick["ticker"]
            base_price = pick["current_price"]

            price_1w = get_price_on_date(ticker, date_1w)
            price_2w = get_price_on_date(ticker, date_2w)
            price_4w = get_price_on_date(ticker, date_4w)

            prices = {"1w": price_1w, "2w": price_2w, "4w": price_4w}

            # MD 업데이트
            if md_path.exists():
                if update_md_price_table(md_path, ticker, base_price, prices):
                    updated_count += 1

            # 수익률 계산
            def ret(p):
                if p is None or base_price == 0:
                    return None
                return round((p - base_price) / base_price * 100, 2)

            # 팩터 추출
            detail = pick.get("detail", {})
            tech = detail.get("technical", {})
            val = detail.get("valuation", {})
            prof = detail.get("profitability", {})
            breakdown = pick.get("breakdown", {})

            catalyst_score = parse_catalyst_score_from_md(md_path, ticker)

            analysis_picks.append({
                "scan_date": scan_date,
                "ticker": ticker,
                "base_price": base_price,
                "returns": {
                    "1w": ret(price_1w),
                    "2w": ret(price_2w),
                    "4w": ret(price_4w),
                },
                "factors": {
                    "52w_pct_from_low": tech.get("52w_low", {}).get("pct_from_low"),
                    "rsi": tech.get("rsi", {}).get("value"),
                    "macd_golden_cross": tech.get("macd", {}).get("golden_cross"),
                    "bollinger_pct_b": tech.get("bollinger", {}).get("pct_b"),
                    "fcf_yield": val.get("fcf_yield", {}).get("value"),
                    "peg": val.get("peg", {}).get("value"),
                    "quant_score": pick.get("undervalue_score"),
                    "valuation_score": breakdown.get("valuation"),
                    "profitability_score": breakdown.get("profitability"),
                    "technical_score": breakdown.get("technical"),
                    "catalyst_score": catalyst_score,
                },
            })

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_scans": len(scans),
        "total_picks": len(analysis_picks),
        "md_updated_count": updated_count,
        "picks": analysis_picks,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest price updater")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "results",
        help="results 디렉토리 경로",
    )
    parser.add_argument(
        "--backtesting-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "results-backtesting",
        help="백테스팅 결과 저장 디렉토리 경로",
    )
    args = parser.parse_args()

    args.backtesting_dir.mkdir(exist_ok=True)

    print(f"결과 디렉토리: {args.results_dir}", file=sys.stderr)
    data = run(args.results_dir)

    print(f"스캔 수: {data['total_scans']}, 픽 수: {data['total_picks']}, MD 업데이트: {data['md_updated_count']}건", file=sys.stderr)

    # 분석 JSON을 results-backtesting에 저장
    date_str = datetime.now().strftime("%Y-%m-%d")
    json_path = args.backtesting_dir / f"analysis-{date_str}.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"분석 JSON 저장됨: {json_path}", file=sys.stderr)

    # stdout으로도 출력 (SKILL.md에서 읽음)
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
