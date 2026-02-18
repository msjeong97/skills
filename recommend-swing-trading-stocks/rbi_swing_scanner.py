#!/usr/bin/env python3
"""
RBI-Scan v1.0: 30일 단기 급등주 포착 시스템
Moon Dev's R-B-I (Research-Backtest-Implement) System

Usage:
    python rbi_swing_scanner.py                      # 기본 종목 리스트 스캔
    python rbi_swing_scanner.py AAPL MSFT NVDA       # 특정 종목 스캔
    python rbi_swing_scanner.py 005930.KS 000660.KS  # KOSPI 종목 스캔
"""

import warnings

warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    import pandas_ta as ta

    def compute_rsi(series, length=14):
        return ta.rsi(series, length=length)
except ImportError:
    ta = None

    def compute_rsi(series, length=14):
        """RSI 수동 계산 (pandas_ta 미설치 시 폴백)."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(length).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

try:
    from investgo import get_pair_id, get_technical_data

    _HAS_INVESTGO = True
except ImportError:
    _HAS_INVESTGO = False


# ============================================================
# Configuration
# ============================================================
TP_PCT = 0.10  # 익절 +10%
SL_PCT = 0.05  # 손절 -5%
VOLUME_MULT = 2.0  # 거래량 배수 기준
RSI_LOW = 40
RSI_HIGH = 60
RESISTANCE_RANGE = 0.05  # 매물대 체크 범위 (+5%)
GC_LOOKBACK = 5  # 골든크로스 탐색 기간 (일)
DOWNLOAD_WORKERS = 8
INVESTING_URL = "https://kr.investing.com/search/?q={ticker}"
TECH_SIGNAL_KR = {
    "Strong Buy": "적극 매수",
    "Buy": "매수",
    "Neutral": "중립",
    "Sell": "매도",
    "Strong Sell": "적극 매도",
}
TECH_SUMMARY_WORKERS = 4

# ============================================================
# Built-in Ticker Lists
# ============================================================
NASDAQ_MAJOR = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "TSLA", "AVGO",
    "COST", "NFLX", "AMD", "ADBE", "PEP", "CSCO", "INTC", "TMUS",
    "CMCSA", "TXN", "AMGN", "QCOM", "INTU", "AMAT", "ISRG", "HON",
    "BKNG", "LRCX", "VRTX", "MDLZ", "ADI", "REGN", "SBUX", "ADP",
    "GILD", "PANW", "MU", "SNPS", "KLAC", "CDNS", "MELI", "PYPL",
    "CRWD", "ABNB", "WDAY", "MNST", "FTNT", "DASH", "MRVL", "PCAR",
    "ROST", "CPRT", "NXPI", "FAST", "LULU", "EA", "FANG", "ON",
    "DDOG", "TTD", "TEAM", "ARM", "SMCI", "COIN",
]

SP500_MAJOR = [
    "JPM", "V", "JNJ", "WMT", "PG", "UNH", "MA", "HD", "DIS",
    "BAC", "XOM", "CVX", "KO", "MRK", "ABBV", "PFE", "LLY",
    "CRM", "ACN", "NKE", "MCD", "TMO", "ORCL", "ABT", "DHR",
    "NEE", "RTX", "LOW", "GS", "CAT", "BA", "DE", "BLK",
    "SPGI", "SCHW", "AXP", "ANET", "NOW", "UBER",
]

DEFAULT_TICKERS = NASDAQ_MAJOR + SP500_MAJOR


# ============================================================
# Data Download
# ============================================================
def download_data(ticker, period="1y"):
    """단일 종목 OHLCV 데이터 다운로드 (thread-safe)."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period)
        if df.empty or len(df) < 60:
            return None
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                return None
        return df[required].copy()
    except Exception:
        return None


def get_company_name(ticker):
    """종목명 조회."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName", info.get("longName", ticker))
    except Exception:
        return ticker


def _stock_url(ticker):
    """종목의 kr.investing.com 검색 URL을 반환한다."""
    return INVESTING_URL.format(ticker=ticker)


def _strip_ticker_suffix(ticker):
    """yfinance 접미사(.KS, .KQ 등)를 제거한 순수 티커를 반환한다."""
    for suffix in (".KS", ".KQ", ".L", ".T", ".HK"):
        if ticker.endswith(suffix):
            return ticker[: -len(suffix)]
    return ticker


def get_tech_summary(ticker):
    """단일 종목의 Investing.com 기술적 분석 요약을 조회한다."""
    if not _HAS_INVESTGO:
        return "-"
    try:
        search_key = _strip_ticker_suffix(ticker)
        pair_ids = get_pair_id(search_key)
        if not pair_ids:
            return "-"
        df = get_technical_data(pair_ids[0], tech_type="summary", interval="daily")
        overall = df[df["type"] == "Overall"]
        if overall.empty:
            return "-"
        signal = overall.iloc[0]["signal"]
        return TECH_SIGNAL_KR.get(signal, signal)
    except Exception:
        return "-"


def fetch_tech_summaries(stocks):
    """종목 리스트에 대해 기술적 분석 요약을 병렬 조회하여 추가한다."""
    if not stocks:
        return
    if not _HAS_INVESTGO:
        for s in stocks:
            s["tech_summary"] = "-"
        return

    print(f"\n  Fetching technical summaries from kr.investing.com...")

    def _fetch_one(stock):
        stock["tech_summary"] = get_tech_summary(stock["ticker"])
        return stock

    with ThreadPoolExecutor(max_workers=TECH_SUMMARY_WORKERS) as executor:
        list(executor.map(_fetch_one, stocks))


# ============================================================
# Phase 1: Research - RBI Condition Checker
# ============================================================
def analyze_stock(ticker):
    """단일 종목에 대해 5가지 RBI 조건을 검사한다.

    Returns:
        dict with analysis results, or None if data insufficient.
    """
    df = download_data(ticker)
    if df is None:
        return None

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["RSI14"] = compute_rsi(df["Close"], length=14)
    df["AvgVol5"] = df["Volume"].rolling(5).mean()

    # 최소 데이터 확인 (NaN check)
    if pd.isna(df["MA20"].iloc[-1]) or pd.isna(df["MA5"].iloc[-1]):
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    conditions = {}

    # 1) 추세: Close > MA20 이고 MA20 우상향
    conditions["trend"] = bool(
        latest["Close"] > latest["MA20"] and latest["MA20"] > prev["MA20"]
    )

    # 2) 모멘텀: MA5가 MA20을 상향 돌파 (최근 N일 이내)
    gc_found = False
    gc_days_ago = 0
    for offset in range(0, GC_LOOKBACK):
        idx = len(df) - 1 - offset
        if idx < 1:
            break
        cur = df.iloc[idx]
        prv = df.iloc[idx - 1]
        if (
            not np.isnan(cur["MA5"])
            and not np.isnan(cur["MA20"])
            and not np.isnan(prv["MA5"])
            and not np.isnan(prv["MA20"])
        ):
            if cur["MA5"] > cur["MA20"] and prv["MA5"] <= prv["MA20"]:
                gc_found = True
                gc_days_ago = offset
                break
    conditions["golden_cross"] = gc_found
    conditions["gc_days_ago"] = gc_days_ago

    # 3) 강도: RSI(14) 40~60
    rsi_val = latest["RSI14"]
    conditions["rsi_ok"] = bool(
        not np.isnan(rsi_val) and RSI_LOW <= rsi_val <= RSI_HIGH
    )

    # 4) 거래량: 5일 평균 대비 2배 이상
    avg_vol = latest["AvgVol5"]
    vol_ratio = float(latest["Volume"] / avg_vol) if avg_vol > 0 else 0.0
    conditions["volume_ok"] = bool(vol_ratio >= VOLUME_MULT)

    # 5) 매물대: 현재가 +5% 이내에 3개월 최고점(저항) 없음
    three_month_high = float(df["High"].tail(63).max())
    current_price = float(latest["Close"])
    has_resistance = current_price < three_month_high <= current_price * (
        1 + RESISTANCE_RANGE
    )
    conditions["no_resistance"] = not has_resistance

    score = sum(
        [
            conditions["trend"],
            conditions["golden_cross"],
            conditions["rsi_ok"],
            conditions["volume_ok"],
            conditions["no_resistance"],
        ]
    )

    return {
        "ticker": ticker,
        "close": current_price,
        "ma5": float(latest["MA5"]),
        "ma20": float(latest["MA20"]),
        "rsi": float(rsi_val) if not np.isnan(rsi_val) else 0.0,
        "vol_ratio": round(vol_ratio, 2),
        "three_month_high": three_month_high,
        "conditions": conditions,
        "score": score,
        "passed": score == 5,
        "df": df,
    }


def find_rbi_stocks(tickers):
    """전체 종목 리스트에서 RBI 조건 통과 종목을 찾는다."""
    results = []
    near_misses = []
    analyzed = 0
    skipped = 0
    completed = 0
    total = len(tickers)

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(analyze_stock, t): t for t in tickers}
        for future in as_completed(futures):
            completed += 1
            pct = completed / total * 100
            bar_len = 30
            filled = int(bar_len * completed / total)
            bar = "=" * filled + "-" * (bar_len - filled)
            print(
                f"\r  [{bar}] {completed}/{total} ({pct:.0f}%)",
                end="",
                flush=True,
            )

            try:
                result = future.result()
            except Exception:
                skipped += 1
                continue

            if result is None:
                skipped += 1
                continue

            analyzed += 1
            if result["passed"]:
                results.append(result)
            elif result["score"] >= 3:
                result["trade_prices"] = calc_trade_prices(result, result["df"])
                del result["df"]
                near_misses.append(result)
            else:
                del result["df"]

    print(
        f"\r  Scan done: {analyzed} analyzed, {skipped} skipped, "
        f"{len(results)} passed, {len(near_misses)} near-miss" + " " * 10
    )
    return results, near_misses


# ============================================================
# Phase 2: Backtest
# ============================================================

# backtesting.py 라이브러리 사용 시도
_USE_BT_LIB = False
try:
    from backtesting import Backtest, Strategy

    class RBIStrategy(Strategy):
        """backtesting.py용 RBI 전략 클래스.

        백테스트는 핵심 3조건(추세+골든크로스+RSI)으로 실행하여
        충분한 거래 표본을 확보한다. (거래량/매물대는 실시간 확인용)
        """

        def init(self):
            close = self.data.Close
            self.ma5 = self.I(
                lambda x: pd.Series(x).rolling(5).mean(), close, name="MA5"
            )
            self.ma20 = self.I(
                lambda x: pd.Series(x).rolling(20).mean(), close, name="MA20"
            )
            self.rsi = self.I(
                lambda x: compute_rsi(pd.Series(x), length=14), close, name="RSI"
            )

        def next(self):
            if self.position:
                return

            price = self.data.Close[-1]

            try:
                trend = price > self.ma20[-1] and self.ma20[-1] > self.ma20[-2]
                golden = self.ma5[-1] > self.ma20[-1] and self.ma5[-2] <= self.ma20[-2]
                rsi_ok = RSI_LOW <= self.rsi[-1] <= RSI_HIGH
            except (IndexError, ValueError):
                return

            if any(np.isnan(x) for x in [self.ma20[-1], self.ma5[-1], self.rsi[-1]]):
                return

            if trend and golden and rsi_ok:
                self.buy(
                    tp=price * (1 + TP_PCT),
                    sl=price * (1 - SL_PCT),
                )

    _USE_BT_LIB = True
except ImportError:
    pass


def run_backtest_library(df):
    """backtesting.py를 사용한 백테스트."""
    bt_df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    if len(bt_df) < 60:
        return None

    try:
        bt = Backtest(
            bt_df,
            RBIStrategy,
            cash=10_000_000,
            commission=0.001,
            exclusive_orders=True,
        )
        stats = bt.run()

        n_trades = int(stats["# Trades"])
        if n_trades == 0:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "expected_value": 0.0,
            }

        win_rate = float(stats.get("Win Rate [%]", 0))
        wins = int(round(n_trades * win_rate / 100))
        ev = (win_rate / 100 * TP_PCT * 100) + ((1 - win_rate / 100) * (-SL_PCT * 100))

        return {
            "total_trades": n_trades,
            "wins": wins,
            "losses": n_trades - wins,
            "win_rate": round(win_rate, 1),
            "avg_return": round(float(stats.get("Avg. Trade [%]", ev)), 2),
            "expected_value": round(ev, 2),
        }
    except Exception:
        return None


def run_backtest_manual(df):
    """수동 백테스트 시뮬레이션 (폴백).

    핵심 3조건(추세+골든크로스+RSI)으로 실행하여 충분한 거래 표본 확보.
    """
    df = df.copy()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["RSI14"] = compute_rsi(df["Close"], length=14)

    trades = []
    in_position = False
    entry_price = 0.0

    for i in range(21, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if any(np.isnan(v) for v in [row["MA5"], row["MA20"], row["RSI14"]]):
            continue

        if not in_position:
            trend = row["Close"] > row["MA20"] and row["MA20"] > prev["MA20"]
            golden = row["MA5"] > row["MA20"] and prev["MA5"] <= prev["MA20"]
            rsi_ok = RSI_LOW <= row["RSI14"] <= RSI_HIGH

            if trend and golden and rsi_ok:
                entry_price = row["Close"]
                in_position = True
        else:
            # TP/SL 체크 (장중 고가/저가 기준)
            if row["High"] >= entry_price * (1 + TP_PCT):
                trades.append(TP_PCT * 100)
                in_position = False
            elif row["Low"] <= entry_price * (1 - SL_PCT):
                trades.append(-SL_PCT * 100)
                in_position = False

    if not trades:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "expected_value": 0.0,
        }

    wins = sum(1 for t in trades if t > 0)
    total = len(trades)
    win_rate = wins / total * 100
    avg_ret = sum(trades) / total
    ev = (win_rate / 100 * TP_PCT * 100) + ((1 - win_rate / 100) * (-SL_PCT * 100))

    return {
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(win_rate, 1),
        "avg_return": round(avg_ret, 2),
        "expected_value": round(ev, 2),
    }


def run_backtest(df):
    """백테스트 실행 (라이브러리 우선, 실패 시 수동 폴백)."""
    if _USE_BT_LIB:
        result = run_backtest_library(df)
        if result is not None:
            return result
    return run_backtest_manual(df)


# ============================================================
# Dynamic Trade Price Calculator
# ============================================================
def calc_trade_prices(stock, df):
    """기술적 분석 기반 동적 매매가격을 산출한다.

    Returns:
        dict: ideal_entry, max_entry, entry_reason, stop_loss, sl_pct,
              sl_reason, target, tp_pct, tp_reason, rr_ratio, atr, swing_low_10d
    """
    close = float(df["Close"].iloc[-1])
    ma5 = float(df["MA5"].iloc[-1]) if "MA5" in df.columns else float(df["Close"].rolling(5).mean().iloc[-1])
    ma20 = float(df["MA20"].iloc[-1]) if "MA20" in df.columns else float(df["Close"].rolling(20).mean().iloc[-1])

    # ATR(14)
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])

    # ---- 1. 구매추천가 (Ideal Entry Price) ----
    if ma5 < close * 0.99:
        ideal_entry = round(ma5, 2)
        entry_reason = "MA5 눌림목 매수 (지정가)"
    else:
        ideal_entry = round(close, 2)
        entry_reason = "MA5 근접, 현재가 진입 OK"
    max_entry = round(close * 1.01, 2)

    # ---- 2. 손절가 (Stop Loss) ----
    # 10일 스윙로우
    swing_low_10d = float(df["Low"].tail(10).min())
    sl_swing = swing_low_10d * 0.995

    # MA20 이탈
    sl_ma20 = ma20 * 0.99

    # ATR 기반
    sl_atr = ideal_entry - 2 * atr

    # 가장 보수적(높은 값) 채택
    sl_candidates = [
        (sl_swing, f"10일 스윙로우(${swing_low_10d:.2f}) 하단 이탈 기준"),
        (sl_ma20, f"MA20(${ma20:.2f}) -1% 이탈 기준"),
        (sl_atr, f"ATR(${atr:.2f}) x2 변동성 기반"),
    ]
    sl_candidates.sort(key=lambda x: x[0], reverse=True)
    raw_sl = sl_candidates[0][0]
    sl_reason = sl_candidates[0][1]

    # 클램핑: 진입가 대비 -3% ~ -8%
    sl_floor = ideal_entry * 0.92  # -8%
    sl_ceil = ideal_entry * 0.97   # -3%
    if raw_sl > sl_ceil:
        stop_loss = round(sl_ceil, 2)
        sl_reason += " -> 최소 -3% 적용"
    elif raw_sl < sl_floor:
        stop_loss = round(sl_floor, 2)
        sl_reason += " -> 최대 -8% 제한"
    else:
        stop_loss = round(raw_sl, 2)

    sl_pct = round((stop_loss / ideal_entry - 1) * 100, 1)

    # ---- 3. 목표매도가 (Target / Take Profit) ----
    # 3개월 고점
    three_month_high = float(df["High"].tail(63).max())
    if three_month_high <= close:
        # 이미 돌파 -> 6개월 고점 또는 +15%
        six_month_high = float(df["High"].tail(126).max())
        if six_month_high > close:
            tp_resist = six_month_high
        else:
            tp_resist = ideal_entry * 1.15
    else:
        tp_resist = three_month_high

    # ATR 기반
    tp_atr = ideal_entry + 3 * atr

    # 피보나치 확장: 20일 스윙저점 + (스윙고-스윙저) * 1.618
    swing_low_20d = float(df["Low"].tail(20).min())
    swing_high_20d = float(df["High"].tail(20).max())
    tp_fib = swing_low_20d + (swing_high_20d - swing_low_20d) * 1.618

    tp_candidates = [
        (tp_resist, "저항선 기반"),
        (tp_atr, "ATR x3 변동성 목표"),
        (tp_fib, "피보나치 1.618 확장"),
    ]

    # +8% 이상인 후보 중 가장 낮은 값
    min_target_pct = 1.08
    valid_targets = [(p, r) for p, r in tp_candidates if p >= ideal_entry * min_target_pct]

    if valid_targets:
        valid_targets.sort(key=lambda x: x[0])
        target = round(valid_targets[0][0], 2)
        tp_reason = valid_targets[0][1]
    else:
        # 후보 모두 +8% 미만 -> 기본 +10%
        target = round(ideal_entry * 1.10, 2)
        tp_reason = "기본 +10% 적용 (R:R 보장)"

    tp_pct = round((target / ideal_entry - 1) * 100, 1)

    # ---- R:R Ratio ----
    risk = abs(ideal_entry - stop_loss)
    reward = abs(target - ideal_entry)
    rr_ratio = round(reward / risk, 1) if risk > 0 else 0.0

    return {
        "ideal_entry": ideal_entry,
        "max_entry": max_entry,
        "entry_reason": entry_reason,
        "stop_loss": stop_loss,
        "sl_pct": sl_pct,
        "sl_reason": sl_reason,
        "target": target,
        "tp_pct": tp_pct,
        "tp_reason": tp_reason,
        "rr_ratio": rr_ratio,
        "atr": round(atr, 2),
        "swing_low_10d": round(swing_low_10d, 2),
    }


# ============================================================
# Phase 3: Output Formatter
# ============================================================
def print_header():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    engine = "backtesting.py" if _USE_BT_LIB else "Built-in Simulator"
    print()
    print("=" * 70)
    print("  RBI-Scan v1.0 : 30-Day Swing Trading Scanner")
    print("  Moon Dev's R-B-I (Research-Backtest-Implement)")
    print(f"  Scan Time  : {now}")
    print(f"  Backtest   : {engine}")
    print(f"  TP / SL    : +{TP_PCT*100:.0f}% / -{SL_PCT*100:.0f}%")
    print("=" * 70)


def print_results(results, near_misses):
    """최종 결과를 포맷에 맞게 출력한다."""

    total_passed = len(results)
    total_near = len(near_misses)

    print()
    print("=" * 70)
    print(f"  SCAN RESULT : {total_passed} stocks passed | {total_near} near-miss (3+/5)")
    print("=" * 70)

    # ---- 통과 종목 없음 ----
    if not results:
        print()
        print("  [!] All 5 RBI conditions are not met for any stock.")
        print("      (This is normal - strict filters catch only early-stage breakouts)")

        if near_misses:
            # P1(거래량)+P2(매물대) 통과 종목 우선 정렬
            near_misses.sort(
                key=lambda x: (
                    x["conditions"]["volume_ok"],
                    x["conditions"]["no_resistance"],
                    -x["score"],
                ),
                reverse=True,
            )
            print()
            print(f"  [REF] Near-miss stocks ({len(near_misses)} found):")
            print(
                f"  {'Ticker':>8s}  {'Price($)':>10s}  {'Entry($)':>10s}  "
                f"{'SL($)':>9s}  {'Target($)':>10s}  {'R:R':>6s}  "
                f"{'RSI':>6s}  {'Vol':>6s}  {'기술분석':>10s}  Failed"
            )
            print(
                f"  {'------':>8s}  {'--------':>10s}  {'--------':>10s}  "
                f"{'-----':>9s}  {'---------':>10s}  {'---':>6s}  "
                f"{'---':>6s}  {'---':>6s}  {'------':>10s}  ------"
            )
            for nm in near_misses[:10]:
                failed = _get_failed_conditions_priority(nm)
                url = _stock_url(nm["ticker"])
                ts = nm.get("tech_summary", "-")
                tp = nm.get("trade_prices", {})
                entry = tp.get("ideal_entry", nm["close"])
                sl = tp.get("stop_loss", nm["close"] * 0.95)
                tgt = tp.get("target", nm["close"] * 1.10)
                rr = tp.get("rr_ratio", 0)
                print(
                    f"  {nm['ticker']:>8s}  ${nm['close']:>9.2f}  ${entry:>9.2f}  "
                    f"${sl:>8.2f}  ${tgt:>9.2f}  {f'1:{rr:.1f}':>6s}  "
                    f"{nm['rsi']:>5.1f}  {nm['vol_ratio']:>5.1f}x  "
                    f"{ts:>10s}  {failed}"
                )
                print(f"  {'':8s}  {'':10s}  {'':10s}  {'':9s}  {'':10s}  {'':6s}  {'':6s}  {'':6s}  {'':10s}  {url}")

        # ---- Priority Legend ----
        print()
        print("  [PRIORITY GUIDE / 조건 우선순위 참고]")
        print("    P1 Volume     (★★★★★) 엔진의 시동. 거래량은 속일 수 없다.")
        print("    P2 Resistance (★★★★☆) 목적지까지의 장애물. 매물대 없어야 가볍게 상승.")
        print("    P3 Trend/MA   (★★★☆☆) 진입 타이밍. 과거 데이터 기반, 미래 보장 아님.")
        print("    P4 RSI        (★★☆☆☆) 보조 지표. 강한 종목은 과열권에서도 상승 지속.")
        print()
        return

    # ---- 추천 종목 테이블 ----
    results.sort(key=lambda x: x.get("backtest", {}).get("win_rate", 0), reverse=True)

    print()
    print("-" * 70)
    print("  [Phase 1] Research : RBI Filter Results")
    print("-" * 70)
    print(
        f"  {'#':>2}  {'Ticker':>8s}  {'Price($)':>10s}  {'Entry($)':>10s}  "
        f"{'SL($)':>9s}  {'Target($)':>10s}  {'R:R':>6s}  "
        f"{'WinRate':>8s}  {'E[R]':>7s}  {'기술분석':>10s}  Link"
    )
    print(
        f"  {'--':>2}  {'------':>8s}  {'--------':>10s}  {'--------':>10s}  "
        f"{'-----':>9s}  {'---------':>10s}  {'---':>6s}  "
        f"{'-------':>8s}  {'----':>7s}  {'------':>10s}  ----"
    )

    for i, r in enumerate(results, 1):
        bt = r.get("backtest", {})
        tp = r.get("trade_prices", {})
        wr = bt.get("win_rate", 0)
        ev = bt.get("expected_value", 0)
        ts = r.get("tech_summary", "-")
        url = _stock_url(r["ticker"])
        entry = tp.get("ideal_entry", r["close"])
        sl = tp.get("stop_loss", r["close"] * 0.95)
        tgt = tp.get("target", r["close"] * 1.10)
        rr = tp.get("rr_ratio", 0)
        print(
            f"  {i:>2}  {r['ticker']:>8s}  ${r['close']:>9.2f}  ${entry:>9.2f}  "
            f"${sl:>8.2f}  ${tgt:>9.2f}  {f'1:{rr:.1f}':>6s}  "
            f"{wr:>7.1f}%  {ev:>+6.2f}%  {ts:>10s}  {url}"
        )

    # ---- 종목별 상세 분석 ----
    print()
    print("=" * 70)
    print("  [Phase 2] Backtest : Detailed Analysis")
    print("=" * 70)

    for i, r in enumerate(results, 1):
        bt = r.get("backtest", {})
        tp = r.get("trade_prices", {})
        name = r.get("name", r["ticker"])

        url = _stock_url(r["ticker"])
        print()
        print("-" * 70)
        print(f"  {i}. {r['ticker']} ({name})")
        print(f"     {url}")
        print("-" * 70)
        print(f"  Price: ${r['close']:.2f}  |  MA5: ${r['ma5']:.2f}  |  MA20: ${r['ma20']:.2f}")
        ts = r.get("tech_summary", "-")
        if ts != "-":
            print(f"  [INVESTING.COM] 기술적 분석 요약: {ts}")

        # 매수 사유 (우선순위별 정렬)
        print()
        print("  [BUY SIGNAL / 매수 사유 - 우선순위순]")
        # P1: Volume
        print(f"    [P1] Volume   : {r['vol_ratio']:.1f}x vs 5D avg")
        if r["vol_ratio"] >= 3.0:
            print(f"         -> 강력한 세력/기관 개입 신호. 엔진 시동 완료.")
        elif r["vol_ratio"] >= 2.0:
            print(f"         -> 평소 대비 2배 이상 거래량 확인. 큰손 유입 감지.")
        else:
            print(f"         -> 거래량 기준 미달. 상승 동력 부족 가능성.")
        # P2: Resistance
        print(
            f"    [P2] Overhead : 3M high ${r['three_month_high']:.2f} "
            f"(+{(r['three_month_high']/r['close']-1)*100:.1f}%)"
        )
        if r["three_month_high"] <= r["close"]:
            print(f"         -> 신고가 돌파 구간! 위에 매물대 없음. 최상의 조건.")
        else:
            print(f"         -> +5% 이내 저항 없음. 목적지까지 장애물 없는 구간.")
        # P3: Trend & Momentum
        gc_days = r["conditions"].get("gc_days_ago", 0)
        gc_desc = "당일" if gc_days == 0 else f"{gc_days}일 전"
        print(f"    [P3] Trend    : Close(${r['close']:.2f}) > MA20(${r['ma20']:.2f}), MA20 우상향")
        print(f"         Momentum : MA5/MA20 골든크로스 ({gc_desc} 발생)")
        print(f"         -> 진입 타이밍 확인용. 추세 살아있으나 과거 데이터 기반임을 유의.")
        # P4: RSI
        print(f"    [P4] RSI(14)  : {r['rsi']:.1f}")
        if r["rsi"] <= 50:
            print(f"         -> 상승 초기 구간. 보조 지표로 양호.")
        elif r["rsi"] <= 60:
            print(f"         -> 상승 중반. 강한 종목은 70~80에서도 상승 지속 가능.")
        else:
            print(f"         -> RSI 단독으로는 결정적이지 않음. 거래량과 함께 판단 필요.")

        # 매매 계획
        print()
        print("  [TRADE PLAN / 매매 계획]")
        if tp:
            print(f"    구매추천가 : ${tp['ideal_entry']:.2f}  ({tp['entry_reason']})")
            print(f"    최대진입가 : ${tp['max_entry']:.2f}  (이 가격 초과 시 추격매수 금지)")
            print(f"    손절가     : ${tp['stop_loss']:.2f}  ({tp['sl_pct']:+.1f}%)  ({tp['sl_reason']})")
            print(f"    목표매도가 : ${tp['target']:.2f}  (+{tp['tp_pct']:.1f}%)  ({tp['tp_reason']})")
            print(f"    R:R Ratio  : 1 : {tp['rr_ratio']:.1f}")
            if tp.get('atr'):
                print(f"    ATR(14)    : ${tp['atr']:.2f}  |  10D Swing Low: ${tp['swing_low_10d']:.2f}")
        else:
            tp_price = r["close"] * (1 + TP_PCT)
            sl_price = r["close"] * (1 - SL_PCT)
            print(f"    Target(TP) : ${tp_price:.2f}  (+{TP_PCT*100:.0f}%)")
            print(f"    Stop  (SL) : ${sl_price:.2f}  (-{SL_PCT*100:.0f}%)")
            print(f"    R:R Ratio  : 1 : {TP_PCT/SL_PCT:.1f}")

        # 백테스트 결과
        print()
        if bt.get("total_trades", 0) > 0:
            print("  [BACKTEST / 백테스트 결과 (1Y)]")
            print(
                f"    Trades : {bt['total_trades']}  |  "
                f"Win: {bt['wins']}  |  Loss: {bt['losses']}"
            )
            print(
                f"    Win Rate     : {bt['win_rate']:.1f}%"
            )
            print(
                f"    Expected Val : {bt['expected_value']:+.2f}% per trade"
            )
        else:
            print("  [BACKTEST]")
            print("    No historical entry signals in past 1Y (first occurrence)")

        # 종합 확신도
        conviction = _calc_conviction(r)
        print()
        print(f"  [CONVICTION / 종합 확신도: {conviction['label']}]")
        print(f"    점수: {conviction['score']}/10  ({conviction['desc']})")

        # 리스크
        print()
        print("  [RISK / 주의사항]")
        print("    - 기술적 분석 참고 자료이며 투자 권유가 아닙니다")
        sl_display = tp.get("stop_loss", r["close"] * (1 - SL_PCT))
        print(f"    - 손절가 ${sl_display:.2f} 하회 시 즉시 청산 권장")
        print("    - 실적 발표, 금리 결정 등 거시경제 이벤트 일정 확인 필요")
        if r["vol_ratio"] < 2.5:
            print(f"    - [P1] 거래량({r['vol_ratio']:.1f}x)이 기준 근처. 추가 거래량 확인 후 진입 고려")
        if r["vol_ratio"] > 4:
            print(f"    - [P1] 거래량 급증({r['vol_ratio']:.1f}x). 단기 변동성 확대 주의")
        if r["three_month_high"] > r["close"] * 1.02:
            pct_to_high = (r["three_month_high"] / r["close"] - 1) * 100
            print(f"    - [P2] 3개월 최고가까지 {pct_to_high:.1f}% 남음. 목표가 근처 매물 출회 가능")
        if r["rsi"] > 55:
            print(f"    - [P4] RSI({r['rsi']:.1f}) 상단 근접. 단, 강한 종목은 과열권에서도 상승 지속")

    # ---- Near-miss 참고 ----
    if near_misses:
        # 우선순위 가중치로 정렬: P1(거래량)+P2(매물대) 통과한 종목 우선
        near_misses.sort(
            key=lambda x: (
                x["conditions"]["volume_ok"],
                x["conditions"]["no_resistance"],
                -x["score"],
            ),
            reverse=True,
        )
        print()
        print("-" * 70)
        print(f"  [REF] Near-miss stocks ({len(near_misses)} found):")
        print("-" * 70)
        for nm in near_misses[:10]:
            failed = _get_failed_conditions_priority(nm)
            url = _stock_url(nm["ticker"])
            ts = nm.get("tech_summary", "-")
            tp = nm.get("trade_prices", {})
            entry = tp.get("ideal_entry", nm["close"])
            sl = tp.get("stop_loss", nm["close"] * 0.95)
            tgt = tp.get("target", nm["close"] * 1.10)
            rr = tp.get("rr_ratio", 0)
            print(
                f"    {nm['ticker']:>8s}  ${nm['close']:>9.2f}  "
                f"Entry ${entry:.2f}  SL ${sl:.2f}  Target ${tgt:.2f}  R:R 1:{rr:.1f}  "
                f"RSI {nm['rsi']:.1f}  {nm['vol_ratio']:.1f}x  {ts}  | {failed}"
            )
            print(f"              {url}")

    # ---- Priority Legend ----
    print()
    print("-" * 70)
    print("  [PRIORITY GUIDE / 조건 우선순위 참고]")
    print("-" * 70)
    print("    P1 Volume     (★★★★★) 엔진의 시동. 거래량은 속일 수 없다.")
    print("    P2 Resistance (★★★★☆) 목적지까지의 장애물. 매물대 없어야 가볍게 상승.")
    print("    P3 Trend/MA   (★★★☆☆) 진입 타이밍. 과거 데이터 기반, 미래 보장 아님.")
    print("    P4 RSI        (★★☆☆☆) 보조 지표. 강한 종목은 과열권에서도 상승 지속.")

    # ---- Disclaimer ----
    print()
    print("=" * 70)
    print("  DISCLAIMER: 기술적 분석 참고 자료이며 투자 권유가 아닙니다.")
    print("  모든 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.")
    print("=" * 70)
    print()


def _calc_conviction(stock):
    """우선순위 가중 확신도를 계산한다.

    가중치: P1(거래량)=4, P2(매물대)=3, P3(추세+모멘텀)=2, P4(RSI)=1  -> 합계 10
    거래량 크기에 따라 P1 세분화: 2x=3점, 3x+=4점
    """
    c = stock["conditions"]
    score = 0
    # P1: 거래량 (최대 4점) - 크기에 따라 차등
    if c["volume_ok"]:
        if stock["vol_ratio"] >= 3.0:
            score += 4  # 강력한 세력 개입
        else:
            score += 3  # 기준 통과
    # P2: 매물대 (3점)
    if c["no_resistance"]:
        score += 3
    # P3: 추세 + 모멘텀 (각 1점 = 2점)
    if c["trend"]:
        score += 1
    if c["golden_cross"]:
        score += 1
    # P4: RSI (1점)
    if c["rsi_ok"]:
        score += 1

    if score >= 9:
        label = "A+ (매우 높음)"
        desc = "핵심 조건(거래량+매물대) 완벽. 자신감 있는 진입 가능."
    elif score >= 7:
        label = "A  (높음)"
        desc = "주요 조건 충족. P1/P2 중 하나 미달 시 주의."
    elif score >= 5:
        label = "B  (보통)"
        desc = "일부 핵심 조건 미달. 추가 확인 후 진입 권장."
    else:
        label = "C  (낮음)"
        desc = "핵심 동력 부족. 관망 또는 소액 테스트 진입만 고려."

    return {"score": score, "label": label, "desc": desc}


def _get_failed_conditions(stock):
    """통과하지 못한 조건을 문자열로 반환."""
    c = stock["conditions"]
    failed = []
    if not c["trend"]:
        failed.append("Trend")
    if not c["golden_cross"]:
        failed.append("GoldenCross")
    if not c["rsi_ok"]:
        failed.append(f"RSI({stock['rsi']:.0f})")
    if not c["volume_ok"]:
        failed.append(f"Vol({stock['vol_ratio']:.1f}x)")
    if not c["no_resistance"]:
        failed.append("Resistance")
    return ", ".join(failed) if failed else "None"


def _get_failed_conditions_priority(stock):
    """우선순위 라벨 포함하여 실패 조건을 반환한다."""
    c = stock["conditions"]
    failed = []
    # P1 (최중요)
    if not c["volume_ok"]:
        failed.append(f"[P1]Vol({stock['vol_ratio']:.1f}x)")
    # P2
    if not c["no_resistance"]:
        failed.append("[P2]Resist")
    # P3
    if not c["trend"]:
        failed.append("[P3]Trend")
    if not c["golden_cross"]:
        failed.append("[P3]GC")
    # P4
    if not c["rsi_ok"]:
        failed.append(f"[P4]RSI({stock['rsi']:.0f})")

    if not failed:
        return "All passed"
    # 핵심 조건(P1/P2) 실패 시 경고 추가
    has_p1_fail = not c["volume_ok"]
    has_p2_fail = not c["no_resistance"]
    result = ", ".join(failed)
    if has_p1_fail and has_p2_fail:
        result += "  << 엔진+장애물 모두 미달"
    elif has_p1_fail:
        result += "  << 엔진(거래량) 미점화"
    elif has_p2_fail:
        result += "  << 위에 매물대 존재"
    return result


# ============================================================
# Main
# ============================================================
def main():
    # 인자 파싱
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
        print(f"\n  Custom tickers: {', '.join(tickers)}")
    else:
        tickers = DEFAULT_TICKERS

    print_header()
    print(f"\n  Scanning {len(tickers)} stocks...")

    # Phase 1: Research
    results, near_misses = find_rbi_stocks(tickers)

    # Phase 2: Backtest (통과 종목만)
    if results:
        print(f"\n  Running backtests on {len(results)} qualified stocks...")
        for r in results:
            r["backtest"] = run_backtest(r["df"])
            r["trade_prices"] = calc_trade_prices(r, r["df"])
            # 종목명 조회
            r["name"] = get_company_name(r["ticker"])
            del r["df"]  # 메모리 해제

    # Investing.com 기술적 분석 요약 조회
    fetch_tech_summaries(results)
    fetch_tech_summaries(near_misses)

    # Phase 3: Output
    print_results(results, near_misses)


if __name__ == "__main__":
    main()
