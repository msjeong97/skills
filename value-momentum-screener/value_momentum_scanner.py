#!/usr/bin/env python3
"""
Value Momentum Screener v2.0
미국 시총 상위 150개 우량주 중 저평가 + 단기 반등 신호 종목 Top 10 출력

Usage:
    python value_momentum_scanner.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import json
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


def parse_args(argv=None):
    """CLI 인자 파싱."""
    parser = argparse.ArgumentParser(description="Value Momentum Scanner")
    parser.add_argument(
        "--json-only",
        action="store_true",
        dest="json_only",
        help="터미널 서식 출력 생략, JSON만 출력 (토큰 절감용)",
    )
    return parser.parse_args(argv)


# ── 시총 상위 150개 우량주 (금융주 제외, 분기 1회 업데이트) ─────────────────
TOP_150_TICKERS = [
    # Mega Cap Tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "NFLX",
    # Semiconductors
    "AMD", "QCOM", "TXN", "AMAT", "MU", "LRCX", "KLAC", "MRVL", "ADI", "NXPI",
    "INTC", "CSCO", "SNPS", "CDNS", "MCHP",
    # Cloud / SaaS / Cybersecurity
    "CRM", "NOW", "ADBE", "INTU", "WDAY", "SNOW", "DDOG", "CRWD", "ZS", "PANW",
    "FTNT", "NET", "PLTR", "IBM", "FICO",
    # Consumer Discretionary
    "AMZN", "TSLA", "NKE", "SBUX", "MCD", "CMG", "COST", "WMT", "TGT", "HD",
    "LOW", "BKNG", "MAR", "HLT", "ABNB", "EBAY", "ETSY",
    # Healthcare / Biotech / Pharma
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BSX",
    "ISRG", "VRTX", "REGN", "GILD", "AMGN", "ZTS", "ELV", "CI", "CVS", "HCA",
    # Industrials / Aerospace
    "CAT", "DE", "HON", "RTX", "LMT", "GE", "ETN", "ITW", "PH", "ROK",
    "EMR", "CMI", "PCAR", "FDX", "UPS", "UNP", "CSX", "NSC", "GEV",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY",
    # Communication / Media
    "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
    # Materials
    "LIN", "APD", "ECL", "SHW", "NEM", "FCX",
    # Utilities
    "NEE", "DUK", "SO", "AEP",
    # Real Estate (REITs - non-financial)
    "AMT", "PLD", "CCI", "EQIX", "PSA", "WELL",
    # Payment / Fintech (non-bank)
    "V", "MA", "PYPL",
    # Other
    "UBER", "SPOT", "TTD", "ADSK", "IDXX", "EPAM", "IT", "MSTR",
]

# 금융주 제외 목록
FINANCIAL_TICKERS = {
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB",
    "PNC", "TFC", "COF", "DFS", "SYF", "AIG", "MET", "PRU", "AFL", "ALL",
    "MMC", "AON", "TRV", "CB", "ICE", "CME", "SPGI", "MCO", "FIS", "FISV",
}

# 중복 제거 + 금융주 제외
TOP_150_TICKERS = list(dict.fromkeys(
    [t for t in TOP_150_TICKERS if t not in FINANCIAL_TICKERS]
))[:150]


# ── 데이터 수집 ──────────────────────────────────────────────────────────────

def collect_ticker_data(ticker_symbol: str) -> dict:
    """단일 종목 데이터 수집. 실패 시 None 반환."""
    try:
        t = yf.Ticker(ticker_symbol)
        info = t.info

        if not info or not info.get('shortName'):
            return None

        hist = t.history(period="1y")
        if hist.empty or len(hist) < 60:
            return None

        return {
            'ticker': ticker_symbol,
            'name': info.get('shortName', ticker_symbol),
            'sector': info.get('sector', 'Unknown'),
            'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
            'market_cap': info.get('marketCap'),
            'trailing_pe': info.get('trailingPE'),
            'peg_ratio': info.get('trailingPegRatio'),
            # yfinance debtToEquity는 퍼센트 단위 (100 = D/E 1.0)
            'debt_to_equity': info.get('debtToEquity'),
            'roe': info.get('returnOnEquity'),
            '52w_low': info.get('fiftyTwoWeekLow'),
            '52w_high': info.get('fiftyTwoWeekHigh'),
            'hist': hist,
            'cashflow': t.cashflow,
            'quarterly_cashflow': t.quarterly_cashflow,
            'quarterly_financials': t.quarterly_financials,
            'info': info,
        }
    except Exception as e:
        print(f"  [SKIP] {ticker_symbol}: {type(e).__name__}: {e}")
        return None


def collect_all_data(tickers: list) -> list:
    """병렬로 전체 종목 데이터 수집."""
    results = []
    print(f"  데이터 수집 중 (병렬, 최대 8 workers)...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(collect_ticker_data, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures), 1):
            data = future.result()
            if data:
                results.append(data)
            if i % 30 == 0:
                print(f"    {i}/{len(tickers)} 처리 완료 (수집 성공: {len(results)}개)...")
    return results


# ── 사전 자격 필터 ───────────────────────────────────────────────────────────

def _get_quarterly_op_income(d: dict):
    """최근 4분기 영업이익 시리즈 반환. 없으면 None."""
    try:
        qf = d.get('quarterly_financials')
        if qf is None or qf.empty:
            return None
        # yfinance 버전별 키 이름 fallback
        for key in ['Operating Income', 'EBIT', 'Total Operating Income As Reported',
                    'Operating Income Loss']:
            if key in qf.index:
                return qf.loc[key].iloc[:4]
        return None
    except Exception:
        return None


def apply_qualification_filter(all_data: list) -> list:
    """Pass/Fail 필터: ROE > 10%, 최근 4분기 영업이익 흑자, D/E < 100 (퍼센트 단위)"""
    qualified = []
    skipped = []

    for d in all_data:
        ticker = d['ticker']
        reasons = []

        # ROE 필터 (returnOnEquity는 소수점, 0.10 = 10%)
        roe = d.get('roe')
        if roe is None:
            reasons.append("ROE 데이터 없음")
        elif roe <= 0.10:
            reasons.append(f"ROE {roe:.1%} ≤ 10%")

        # D/E 필터 (yfinance는 퍼센트 단위: 100 = D/E 1.0)
        de = d.get('debt_to_equity')
        if de is None:
            pass  # D/E 없으면 허용 (일부 종목 데이터 미제공)
        elif de >= 100:
            reasons.append(f"D/E {de/100:.1f} ≥ 1.0 (과부채)")

        # 영업이익 4분기 흑자 필터
        op_series = _get_quarterly_op_income(d)
        if op_series is not None and len(op_series) >= 2:
            if (op_series <= 0).any():
                reasons.append("최근 4분기 중 영업이익 적자 분기 존재")

        if reasons:
            skipped.append((ticker, reasons))
        else:
            qualified.append(d)

    if skipped:
        print(f"  ⛔ 자격 미달 {len(skipped)}개 (처음 5개):")
        for ticker, reasons in skipped[:5]:
            print(f"    {ticker}: {', '.join(reasons)}")
        if len(skipped) > 5:
            print(f"    ... 외 {len(skipped)-5}개")

    return qualified


# ── 섹터 통계 ────────────────────────────────────────────────────────────────

def _calc_op_margin(d: dict):
    """TTM 영업이익률 계산."""
    try:
        qf = d.get('quarterly_financials')
        if qf is None or qf.empty:
            return None
        op_key = next((k for k in ['Operating Income', 'EBIT', 'Total Operating Income As Reported']
                       if k in qf.index), None)
        rev_key = next((k for k in ['Total Revenue', 'Revenue', 'Revenues']
                        if k in qf.index), None)
        if not op_key or not rev_key:
            return None
        op_income = qf.loc[op_key].iloc[:4].sum()
        revenue = qf.loc[rev_key].iloc[:4].sum()
        if revenue <= 0:
            return None
        return float(op_income / revenue)
    except Exception:
        return None


def _calc_fcf_yield(d: dict):
    """FCF Yield 계산 (연간 FCF / 시총)."""
    try:
        market_cap = d.get('market_cap')
        if not market_cap or market_cap <= 0:
            return None

        # 1순위: 연간 cashflow에서 Free Cash Flow 직접
        cf = d.get('cashflow')
        if cf is not None and not cf.empty:
            for key in ['Free Cash Flow', 'FreeCashFlow']:
                if key in cf.index:
                    fcf = float(cf.loc[key].iloc[0])
                    return fcf / market_cap

            # Operating CF - CapEx 계산
            op_key = next((k for k in ['Operating Cash Flow',
                                       'Cash Flow From Continuing Operating Activities',
                                       'Net Cash Provided By Operating Activities']
                           if k in cf.index), None)
            cap_key = next((k for k in ['Capital Expenditure', 'Purchase Of PPE',
                                        'Purchases Of Property Plant And Equipment']
                            if k in cf.index), None)
            if op_key and cap_key:
                op_cf = float(cf.loc[op_key].iloc[0])
                capex = float(cf.loc[cap_key].iloc[0])  # 음수로 표기됨
                return (op_cf + capex) / market_cap

        # 2순위: 분기 cashflow TTM
        qcf = d.get('quarterly_cashflow')
        if qcf is not None and not qcf.empty:
            op_key = next((k for k in ['Operating Cash Flow',
                                       'Cash Flow From Continuing Operating Activities']
                           if k in qcf.index), None)
            cap_key = next((k for k in ['Capital Expenditure', 'Purchase Of PPE']
                            if k in qcf.index), None)
            if op_key and cap_key:
                op_cf = qcf.loc[op_key].iloc[:4].sum()
                capex = qcf.loc[cap_key].iloc[:4].sum()
                return float((op_cf + capex) / market_cap)

        return None
    except Exception:
        return None


def compute_sector_stats(qualified: list) -> dict:
    """섹터별 PE 중위수, 영업이익률 분포 계산."""
    sector_data = {}
    for d in qualified:
        sector = d['sector']
        if sector not in sector_data:
            sector_data[sector] = {'pe': [], 'op_margin': []}

        pe = d.get('trailing_pe')
        if pe and isinstance(pe, (int, float)) and 0 < pe < 500:
            sector_data[sector]['pe'].append(float(pe))

        op_margin = _calc_op_margin(d)
        if op_margin is not None and not np.isnan(op_margin):
            sector_data[sector]['op_margin'].append(op_margin)

    stats = {}
    for sector, vals in sector_data.items():
        pe_list = vals['pe']
        n = len(pe_list)
        stats[sector] = {
            'pe_median': float(np.median(pe_list)) if pe_list else None,
            'pe_30pct': float(np.percentile(pe_list, 30)) if n >= 5 else None,
            'pe_50pct': float(np.percentile(pe_list, 50)) if n >= 5 else None,
            'op_margin_list': vals['op_margin'],
            'count': n,
        }
    return stats


# ── 점수 계산 ────────────────────────────────────────────────────────────────

def score_valuation(d: dict, sector_stats: dict) -> tuple:
    """밸류에이션 점수 (25점 만점)."""
    score = 0.0
    detail = {}
    sector = d['sector']
    stats = sector_stats.get(sector, {})

    # PE 점수 (10점)
    pe = d.get('trailing_pe')
    pe_score = 0
    pe_30 = stats.get('pe_30pct')
    pe_50 = stats.get('pe_50pct')
    if pe and isinstance(pe, (int, float)) and pe > 0 and pe_30:
        if pe <= pe_30:
            pe_score = 10
        elif pe_50 and pe <= pe_50:
            pe_score = 5
    detail['pe'] = {'value': round(pe, 1) if pe else None,
                    'sector_median': round(pe_50, 1) if pe_50 else None,
                    'score': pe_score}
    score += pe_score

    # FCF Yield 점수 (10점)
    fcf_yield = _calc_fcf_yield(d)
    fcf_score = 0
    if fcf_yield is not None and not np.isnan(fcf_yield):
        if fcf_yield > 0.05:
            fcf_score = 10
        elif fcf_yield > 0.03:
            fcf_score = 6
        elif fcf_yield > 0.01:
            fcf_score = 3
    detail['fcf_yield'] = {'value': round(fcf_yield * 100, 2) if fcf_yield else None,
                           'score': fcf_score}
    score += fcf_score

    # PEG 점수 (5점)
    peg = d.get('peg_ratio')
    peg_score = 0
    if peg and isinstance(peg, (int, float)) and peg > 0:
        if peg < 1.0:
            peg_score = 5
        elif peg <= 1.5:
            peg_score = 3
    detail['peg'] = {'value': round(peg, 2) if peg else None, 'score': peg_score}
    score += peg_score

    return score, detail


def score_profitability(d: dict, sector_stats: dict) -> tuple:
    """수익성 품질 점수 (20점 만점)."""
    score = 0.0
    detail = {}
    sector = d['sector']
    stats = sector_stats.get(sector, {})

    # ROE 점수 (10점)
    roe = d.get('roe') or 0
    if roe > 0.20:
        roe_score = 10
    elif roe > 0.15:
        roe_score = 7
    elif roe > 0.10:
        roe_score = 4
    else:
        roe_score = 0
    detail['roe'] = {'value': round(roe * 100, 1), 'score': roe_score}
    score += roe_score

    # 영업이익률 섹터 내 백분위 (10점)
    op_margin = _calc_op_margin(d)
    op_score = 0
    op_margin_list = stats.get('op_margin_list', [])
    if op_margin is not None and len(op_margin_list) >= 5:
        percentile = np.mean([op_margin > m for m in op_margin_list]) * 100
        if percentile >= 75:
            op_score = 10
        elif percentile >= 50:
            op_score = 5
    detail['op_margin'] = {
        'value': round(op_margin * 100, 1) if op_margin is not None else None,
        'score': op_score
    }
    score += op_score

    return score, detail


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI 계산."""
    if HAS_PANDAS_TA:
        result = ta.rsi(close, length=period)
        if result is not None and not result.empty:
            return result
    # 수동 계산 fallback
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _calc_macd(close: pd.Series):
    """MACD(12,26,9). (macd_line, signal_line) 반환."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def _calc_bollinger_pct_b(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.Series:
    """볼린저밴드 %B."""
    ma = close.rolling(period).mean()
    std_dev = close.rolling(period).std()
    upper = ma + std_mult * std_dev
    lower = ma - std_mult * std_dev
    band_width = (upper - lower).replace(0, np.nan)
    return (close - lower) / band_width


def score_technical(d: dict) -> tuple:
    """단기 기술적 반전 신호 점수 (55점 만점)."""
    score = 0.0
    detail = {}
    hist = d.get('hist')

    if hist is None or len(hist) < 60:
        return 0.0, {'error': '히스토리 데이터 부족'}

    close = hist['Close'].dropna()
    volume = hist['Volume'].dropna()

    if len(close) < 30:
        return 0.0, {'error': '유효 가격 데이터 부족'}

    # ── 52주 저점 대비 위치 (15점) ──────────────────────────────────────────
    low_52w = d.get('52w_low')
    current = float(close.iloc[-1])
    if low_52w and low_52w > 0:
        pct_from_low = (current - low_52w) / low_52w * 100
    else:
        pct_from_low = (current - float(close.min())) / float(close.min()) * 100

    # 200% 초과는 yfinance 스플릿 미조정 데이터 이슈로 0점 처리
    if 5 <= pct_from_low <= 20:
        low_score = 15
    elif 20 < pct_from_low <= 35:
        low_score = 7
    else:
        low_score = 0
    detail['52w_low'] = {'pct_from_low': round(pct_from_low, 1), 'score': low_score}
    score += low_score

    # ── RSI(14) (12점) ──────────────────────────────────────────────────────
    rsi_series = _calc_rsi(close)
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]) else None
    if rsi is not None:
        if 30 <= rsi <= 45:
            rsi_score = 12
        elif 45 < rsi <= 55:
            rsi_score = 6
        else:
            rsi_score = 0
    else:
        rsi_score = 0
    detail['rsi'] = {'value': round(rsi, 1) if rsi else None, 'score': rsi_score}
    score += rsi_score

    # ── MACD(12,26,9) 골든크로스 최근 5거래일 (12점) ────────────────────────
    macd_line, signal_line = _calc_macd(close)
    macd_score = 0
    if len(macd_line) >= 6:
        recent_macd = macd_line.iloc[-6:].values
        recent_signal = signal_line.iloc[-6:].values
        for i in range(len(recent_macd) - 1):
            if recent_macd[i] < recent_signal[i] and recent_macd[i+1] >= recent_signal[i+1]:
                macd_score = 12
                break
    detail['macd'] = {
        'golden_cross': macd_score > 0,
        'macd_last': round(float(macd_line.iloc[-1]), 4),
        'signal_last': round(float(signal_line.iloc[-1]), 4),
        'score': macd_score
    }
    score += macd_score

    # ── 볼린저밴드 %B 반등 (8점) ────────────────────────────────────────────
    pct_b = _calc_bollinger_pct_b(close)
    bb_score = 0
    current_b = float(pct_b.iloc[-1]) if not pd.isna(pct_b.iloc[-1]) else None
    if len(pct_b) >= 11 and current_b is not None:
        prev_10 = pct_b.iloc[-11:-1].dropna()
        touched_low = any(v < 0.2 for v in prev_10)
        if touched_low and current_b > 0.3:
            bb_score = 8
    detail['bollinger'] = {'pct_b': round(current_b, 3) if current_b is not None else None,
                           'score': bb_score}
    score += bb_score

    # ── 가격+거래량 동반 상승 (8점) ─────────────────────────────────────────
    vol_score = 0
    if len(hist) >= 25:
        try:
            vol_20d_avg = float(volume.iloc[-25:-5].mean())
            recent_hist = hist.iloc[-5:]
            for i in range(len(recent_hist) - 1):
                prev_close = float(recent_hist['Close'].iloc[i])
                curr_close = float(recent_hist['Close'].iloc[i + 1])
                curr_vol = float(recent_hist['Volume'].iloc[i + 1])
                price_chg = (curr_close - prev_close) / prev_close if prev_close > 0 else 0
                vol_ratio = curr_vol / vol_20d_avg if vol_20d_avg > 0 else 0
                if price_chg >= 0.01 and vol_ratio >= 1.5:
                    vol_score = 8
                    break
        except Exception:
            pass
    detail['volume'] = {'score': vol_score}
    score += vol_score

    return score, detail


# ── 종합 점수 계산 ───────────────────────────────────────────────────────────

def compute_scores(qualified: list, sector_stats: dict) -> list:
    """전체 종목 저평가 지수 계산."""
    results = []
    for d in qualified:
        try:
            val_score, val_detail = score_valuation(d, sector_stats)
            prof_score, prof_detail = score_profitability(d, sector_stats)
            tech_score, tech_detail = score_technical(d)
            total = val_score + prof_score + tech_score

            results.append({
                'ticker': d['ticker'],
                'name': d['name'],
                'sector': d['sector'],
                'current_price': d['current_price'],
                '52w_low': d.get('52w_low'),
                '52w_high': d.get('52w_high'),
                'undervalue_score': round(total, 1),
                'breakdown': {
                    'valuation': round(val_score, 1),
                    'profitability': round(prof_score, 1),
                    'technical': round(tech_score, 1),
                },
                'detail': {
                    'valuation': val_detail,
                    'profitability': prof_detail,
                    'technical': tech_detail,
                }
            })
        except Exception as e:
            print(f"  [ERROR] {d['ticker']} 점수 계산 실패: {e}")

    return results


# ── 결과 출력 ────────────────────────────────────────────────────────────────

def output_results(top10: list):
    """상위 10개 결과를 텍스트 + JSON으로 출력."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"\n{'='*65}")
    print(f"  📊 저평가 지수 Top {len(top10)} — {now}")
    print(f"{'='*65}\n")

    for i, r in enumerate(top10, 1):
        price = f"${r['current_price']:.2f}" if r['current_price'] else "N/A"
        low = f"${r['52w_low']:.2f}" if r['52w_low'] else "N/A"
        tech = r['detail']['technical']
        rsi_val = tech.get('rsi', {}).get('value')
        rsi_str = f"RSI {rsi_val:.0f}" if rsi_val else "RSI N/A"
        pct_low = tech.get('52w_low', {}).get('pct_from_low')
        pct_str = f"저점대비+{pct_low:.0f}%" if pct_low is not None else ""
        macd_ok = "MACD✅" if tech.get('macd', {}).get('golden_cross') else "MACD❌"
        bb_ok = "BB✅" if tech.get('bollinger', {}).get('score', 0) > 0 else "BB❌"
        vol_ok = "Vol✅" if tech.get('volume', {}).get('score', 0) > 0 else "Vol❌"

        val = r['detail']['valuation']
        fcf = val.get('fcf_yield', {}).get('value')
        pe = val.get('pe', {}).get('value')
        pe_med = val.get('pe', {}).get('sector_median')
        roe = r['detail']['profitability'].get('roe', {}).get('value')

        print(f"{'─'*65}")
        print(f"  {i:2}. {r['ticker']:<6}  {r['name'][:30]}")
        print(f"      현재가: {price:<10} | 52주저점: {low}")
        print(f"      저평가 지수: {r['undervalue_score']:5.1f}/100  "
              f"[밸류에이션 {r['breakdown']['valuation']:.0f}/25 | "
              f"수익성 {r['breakdown']['profitability']:.0f}/20 | "
              f"기술신호 {r['breakdown']['technical']:.0f}/55]")
        pe_str = f"PE {pe:.0f}x(섹터중위수 {pe_med:.0f}x)" if pe and pe_med else f"PE {pe:.0f}x" if pe else "PE N/A"
        fcf_str = f"FCF {fcf:.1f}%" if fcf else "FCF N/A"
        roe_str = f"ROE {roe:.0f}%" if roe else "ROE N/A"
        print(f"      {pe_str} | {fcf_str} | {roe_str}")
        print(f"      {rsi_str} | {pct_str} | {macd_ok} | {bb_ok} | {vol_ok}")
        print()

    # JSON 출력 (AI 상승 신호 단계에서 활용)
    print(f"\n{'='*65}")
    print(f"  📋 AI 웹서치용 JSON (Batch)")
    print(f"{'='*65}\n")
    json_data = []
    for i, r in enumerate(top10):
        tech = r['detail']['technical']
        val = r['detail']['valuation']
        json_data.append({
            'rank': i + 1,
            'ticker': r['ticker'],
            'name': r['name'],
            'sector': r['sector'],
            'current_price': r['current_price'],
            'quant_score_70': round(r['undervalue_score'] * 0.7, 1),  # 100점 → 70점 리스케일
            'key_signals': {
                'rsi': tech.get('rsi', {}).get('value'),
                'pct_from_52w_low': tech.get('52w_low', {}).get('pct_from_low'),
                'macd_golden_cross': tech.get('macd', {}).get('golden_cross', False),
                'bb_bounce': tech.get('bollinger', {}).get('score', 0) > 0,
                'volume_surge': tech.get('volume', {}).get('score', 0) > 0,
                'fcf_yield_pct': val.get('fcf_yield', {}).get('value'),
                'pe': val.get('pe', {}).get('value'),
            }
        })
    print(json.dumps(json_data, ensure_ascii=False, indent=2))


# ── 결과 저장 ────────────────────────────────────────────────────────────────

def save_top10_json(top10: list, skill_dir: str):
    """Top 10 정량 데이터를 날짜별 JSON으로 저장 (AI 웹서치 단계 및 백테스팅용)."""
    import os
    results_dir = os.path.join(skill_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    date_str = datetime.now().strftime('%Y-%m-%d')
    filepath = os.path.join(results_dir, f"{date_str}-top10-raw.json")

    payload = {
        "scan_date": datetime.now().strftime('%Y-%m-%d %H:%M'),
        "universe_size": len(TOP_150_TICKERS),
        "top10": top10,
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n  💾 Top 10 저장됨: {filepath}")
    return filepath


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.json_only:
        print(f"\n🔍 Value Momentum Screener v1.0")
        print(f"   대상 종목: {len(TOP_150_TICKERS)}개 | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # 1. 데이터 수집
    all_data = collect_all_data(TOP_150_TICKERS)
    if not args.json_only:
        print(f"\n  ✅ 데이터 수집 완료: {len(all_data)}/{len(TOP_150_TICKERS)}개\n")

    if not all_data:
        print("❌ 데이터 수집 실패. 네트워크 연결을 확인하세요.")
        sys.exit(1)

    # 2. 사전 자격 필터
    qualified = apply_qualification_filter(all_data)
    if not args.json_only:
        print(f"\n  ✅ 자격 통과: {len(qualified)}개 종목\n")

    if not qualified:
        print("❌ 자격 통과 종목 없음.")
        sys.exit(1)

    # 3. 섹터 통계
    sector_stats = compute_sector_stats(qualified)

    # 4. 저평가 지수 계산
    scored = compute_scores(qualified, sector_stats)

    if not scored:
        print("❌ 점수 계산 실패.")
        sys.exit(1)

    # 5. 상위 10개 추출
    top10 = sorted(scored, key=lambda x: x['undervalue_score'], reverse=True)[:10]

    # 6. 터미널 서식 출력 (--json-only 시 생략)
    if not args.json_only:
        output_results(top10)

    # 7. Top 10 JSON 파일 저장
    import os
    skill_dir = os.path.dirname(os.path.abspath(__file__))
    save_top10_json(top10, skill_dir)

    if not args.json_only:
        print(f"\n{'='*65}")
        print(f"  ✅ 스캔 완료. 위 JSON을 AI에게 전달해 상승 신호 점수를 받으세요.")
        print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
