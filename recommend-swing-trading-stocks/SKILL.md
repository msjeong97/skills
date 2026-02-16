---
name: recommend-swing-trading-stocks
description: >
  RBI-Scan: 30-day swing trading stock scanner. Analyzes NASDAQ/S&P500 stocks
  using 5 technical conditions (trend, golden cross, RSI, volume, resistance)
  and backtests with TP+10%/SL-5%. Use when user wants stock recommendations
  or swing trading analysis.
user_invocable: true
---

# RBI-Scan: 30-Day Swing Trading Scanner

Moon Dev's R-B-I (Research-Backtest-Implement) system for finding short-term breakout stocks.

## Trigger

This skill is invoked via `/recommend-swing-trading-stocks`.

## Workflow

### Step 1: Run the Scanner

Execute the RBI scanner script. The user may optionally provide specific tickers.

**Default scan (101 NASDAQ + S&P500 stocks):**
```bash
python {{SKILL_DIR}}/rbi_swing_scanner.py
```

**Custom tickers:**
```bash
python {{SKILL_DIR}}/rbi_swing_scanner.py AAPL NVDA TSLA META
```

**KOSPI stocks:**
```bash
python {{SKILL_DIR}}/rbi_swing_scanner.py 005930.KS 000660.KS 035420.KS
```

### Step 2: Interpret the Results

The scanner outputs three sections. Explain each to the user in Korean:

1. **Phase 1 - Research Table**: Stocks passing all 5 RBI conditions
   - Ticker, price, RSI, volume ratio, backtest win rate, expected return

2. **Phase 2 - Detailed Analysis**: Per-stock breakdown
   - Buy signal rationale with priority labels ([P1]~[P4])
   - Trade plan (target price +10%, stop loss -5%, R:R ratio)
   - Backtest results (1Y historical performance)
   - Conviction score (10-point weighted: P1=4, P2=3, P3=2, P4=1)
   - Risk warnings with priority context
   - kr.investing.com link (clickable in terminal)

3. **Near-miss Reference**: Stocks meeting 3-4 out of 5 conditions
   - Shows which condition(s) failed with priority labels
   - Highlights critical failures: "엔진 미점화" (P1), "매물대 존재" (P2)
   - kr.investing.com link per stock

### Step 3: Provide Context

After presenting results, add:

- **Market context**: Brief note on current market conditions that may affect the signals
- **Timing guidance**: Remind the user that signals are based on the most recent trading day's close
- **Disclaimer**: This is technical analysis for educational purposes only, not investment advice

## RBI Filter Conditions (5/5 required)

Conditions are listed by **profitability priority** (P1 = most critical):

| Priority | Condition | Rule | Rationale |
|----------|-----------|------|-----------|
| P1 ★★★★★ | Volume | Volume > 2x 5-day avg | "엔진의 시동" - 거래량은 속일 수 없다. 세력/기관 개입 신호. |
| P2 ★★★★☆ | Resistance | No 3M high within +5% | "장애물 유무" - 위에 매물대 없어야 가볍게 치고 나간다. |
| P3 ★★★☆☆ | Trend | Close > MA20, MA20 rising | 진입 타이밍용. 과거 데이터 기반이며 미래 보장 아님. |
| P3 ★★★☆☆ | Momentum | MA5 x MA20 Golden Cross | 진입 타이밍용. 추세 확인 보조. |
| P4 ★★☆☆☆ | RSI | 40 <= RSI(14) <= 60 | 보조 지표. 강한 종목은 70~80에서도 상승 지속. |

## Dependencies

Install all packages at once:
```bash
pip install -r {{SKILL_DIR}}/requirements.txt
```

| Package | Required | Description |
|---------|----------|-------------|
| `numpy` | Yes | 수치 연산 |
| `pandas` | Yes | 데이터프레임 처리 |
| `yfinance` | Yes | Yahoo Finance 주가 데이터 |
| `pandas_ta` | Optional | 기술적 지표 계산 (미설치 시 RSI 수동 계산 폴백) |
| `investgo` | Optional | Investing.com 기술 분석 데이터 |
| `backtesting` | Optional | 백테스트 엔진 |

## Configuration

Defaults in `rbi_swing_scanner.py` (adjustable):
- TP: +10%, SL: -5%
- Volume multiplier: 2.0x
- RSI range: 40-60
- Golden cross lookback: 5 days
- Resistance check range: +5%
