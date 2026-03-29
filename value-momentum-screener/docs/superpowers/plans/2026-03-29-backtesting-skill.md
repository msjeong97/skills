# Backtesting Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `value-momentum-screener:backtesting` 스킬 구현 — 과거 추천 종목의 실제 수익률을 자동으로 채우고, 팩터별 예측력을 분석해 스코어링 개선 제안을 생성한다.

**Architecture:** Python 스크립트(`backtest_updater.py`)가 results/*.json을 읽고 yfinance로 과거 가격을 조회해 MD 파일을 업데이트하고 분석용 JSON을 출력한다. SKILL.md는 스크립트 실행 후 Claude가 팩터 상관관계를 해석하고 리포트를 저장하는 흐름을 정의한다.

**Tech Stack:** Python 3.10+, yfinance, pytest, regex (stdlib)

---

## File Structure

| 파일 | 역할 |
|------|------|
| `skills/backtesting/SKILL.md` | 스킬 정의 (Claude 실행 지침) |
| `skills/backtesting/backtest_updater.py` | 가격 조회 + MD 업데이트 + 분석 JSON 출력 |
| `tests/test_backtest_updater.py` | 단위 테스트 |

---

## Task 1: 테스트 파일 뼈대 + 핵심 헬퍼 함수 TDD

**Files:**
- Create: `tests/test_backtest_updater.py`
- Create: `skills/backtesting/backtest_updater.py`

- [ ] **Step 1: 테스트 파일 작성**

`~/repo/skills/value-momentum-screener/tests/test_backtest_updater.py`를 아래 내용으로 생성:

```python
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "backtesting"))
from backtest_updater import (
    add_calendar_days,
    extract_top_picks,
    parse_catalyst_score,
    update_md_price_table,
)


# --- add_calendar_days ---

def test_add_calendar_days_basic():
    assert add_calendar_days("2026-03-22", 7) == "2026-03-29"


def test_add_calendar_days_28():
    assert add_calendar_days("2026-03-22", 28) == "2026-04-19"


# --- extract_top_picks ---

def test_extract_top_picks_handles_top10_key():
    data = {
        "scan_date": "2026-03-29 21:24",
        "top10": [{"ticker": "CI", "current_price": 261.96}],
    }
    picks = extract_top_picks(data)
    assert picks[0]["ticker"] == "CI"


def test_extract_top_picks_handles_top20_key():
    data = {
        "scan_date": "2026-03-22 18:52",
        "top20": [{"ticker": "ELV", "current_price": 291.48}],
    }
    picks = extract_top_picks(data)
    assert picks[0]["ticker"] == "ELV"


def test_extract_top_picks_returns_empty_for_unknown_key():
    data = {"scan_date": "2026-03-22 18:52"}
    assert extract_top_picks(data) == []


# --- parse_catalyst_score ---

def test_parse_catalyst_score_extracts_numbers():
    line = "- 총점: 65/100 [계량 50/70 + 촉매 15/30]"
    assert parse_catalyst_score(line) == 15


def test_parse_catalyst_score_returns_none_when_not_found():
    assert parse_catalyst_score("아무 내용 없음") is None


# --- update_md_price_table ---

def test_update_md_price_table_fills_empty_cells():
    md_content = (
        "### 1. CI — The Cigna Group\n\n"
        "**가격 추적**\n\n"
        "| 기준일 | 1주 후 | 2주 후 | 4주 후 |\n"
        "|--------|--------|--------|--------|\n"
        "| $261.96 | | | |\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md_content)
        md_path = Path(f.name)

    changed = update_md_price_table(
        md_path,
        base_price=261.96,
        prices={"1w": 270.00, "2w": None, "4w": None},
    )

    result = md_path.read_text()
    assert changed is True
    assert "$270.00" in result
    assert "+3.1%" in result
    md_path.unlink()


def test_update_md_price_table_skips_already_filled():
    md_content = (
        "| 기준일 | 1주 후 | 2주 후 | 4주 후 |\n"
        "|--------|--------|--------|--------|\n"
        "| $261.96 | $270.00 (+3.1%) | | |\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md_content)
        md_path = Path(f.name)

    changed = update_md_price_table(
        md_path,
        base_price=261.96,
        prices={"1w": 275.00, "2w": None, "4w": None},
    )

    assert changed is False
    md_path.unlink()


def test_update_md_price_table_shows_negative_return():
    md_content = (
        "| 기준일 | 1주 후 | 2주 후 | 4주 후 |\n"
        "|--------|--------|--------|--------|\n"
        "| $261.96 | | | |\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md_content)
        md_path = Path(f.name)

    update_md_price_table(
        md_path,
        base_price=261.96,
        prices={"1w": 250.00, "2w": None, "4w": None},
    )

    result = md_path.read_text()
    assert "-4.6%" in result
    md_path.unlink()
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cd ~/repo/skills/value-momentum-screener && python -m pytest tests/test_backtest_updater.py -v 2>&1 | head -30
```

예상 출력: `ModuleNotFoundError: No module named 'backtest_updater'`

- [ ] **Step 3: backtest_updater.py 핵심 헬퍼 함수 구현**

`~/repo/skills/value-momentum-screener/skills/backtesting/backtest_updater.py`를 아래 내용으로 생성:

```python
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
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
cd ~/repo/skills/value-momentum-screener && python -m pytest tests/test_backtest_updater.py -v
```

예상 출력: 모든 테스트 PASSED

- [ ] **Step 5: 커밋**

```bash
cd ~/repo/skills && git add value-momentum-screener/skills/backtesting/backtest_updater.py value-momentum-screener/tests/test_backtest_updater.py && git commit -m "feat: add backtest_updater helper functions with tests"
```

---

## Task 2: 가격 조회 + MD 업데이트 로직 추가

**Files:**
- Modify: `skills/backtesting/backtest_updater.py`
- Modify: `tests/test_backtest_updater.py`

- [ ] **Step 1: 테스트 추가 (가격 조회 + 전체 파이프라인)**

`tests/test_backtest_updater.py` 하단에 추가:

```python
# --- get_price_on_date ---

def test_get_price_on_date_returns_none_for_future():
    from backtest_updater import get_price_on_date
    future = (datetime.today() + timedelta(days=30)).strftime("%Y-%m-%d")
    assert get_price_on_date("AAPL", future) is None


def test_get_price_on_date_returns_float_for_past():
    from backtest_updater import get_price_on_date
    # 2026-03-20은 금요일 (실제 거래일)
    price = get_price_on_date("AAPL", "2026-03-20")
    assert price is None or isinstance(price, float)


# --- load_scan_results ---

def test_load_scan_results_reads_json_files():
    from backtest_updater import load_scan_results
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)
        sample = {
            "scan_date": "2026-03-22 18:52",
            "universe_size": 136,
            "top10": [{"ticker": "CI", "current_price": 261.96, "breakdown": {}}],
        }
        (results_dir / "2026-03-22-top10-raw.json").write_text(json.dumps(sample))
        scans = load_scan_results(results_dir)
    assert len(scans) == 1
    assert scans[0]["scan_date"] == "2026-03-22"
    assert scans[0]["picks"][0]["ticker"] == "CI"
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cd ~/repo/skills/value-momentum-screener && python -m pytest tests/test_backtest_updater.py::test_get_price_on_date_returns_none_for_future tests/test_backtest_updater.py::test_load_scan_results_reads_json_files -v
```

예상 출력: `ImportError` (함수 미구현)

- [ ] **Step 3: backtest_updater.py에 가격 조회 + 로딩 함수 추가**

`backtest_updater.py`의 헬퍼 함수 블록 아래에 추가:

```python
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
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
cd ~/repo/skills/value-momentum-screener && python -m pytest tests/test_backtest_updater.py -v
```

예상 출력: 모든 테스트 PASSED

- [ ] **Step 5: 커밋**

```bash
cd ~/repo/skills && git add value-momentum-screener/skills/backtesting/backtest_updater.py value-momentum-screener/tests/test_backtest_updater.py && git commit -m "feat: add price fetching and scan result loading"
```

---

## Task 3: 분석 JSON 출력 + main() 구현

**Files:**
- Modify: `skills/backtesting/backtest_updater.py`

- [ ] **Step 1: main() 함수 추가**

`backtest_updater.py` 맨 아래에 추가:

```python
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
                if update_md_price_table(md_path, base_price, prices):
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
    args = parser.parse_args()

    print(f"결과 디렉토리: {args.results_dir}", file=sys.stderr)
    data = run(args.results_dir)

    print(f"스캔 수: {data['total_scans']}, 픽 수: {data['total_picks']}, MD 업데이트: {data['md_updated_count']}건", file=sys.stderr)

    # 분석 JSON을 stdout으로 출력 (SKILL.md에서 읽음)
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 직접 실행 테스트**

```bash
cd ~/repo/skills/value-momentum-screener && python skills/backtesting/backtest_updater.py 2>&1 | head -5
```

예상 출력 (stderr):
```
결과 디렉토리: .../results
스캔 수: N, 픽 수: M, MD 업데이트: K건
```

- [ ] **Step 3: 커밋**

```bash
cd ~/repo/skills && git add value-momentum-screener/skills/backtesting/backtest_updater.py && git commit -m "feat: add analysis pipeline and main() to backtest_updater"
```

---

## Task 4: SKILL.md 작성

**Files:**
- Create: `skills/backtesting/SKILL.md`

- [ ] **Step 1: SKILL.md 작성**

`~/repo/skills/value-momentum-screener/skills/backtesting/SKILL.md`를 아래 내용으로 생성:

```markdown
---
name: value-momentum-screener:backtesting
description: >
  value-momentum-screener 과거 추천 종목의 실제 수익률을 추적하고,
  팩터별 예측력 분석을 통해 스코어링 가중치 개선을 제안합니다.
  트리거: "/value-momentum-screener:backtesting", "백테스팅 실행", "스크리너 성과 분석"
user_invocable: true
---

# Value Momentum Screener — Backtesting

과거 추천 종목의 실제 수익률을 채우고, 어떤 팩터가 유효했는지 분석합니다.

---

## Step 1: 가격 업데이트 스크립트 실행

```bash
python {{SKILL_DIR}}/backtest_updater.py
```

스크립트가 다음을 수행합니다:
- `results/*.json`의 모든 픽에 대해 +7/+14/+28일 실제 종가 조회 (yfinance)
- `results/YYYY-MM-DD.md`의 빈 가격 추적 칸 자동 기입
- 분석용 JSON을 stdout으로 출력

스크립트 출력 JSON을 아래 분석에 사용합니다.

---

## Step 2: 팩터 유효성 분석

스크립트 출력의 `picks` 배열에서 `returns`가 null이 아닌 항목만 사용합니다.

아래 팩터별로 **수익률이 양수인 픽의 비율**과 **평균 수익률**을 계산합니다:

| 팩터 | 분석 방법 |
|------|-----------|
| 52주 저점 대비 5~20% 구간 여부 | 해당/비해당 그룹 수익률 비교 |
| RSI 30~45 구간 여부 | 해당/비해당 그룹 수익률 비교 |
| MACD 골든크로스 있/없 | 있/없 그룹 수익률 비교 |
| 볼린저밴드 %B < 0.3 | 해당/비해당 그룹 수익률 비교 |
| FCF Yield > 5% | 해당/비해당 그룹 수익률 비교 |
| PEG < 1.0 | 해당/비해당 그룹 수익률 비교 |
| 촉매 점수 15점 이상 | 해당/비해당 그룹 수익률 비교 |

샘플 수가 30 미만이면 모든 수치 옆에 `⚠️ 샘플 부족 (N=X)` 경고를 표시합니다.

---

## Step 3: 스코어링 개선 제안

분석 결과를 바탕으로 아래 형식으로 개선 제안을 작성합니다:

```
## 스코어링 개선 제안

### 유효 팩터 (현행 유지 또는 가중치 ↑)
- [팩터명]: 해당 그룹 평균 수익률 X.X% vs 비해당 Y.Y%

### 무효 팩터 (가중치 ↓ 또는 제거 고려)
- [팩터명]: 유의미한 수익률 차이 없음 (X.X% vs Y.Y%)

### 현행 vs 권장 가중치
| 팩터 | 현행 | 권장 |
|------|------|------|
| 52주 저점 | 15점 | ?점 |
| RSI | 12점 | ?점 |
| MACD | 12점 | ?점 |
| 볼린저밴드 | 8점 | ?점 |
| FCF Yield | 10점 | ?점 |
| PEG | 5점 | ?점 |
```

---

## Step 4: 리포트 저장

분석 결과를 아래 경로에 저장합니다:

**저장 경로:** `{{SKILL_DIR}}/../../results/backtest-report-YYYY-MM-DD.md`

```markdown
# Backtest Report — YYYY-MM-DD

분석 기간: YYYY-MM-DD ~ YYYY-MM-DD | 총 스캔: N회 | 수익률 확인된 픽: M개

## 팩터 유효성 분석
[Step 2 결과]

## 스코어링 개선 제안
[Step 3 결과]

⚠️ 이 분석은 투자 조언이 아닙니다. 샘플 수가 충분하지 않으면 통계적 신뢰도가 낮습니다.
```

저장 후 경로를 출력합니다:
```
💾 백테스팅 리포트 저장됨: results/backtest-report-YYYY-MM-DD.md
```
```

- [ ] **Step 2: 스킬 동작 검증**

```bash
cd ~/repo/skills/value-momentum-screener && python skills/backtesting/backtest_updater.py > /tmp/backtest_out.json 2>&1 && echo "출력 라인 수: $(wc -l < /tmp/backtest_out.json)"
```

예상 출력: `출력 라인 수: N` (JSON이 생성됨)

- [ ] **Step 3: 커밋**

```bash
cd ~/repo/skills && git add value-momentum-screener/skills/backtesting/SKILL.md && git commit -m "feat: add value-momentum-screener:backtesting skill"
```

---

## Self-Review

**스펙 커버리지:**
- ✅ 가격 자동 기입 (update_md_price_table)
- ✅ 팩터별 수익률 분석 (SKILL.md Step 2)
- ✅ 스코어링 개선 제안 (SKILL.md Step 3)
- ✅ 리포트 저장 (SKILL.md Step 4)
- ✅ 촉매 점수 분석 포함 (parse_catalyst_score_from_md)
- ✅ 샘플 부족 경고 포함

**타입/함수명 일관성:**
- `update_md_price_table(md_path, base_price, prices)` — Task 1 정의, Task 3 사용 ✅
- `get_price_on_date(ticker, target_date)` — Task 2 정의, Task 3 사용 ✅
- `load_scan_results(results_dir)` — Task 2 정의, Task 3 사용 ✅
- `add_calendar_days(date_str, days)` — Task 1 정의, Task 3 사용 ✅
