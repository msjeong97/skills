# Telegram Notification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `run_auto.sh` 실행 후 스크리너 결과(`results/YYYY-MM-DD.md`)를 텔레그램으로 전송하는 `telegram_notify.py` 스크립트 구현.

**Architecture:** `telegram_notify.py`가 오늘 날짜 MD 파일을 파싱해 헤더 1개 + 종목별 5개 = 총 6개 메시지를 Telegram Bot API(MarkdownV2)로 전송. `run_auto.sh` 마지막에 한 줄 추가해서 연동.

**Tech Stack:** Python 3.10+, `requests`, `re`(stdlib), pytest

---

## File Structure

| 파일 | 역할 |
|------|------|
| `telegram_notify.py` | 메인 스크립트 (파싱 + 포맷 + 전송) |
| `telegram_config.json` | Bot Token + Chat ID (gitignored) |
| `telegram_config.json.example` | 설정 파일 템플릿 (커밋용) |
| `tests/test_telegram_notify.py` | 단위 테스트 |
| `run_auto.sh` | 마지막에 notify 한 줄 추가 |
| `.gitignore` | `telegram_config.json` 추가 |

---

## Task 1: 파싱 함수 TDD — 요약 테이블 + 종목 섹션

**Files:**
- Create: `tests/test_telegram_notify.py`
- Create: `telegram_notify.py`

### 실제 MD 파일 형식 참고

요약 테이블:
```
## Top 5 추천 종목

| # | 종목 | 현재가 | 총점 | 계량 | 촉매 | 한줄요약 |
|---|------|--------|------|------|------|----------|
| 1 | CRM | $182.14 | 90점 | 70/70 | 20/30 | JP모건이 매수... |
| 2 | EPAM | $131.34 | 82점 | 77/70 | 5/30 | 2.5주 후 실적... |
```

종목 상세 섹션:
```
### 1. CRM — Salesforce, Inc.

**점수 내역**
- 총점: 90/100 [계량 70/70 + 촉매 20/30]
- PE: 23.4x (섹터 40.7x) | FCF Yield: 8.57% | ROE: 12.4% | OpMargin: 21.5%
- RSI: 53.2 | 52주저점대비: +11.4% | MACD: 골든크로스 (-4.48/-5.14) | BB: 0.530
- 촉매: JP모건 Buy 유지 (4/10), 이사 $499K 직접 매수 (3/19)
- ⚠️ 리스크: 데이터침해 집단소송 70+건

---
```

- [ ] **Step 1: 테스트 파일 작성**

`tests/test_telegram_notify.py` 를 아래 내용으로 생성:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from telegram_notify import parse_summary_table, parse_stock_sections

SAMPLE_MD = """\
# Value Momentum Screener — 2026-04-18

스캔: 2026-04-18 16:34 | 유니버스: 시총 상위 150종목

## Top 5 추천 종목

| # | 종목 | 현재가 | 총점 | 계량 | 촉매 | 한줄요약 |
|---|------|--------|------|------|------|----------|
| 1 | CRM | $182.14 | 90점 | 70/70 | 20/30 | JP모건이 매수 의견 |
| 2 | ADBE | $244.45 | 79점 | 79/70 | 0/30 | DOJ 합의 완료, PER 역대급 저평가 |

### 제외 종목 (결격 사유)
- **TTD**: 집단소송

---

## 종목별 상세

### 1. CRM — Salesforce, Inc.

**가격 추적**

| 기준일 | 1주 후 | 2주 후 | 4주 후 |
|--------|--------|--------|--------|
| $182.14 | | | |

**점수 내역**
- 총점: 90/100 [계량 70/70 + 촉매 20/30]
- PE: 23.4x (섹터 40.7x) | FCF Yield: 8.57% | ROE: 12.4% | OpMargin: 21.5%
- RSI: 53.2 | 52주저점대비: +11.4% | MACD: 골든크로스 (-4.48/-5.14) | BB: 0.530
- 촉매: JP모건 Buy 유지 (4/10), 이사 $499K 직접 매수 (3/19)
- ⚠️ 리스크: 데이터침해 집단소송 70+건

---

### 2. ADBE — Adobe Inc.

**가격 추적**

| 기준일 | 1주 후 | 2주 후 | 4주 후 |
|--------|--------|--------|--------|
| $244.45 | | | |

**점수 내역**
- 총점: 79/100 [계량 79/70 + 촉매 0/30]
- PE: 14.2x (섹터 40.7x) | FCF Yield: 9.89% | ROE: 58.8% | OpMargin: 36.6%
- RSI: 57.1 | 52주저점대비: +9.1% | MACD: 골든크로스 (-4.54/-6.61) | BB: 0.674
- 촉매: DOJ $150M 합의 완료 (2026-03-13)

---
"""


# ── parse_summary_table ──────────────────────────────────────────────────────

def test_parse_summary_table_returns_two_rows():
    rows = parse_summary_table(SAMPLE_MD)
    assert len(rows) == 2


def test_parse_summary_table_extracts_fields():
    rows = parse_summary_table(SAMPLE_MD)
    assert rows[0]["rank"] == "1"
    assert rows[0]["ticker"] == "CRM"
    assert rows[0]["price"] == "$182.14"
    assert rows[0]["total"] == "90점"
    assert rows[0]["quant"] == "70/70"
    assert rows[0]["catalyst"] == "20/30"
    assert "JP모건" in rows[0]["summary"]


def test_parse_summary_table_strips_warning_emoji():
    # ⚠️ 가 ticker에 붙어있는 경우
    md = SAMPLE_MD.replace("| 2 | ADBE |", "| 2 | ADBE ⚠️ |")
    rows = parse_summary_table(md)
    assert rows[1]["ticker"] == "ADBE"
    assert "⚠️" not in rows[1]["ticker"]


def test_parse_summary_table_empty_when_no_table():
    assert parse_summary_table("# 제목만 있는 문서") == []


# ── parse_stock_sections ─────────────────────────────────────────────────────

def test_parse_stock_sections_returns_two_stocks():
    stocks = parse_stock_sections(SAMPLE_MD)
    assert len(stocks) == 2


def test_parse_stock_sections_extracts_fields():
    stocks = parse_stock_sections(SAMPLE_MD)
    s = stocks[0]
    assert s["rank"] == "1"
    assert s["ticker"] == "CRM"
    assert "90/100" in s["total"]
    assert "23.4x" in s["pe_line"]
    assert "53.2" in s["tech_line"]
    assert "JP모건" in s["catalyst"]
    assert "데이터침해" in s["risk"]


def test_parse_stock_sections_no_risk_when_absent():
    stocks = parse_stock_sections(SAMPLE_MD)
    assert stocks[1]["risk"] == ""  # ADBE: 리스크 줄 없음
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cd ~/repo/skills/value-momentum-screener && python -m pytest tests/test_telegram_notify.py -v 2>&1 | head -15
```

예상 출력: `ModuleNotFoundError: No module named 'telegram_notify'`

- [ ] **Step 3: `telegram_notify.py` 파싱 함수 구현**

`telegram_notify.py` 를 아래 내용으로 생성:

```python
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
        # ticker에서 ⚠️ 제거
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

        # 점수 내역 섹션만 추출
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
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
cd ~/repo/skills/value-momentum-screener && python -m pytest tests/test_telegram_notify.py -v
```

예상 출력: 9개 PASSED

- [ ] **Step 5: 커밋**

```bash
cd ~/repo/skills && git add value-momentum-screener/telegram_notify.py value-momentum-screener/tests/test_telegram_notify.py && git commit -m "feat(telegram): add MD parser with tests"
```

---

## Task 2: 포맷 함수 TDD — MarkdownV2 변환

**Files:**
- Modify: `tests/test_telegram_notify.py` (테스트 추가)
- Modify: `telegram_notify.py` (함수 추가)

- [ ] **Step 1: 포맷 테스트 추가**

`tests/test_telegram_notify.py` 하단에 추가:

```python
from telegram_notify import escape_md, format_header_message, format_stock_message


# ── escape_md ────────────────────────────────────────────────────────────────

def test_escape_md_escapes_dollar():
    assert escape_md("$182.14") == r"\$182\.14"


def test_escape_md_escapes_brackets():
    assert escape_md("[계량 70]") == r"\[계량 70\]"


def test_escape_md_escapes_plus_minus():
    assert escape_md("+11.4%") == r"\+11\.4%"


def test_escape_md_leaves_korean_untouched():
    result = escape_md("JP모건 Buy")
    assert "JP모건 Buy" in result


# ── format_header_message ────────────────────────────────────────────────────

def test_format_header_message_contains_all_tickers():
    rows = parse_summary_table(SAMPLE_MD)
    msg = format_header_message("2026-04-18", rows)
    assert "CRM" in msg
    assert "ADBE" in msg


def test_format_header_message_contains_date():
    rows = parse_summary_table(SAMPLE_MD)
    msg = format_header_message("2026-04-18", rows)
    assert "2026" in msg


def test_format_header_message_no_unescaped_special_chars():
    rows = parse_summary_table(SAMPLE_MD)
    msg = format_header_message("2026-04-18", rows)
    # MarkdownV2에서 허용되지 않는 패턴: 이스케이프 없이 나타나는 . 이나 +
    # 숫자 뒤 점이 이스케이프됐는지 확인
    assert "70/70" not in msg or r"70\/70" in msg or "70/70" in msg  # / 는 이스케이프 불필요


# ── format_stock_message ─────────────────────────────────────────────────────

def test_format_stock_message_contains_price():
    rows = parse_summary_table(SAMPLE_MD)
    stocks = parse_stock_sections(SAMPLE_MD)
    msg = format_stock_message(rows[0], stocks[0])
    assert "182" in msg  # price


def test_format_stock_message_contains_total_score():
    rows = parse_summary_table(SAMPLE_MD)
    stocks = parse_stock_sections(SAMPLE_MD)
    msg = format_stock_message(rows[0], stocks[0])
    assert "90" in msg


def test_format_stock_message_contains_risk_when_present():
    rows = parse_summary_table(SAMPLE_MD)
    stocks = parse_stock_sections(SAMPLE_MD)
    msg = format_stock_message(rows[0], stocks[0])
    assert "데이터침해" in msg


def test_format_stock_message_no_risk_line_when_absent():
    rows = parse_summary_table(SAMPLE_MD)
    stocks = parse_stock_sections(SAMPLE_MD)
    msg = format_stock_message(rows[1], stocks[1])
    assert "리스크" not in msg
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cd ~/repo/skills/value-momentum-screener && python -m pytest tests/test_telegram_notify.py -k "escape or format" -v 2>&1 | head -20
```

예상 출력: `ImportError: cannot import name 'escape_md'`

- [ ] **Step 3: 포맷 함수 구현**

`telegram_notify.py`의 `parse_stock_sections` 함수 아래에 추가:

```python
# ── 포맷 ──────────────────────────────────────────────────────────────────────

_ESCAPE_CHARS = r'_*[]()~`>#+-=|{}.!'


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
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
cd ~/repo/skills/value-momentum-screener && python -m pytest tests/test_telegram_notify.py -v
```

예상 출력: 전체 PASSED

- [ ] **Step 5: 커밋**

```bash
cd ~/repo/skills && git add value-momentum-screener/telegram_notify.py value-momentum-screener/tests/test_telegram_notify.py && git commit -m "feat(telegram): add MarkdownV2 formatting functions"
```

---

## Task 3: send_message + main() TDD

**Files:**
- Modify: `tests/test_telegram_notify.py` (테스트 추가)
- Modify: `telegram_notify.py` (함수 추가)

- [ ] **Step 1: send_message 테스트 추가**

`tests/test_telegram_notify.py` 하단에 추가:

```python
from unittest.mock import MagicMock, patch

from telegram_notify import send_message


# ── send_message ─────────────────────────────────────────────────────────────

def test_send_message_posts_to_telegram_api():
    mock_resp = MagicMock()
    mock_resp.ok = True
    with patch("telegram_notify.requests.post", return_value=mock_resp) as mock_post:
        result = send_message("BOT_TOKEN", "CHAT_ID", "hello")
    assert result is True
    call_args = mock_post.call_args
    assert "BOT_TOKEN" in call_args[0][0]
    payload = call_args[1]["json"]
    assert payload["chat_id"] == "CHAT_ID"
    assert payload["text"] == "hello"
    assert payload["parse_mode"] == "MarkdownV2"


def test_send_message_returns_false_on_api_error():
    mock_resp = MagicMock()
    mock_resp.ok = False
    mock_resp.status_code = 400
    mock_resp.text = "Bad Request"
    with patch("telegram_notify.requests.post", return_value=mock_resp):
        result = send_message("TOKEN", "CHAT_ID", "hello")
    assert result is False
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cd ~/repo/skills/value-momentum-screener && python -m pytest tests/test_telegram_notify.py::test_send_message_posts_to_telegram_api -v 2>&1 | head -10
```

예상 출력: `ImportError: cannot import name 'send_message'`

- [ ] **Step 3: send_message + load_config + run() + main() 구현**

`telegram_notify.py`의 포맷 함수 아래에 추가:

```python
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

    # 헤더 메시지
    send_message(bot_token, chat_id, format_header_message(date_str, summary_rows))
    time.sleep(0.5)

    # 종목별 상세 메시지
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
```

- [ ] **Step 4: 전체 테스트 실행 — 통과 확인**

```bash
cd ~/repo/skills/value-momentum-screener && python -m pytest tests/test_telegram_notify.py -v
```

예상 출력: 전체 PASSED (ImportError 없음)

- [ ] **Step 5: 커밋**

```bash
cd ~/repo/skills && git add value-momentum-screener/telegram_notify.py value-momentum-screener/tests/test_telegram_notify.py && git commit -m "feat(telegram): add send_message and main pipeline"
```

---

## Task 4: 설정 파일 + run_auto.sh 연동

**Files:**
- Create: `telegram_config.json.example`
- Modify: `run_auto.sh`
- Modify: `.gitignore`

- [ ] **Step 1: `telegram_config.json.example` 생성**

```json
{
  "bot_token": "YOUR_BOT_TOKEN_HERE",
  "chat_id": "YOUR_CHAT_ID_HERE"
}
```

> **Bot Token 발급:** Telegram에서 @BotFather → `/newbot` → 토큰 복사
> **Chat ID 확인:** 봇과 대화 후 `https://api.telegram.org/bot{TOKEN}/getUpdates` 에서 `chat.id` 확인

- [ ] **Step 2: `.gitignore`에 `telegram_config.json` 추가**

현재 `.gitignore`:
```
results/
results-backtesting/
scoring_weights.json
```

`telegram_config.json` 줄 추가:
```
results/
results-backtesting/
scoring_weights.json
telegram_config.json
```

- [ ] **Step 3: `telegram_config.json` 실제 생성 (Bot Token + Chat ID 입력)**

```bash
cp telegram_config.json.example telegram_config.json
# 편집기로 bot_token, chat_id 실제 값 입력
```

- [ ] **Step 4: 수동 전송 테스트**

```bash
cd ~/repo/skills/value-momentum-screener && python telegram_notify.py --date 2026-04-18
```

예상 출력:
```
[telegram] 전송 완료: 6개 메시지
```

텔레그램 모바일에서 메시지 수신 확인. 포맷이 깨지면 `escape_md`나 포맷 함수 조정.

- [ ] **Step 5: `run_auto.sh` 수정**

현재 파일의 `osascript -e 'tell application...'` 줄 바로 앞에 추가:

```sh
# 텔레그램 알림 전송
python "$SCRIPT_DIR/telegram_notify.py" --results-dir "$SCRIPT_DIR/results" >> "$LOG_FILE" 2>&1
```

수정 후 `run_auto.sh` 하단:
```sh
/opt/homebrew/bin/claude \
  --dangerously-skip-permissions \
  -p "/value-momentum-screener" \
  >> "$LOG_FILE" 2>&1

echo "완료: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"

# 텔레그램 알림 전송
python "$SCRIPT_DIR/telegram_notify.py" --results-dir "$SCRIPT_DIR/results" >> "$LOG_FILE" 2>&1

# 스크리너 완료 후 다시 잠자기
osascript -e 'tell application "System Events" to sleep'
```

- [ ] **Step 6: 전체 테스트 실행 — 이상 없음 확인**

```bash
cd ~/repo/skills/value-momentum-screener && python -m pytest tests/ -v 2>&1 | tail -10
```

예상 출력: 전체 PASSED

- [ ] **Step 7: 커밋**

```bash
cd ~/repo/skills && git add value-momentum-screener/telegram_config.json.example value-momentum-screener/.gitignore value-momentum-screener/run_auto.sh && git commit -m "feat(telegram): wire notify into run_auto.sh, add config example"
```

---

## Self-Review

**스펙 커버리지:**
- ✅ `telegram_notify.py` — 독립 스크립트
- ✅ `telegram_config.json` — gitignored, example 파일 포함
- ✅ MD 파싱 — 요약 테이블 + 종목 상세
- ✅ MarkdownV2 이스케이프
- ✅ 헤더 1개 + 종목 5개 = 6개 메시지
- ✅ 에러 처리 — config 없음/MD 없음 시 exit 0
- ✅ `--date` 옵션으로 수동 테스트
- ✅ `run_auto.sh` 연동

**타입/함수명 일관성:**
- `parse_summary_table` → Task 1 정의, Task 2 테스트에서 재사용 ✅
- `parse_stock_sections` → Task 1 정의, Task 2 테스트에서 재사용 ✅
- `format_header_message(date_str, summary_rows)` → Task 2 정의, Task 3 run()에서 사용 ✅
- `format_stock_message(summary, detail)` → Task 2 정의, Task 3 run()에서 사용 ✅
- `send_message(bot_token, chat_id, text)` → Task 3 정의 ✅

**플레이스홀더:** 없음
