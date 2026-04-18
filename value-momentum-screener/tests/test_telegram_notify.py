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


from telegram_notify import escape_md, format_header_message, format_stock_message


# ── escape_md ────────────────────────────────────────────────────────────────

def test_escape_md_escapes_dollar():
    assert escape_md("$182.14") == r"\$182\.14"


def test_escape_md_escapes_brackets():
    assert escape_md("[계량 70]") == r"\[계량 70\]"


def test_escape_md_escapes_plus():
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


# ── format_stock_message ─────────────────────────────────────────────────────

def test_format_stock_message_contains_price():
    rows = parse_summary_table(SAMPLE_MD)
    stocks = parse_stock_sections(SAMPLE_MD)
    msg = format_stock_message(rows[0], stocks[0])
    assert "182" in msg


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
