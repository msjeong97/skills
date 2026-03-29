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
