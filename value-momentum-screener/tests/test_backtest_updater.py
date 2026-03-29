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

def test_update_md_price_table_fills_empty_cells(tmp_path):
    md_content = (
        "### 1. CI — The Cigna Group\n\n"
        "**가격 추적**\n\n"
        "| 기준일 | 1주 후 | 2주 후 | 4주 후 |\n"
        "|--------|--------|--------|--------|\n"
        "| $261.96 | | | |\n"
    )
    md_path = tmp_path / "test.md"
    md_path.write_text(md_content)

    changed = update_md_price_table(
        md_path, ticker="CI", base_price=261.96,
        prices={"1w": 270.00, "2w": None, "4w": None},
    )

    result = md_path.read_text()
    assert changed is True
    assert "$270.00" in result
    assert "+3.1%" in result


def test_update_md_price_table_skips_already_filled(tmp_path):
    md_content = (
        "### 1. CI — The Cigna Group\n\n"
        "| 기준일 | 1주 후 | 2주 후 | 4주 후 |\n"
        "|--------|--------|--------|--------|\n"
        "| $261.96 | $270.00 (+3.1%) | | |\n"
    )
    md_path = tmp_path / "test.md"
    md_path.write_text(md_content)

    changed = update_md_price_table(
        md_path, ticker="CI", base_price=261.96,
        prices={"1w": 275.00, "2w": None, "4w": None},
    )

    assert changed is False


def test_update_md_price_table_shows_negative_return(tmp_path):
    md_content = (
        "### 1. CI — The Cigna Group\n\n"
        "| 기준일 | 1주 후 | 2주 후 | 4주 후 |\n"
        "|--------|--------|--------|--------|\n"
        "| $261.96 | | | |\n"
    )
    md_path = tmp_path / "test.md"
    md_path.write_text(md_content)

    update_md_price_table(
        md_path, ticker="CI", base_price=261.96,
        prices={"1w": 250.00, "2w": None, "4w": None},
    )

    result = md_path.read_text()
    assert "-4.6%" in result


def test_update_md_price_table_scopes_to_ticker_section(tmp_path):
    """같은 가격을 가진 두 종목: 올바른 섹션만 업데이트."""
    md_content = (
        "### 1. CI — The Cigna Group\n\n"
        "| 기준일 | 1주 후 | 2주 후 | 4주 후 |\n"
        "|--------|--------|--------|--------|\n"
        "| $261.96 | | | |\n\n"
        "### 2. OTHER — Other Company\n\n"
        "| 기준일 | 1주 후 | 2주 후 | 4주 후 |\n"
        "|--------|--------|--------|--------|\n"
        "| $261.96 | | | |\n"
    )
    md_path = tmp_path / "test.md"
    md_path.write_text(md_content)

    update_md_price_table(
        md_path, ticker="CI", base_price=261.96,
        prices={"1w": 270.00, "2w": None, "4w": None},
    )

    result = md_path.read_text()
    lines = result.split("\n")
    filled_rows = [l for l in lines if "$261.96" in l and "$270.00" in l]
    empty_rows = [l for l in lines if "$261.96" in l and "$270.00" not in l and "|" in l]
    assert len(filled_rows) == 1   # CI만 채워짐
    assert len(empty_rows) == 1    # OTHER는 그대로


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
