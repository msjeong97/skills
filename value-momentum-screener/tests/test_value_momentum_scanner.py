"""Tests for --json-only flag in value_momentum_scanner.py"""
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_parse_args_json_only_flag():
    """--json-only 플래그가 있으면 json_only=True를 반환한다."""
    from value_momentum_scanner import parse_args

    args = parse_args(["--json-only"])
    assert args.json_only is True


def test_parse_args_default_no_json_only():
    """플래그 없이 실행하면 json_only=False를 반환한다."""
    from value_momentum_scanner import parse_args

    args = parse_args([])
    assert args.json_only is False


def _make_minimal_scored():
    """테스트용 최소 scored 데이터."""
    return [{
        'ticker': 'AAPL',
        'name': 'Apple Inc.',
        'sector': 'Technology',
        'current_price': 200.0,
        '52w_low': 160.0,
        '52w_high': 250.0,
        'undervalue_score': 55.0,
        'breakdown': {'valuation': 15.0, 'profitability': 10.0, 'technical': 30.0},
        'detail': {
            'valuation': {'pe': {'value': 28, 'sector_median': 30, 'score': 5},
                          'fcf_yield': {'value': 3.5, 'score': 6},
                          'peg': {'value': 1.2, 'score': 3}},
            'profitability': {'roe': {'value': 20, 'score': 10},
                              'op_margin': {'value': 28.0, 'score': 0}},
            'technical': {'rsi': {'value': 42, 'score': 12},
                          '52w_low': {'pct_from_low': 12.0, 'score': 15},
                          'macd': {'golden_cross': True, 'macd_last': 0.1,
                                   'signal_last': 0.05, 'score': 12},
                          'bollinger': {'pct_b': 0.35, 'score': 8},
                          'volume': {'score': 8}},
        },
    }]


def test_main_json_only_skips_output_results(tmp_path, monkeypatch):
    """--json-only 실행 시 output_results()가 호출되지 않는다."""
    import value_momentum_scanner as vms

    scored = _make_minimal_scored()
    monkeypatch.setattr(vms, "collect_all_data", lambda tickers: [MagicMock()])
    monkeypatch.setattr(vms, "apply_qualification_filter", lambda data: [MagicMock()])
    monkeypatch.setattr(vms, "compute_sector_stats", lambda q: {})
    monkeypatch.setattr(vms, "compute_scores", lambda q, s: scored)
    monkeypatch.setattr(vms, "save_top10_json", lambda top10, skill_dir: str(tmp_path / "out.json"))

    with patch.object(vms, "output_results") as mock_output:
        with patch("sys.argv", ["value_momentum_scanner.py", "--json-only"]):
            vms.main()
        mock_output.assert_not_called()


def test_main_without_flag_calls_output_results(tmp_path, monkeypatch):
    """플래그 없이 실행 시 output_results()가 호출된다."""
    import value_momentum_scanner as vms

    scored = _make_minimal_scored()
    monkeypatch.setattr(vms, "collect_all_data", lambda tickers: [MagicMock()])
    monkeypatch.setattr(vms, "apply_qualification_filter", lambda data: [MagicMock()])
    monkeypatch.setattr(vms, "compute_sector_stats", lambda q: {})
    monkeypatch.setattr(vms, "compute_scores", lambda q, s: scored)
    monkeypatch.setattr(vms, "save_top10_json", lambda top10, skill_dir: str(tmp_path / "out.json"))

    with patch.object(vms, "output_results") as mock_output:
        with patch("sys.argv", ["value_momentum_scanner.py"]):
            vms.main()
        mock_output.assert_called_once()
