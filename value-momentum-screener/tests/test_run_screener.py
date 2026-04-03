"""Tests for run_screener.py"""
import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_run_claude_screener_returns_output_on_success():
    """claude CLI 성공 시 stdout 문자열을 반환한다."""
    from run_screener import run_claude_screener

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "📊 Top 5 결과\n1. AAPL ..."
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        output = run_claude_screener("/some/skill/dir")
        assert "AAPL" in output
        cmd = mock_run.call_args[0][0]
        assert "claude" in cmd[0]
        assert "-p" in cmd or "--print" in cmd


def test_run_claude_screener_raises_on_nonzero_exit():
    """claude CLI가 0이 아닌 종료 코드 반환 시 RuntimeError를 발생시킨다."""
    from run_screener import run_claude_screener

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "Error: skill not found"

    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(RuntimeError, match="claude"):
            run_claude_screener("/some/skill/dir")


def test_run_claude_screener_uses_skill_dir_as_cwd():
    """claude CLI를 skill_dir을 cwd로 설정해 호출한다."""
    from run_screener import run_claude_screener

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "결과"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        run_claude_screener("/my/skill/dir")
        assert mock_run.call_args[1].get("cwd") == "/my/skill/dir"
