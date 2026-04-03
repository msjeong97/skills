"""
Tests for run_screener.py

run_screener.py:
- claude CLI를 subprocess로 호출해 value-momentum-screener 스킬을 실행
- 결과를 stdout으로 받아 Telegram으로 전송
"""
import pytest
from unittest.mock import patch, MagicMock
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── run_claude_screener ────────────────────────────────────────────────────────

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
        # claude -p 플래그로 호출해야 한다
        call_args = mock_run.call_args
        cmd = call_args[0][0]
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
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("cwd") == "/my/skill/dir"


# ── send_telegram ──────────────────────────────────────────────────────────────

def test_send_telegram_posts_to_correct_url(monkeypatch):
    """send_telegram이 올바른 Bot API URL로 POST 요청을 보낸다."""
    from run_screener import send_telegram

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_token_123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999888")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        send_telegram("테스트 메시지")
        url = mock_post.call_args[0][0]
        assert "test_token_123" in url
        assert "sendMessage" in url


def test_send_telegram_raises_when_token_missing(monkeypatch):
    """TELEGRAM_BOT_TOKEN 없으면 RuntimeError를 발생시킨다."""
    from run_screener import send_telegram

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")

    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        send_telegram("메시지")


def test_send_telegram_raises_when_chat_id_missing(monkeypatch):
    """TELEGRAM_CHAT_ID 없으면 RuntimeError를 발생시킨다."""
    from run_screener import send_telegram

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    with pytest.raises(RuntimeError, match="TELEGRAM_CHAT_ID"):
        send_telegram("메시지")


def test_send_telegram_returns_false_on_api_error(monkeypatch):
    """Telegram API가 ok:false 반환 시 False를 반환한다."""
    from run_screener import send_telegram

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")

    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {"ok": False, "description": "Bad Request"}

    with patch("requests.post", return_value=mock_resp):
        result = send_telegram("메시지")
        assert result is False


# ── truncate_for_telegram ──────────────────────────────────────────────────────

def test_truncate_for_telegram_leaves_short_message_intact():
    """4096자 이하 메시지는 그대로 반환한다."""
    from run_screener import truncate_for_telegram

    msg = "짧은 메시지"
    assert truncate_for_telegram(msg) == msg


def test_truncate_for_telegram_cuts_long_message():
    """4096자 초과 메시지를 4096자 이하로 자른다."""
    from run_screener import truncate_for_telegram

    long_msg = "A" * 5000
    result = truncate_for_telegram(long_msg)
    assert len(result) <= 4096


def test_truncate_for_telegram_appends_notice():
    """자른 경우 생략 안내 문구를 추가한다."""
    from run_screener import truncate_for_telegram

    long_msg = "B" * 5000
    result = truncate_for_telegram(long_msg)
    assert "생략" in result or "truncated" in result.lower()
