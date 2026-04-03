"""
run_screener.py

Claude Code CLI를 subprocess로 호출해 value-momentum-screener 스킬을 실행하고,
결과를 Telegram으로 전송한다.

Usage:
    python run_screener.py                    # 실행 후 Telegram 전송
    python run_screener.py --no-telegram      # 실행만, Telegram 전송 안 함

Environment variables:
    TELEGRAM_BOT_TOKEN   — Telegram Bot API 토큰 (@BotFather에서 발급)
    TELEGRAM_CHAT_ID     — 전송 대상 Chat ID
"""

import os
import subprocess
import sys
import requests

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
MAX_TELEGRAM_LENGTH = 4096


def run_claude_screener(skill_dir: str) -> str:
    """
    claude CLI로 /value-momentum-screener 스킬을 비대화형으로 실행한다.

    Args:
        skill_dir: value-momentum-screener 루트 경로 (cwd로 사용)

    Returns:
        Claude의 stdout 출력 문자열

    Raises:
        RuntimeError: claude CLI가 비정상 종료된 경우
    """
    result = subprocess.run(
        ["claude", "-p", "/value-momentum-screener"],
        capture_output=True,
        text=True,
        cwd=skill_dir,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI 실행 실패 (exit {result.returncode})\n"
            f"stderr: {result.stderr}"
        )

    return result.stdout


def truncate_for_telegram(message: str) -> str:
    """Telegram 메시지 길이 제한(4096자)에 맞게 자른다."""
    if len(message) <= MAX_TELEGRAM_LENGTH:
        return message
    notice = "\n\n...(내용이 길어 일부 생략됨)"
    cutoff = MAX_TELEGRAM_LENGTH - len(notice)
    return message[:cutoff] + notice


def send_telegram(message: str) -> bool:
    """
    Telegram Bot API로 메시지를 전송한다.

    Returns:
        True if successful, False otherwise.

    Raises:
        RuntimeError: 환경변수 미설정 시
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN 환경변수가 없습니다. "
            "~/.zshrc 에 export TELEGRAM_BOT_TOKEN='...' 을 추가하세요."
        )
    if not chat_id:
        raise RuntimeError(
            "TELEGRAM_CHAT_ID 환경변수가 없습니다. "
            "~/.zshrc 에 export TELEGRAM_CHAT_ID='...' 을 추가하세요."
        )

    url = TELEGRAM_API_URL.format(token=token)
    payload = {
        "chat_id": chat_id,
        "text": truncate_for_telegram(message),
        "parse_mode": "HTML",
    }

    response = requests.post(url, json=payload, timeout=15)
    data = response.json()

    if not data.get("ok"):
        print(f"❌ Telegram 전송 실패: {data.get('description', 'Unknown error')}")
        return False

    return True


def main():
    no_telegram = "--no-telegram" in sys.argv
    skill_dir = os.path.dirname(os.path.abspath(__file__))

    print("🔍 value-momentum-screener 실행 중...")
    try:
        output = run_claude_screener(skill_dir)
    except RuntimeError as e:
        print(f"❌ 실행 실패:\n{e}")
        sys.exit(1)

    print(output)

    if no_telegram:
        print("ℹ️  --no-telegram 플래그: Telegram 전송 건너뜀")
        return

    if not os.environ.get("TELEGRAM_BOT_TOKEN"):
        print("ℹ️  TELEGRAM_BOT_TOKEN 미설정: Telegram 전송 건너뜀")
        return

    print("\n📨 Telegram 전송 중...")
    success = send_telegram(output)
    if success:
        print("✅ Telegram 전송 완료!")


if __name__ == "__main__":
    main()
