"""
run_screener.py

Claude Code CLI를 subprocess로 호출해 value-momentum-screener 스킬을 실행하고
결과를 stdout으로 출력한다. Telegram 전송은 OpenClaw가 처리한다.

Usage:
    python run_screener.py
"""

import os
import subprocess
import sys


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


def main():
    skill_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        output = run_claude_screener(skill_dir)
    except RuntimeError as e:
        print(f"❌ 실행 실패:\n{e}", file=sys.stderr)
        sys.exit(1)

    print(output)


if __name__ == "__main__":
    main()
