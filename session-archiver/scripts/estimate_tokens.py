#!/usr/bin/env python3
"""Estimate token count from stdin or file and report context window usage."""

import sys

CONTEXT_WINDOW = 200_000


def estimate_tokens(text: str) -> int:
    """Estimate token count using character-based heuristic.

    English: ~4 chars/token, Korean/CJK: ~2 chars/token.
    We scan for CJK characters and weight accordingly.
    """
    cjk_chars = 0
    ascii_chars = 0
    for ch in text:
        cp = ord(ch)
        if (
            0xAC00 <= cp <= 0xD7AF  # Korean Syllables
            or 0x3000 <= cp <= 0x9FFF  # CJK Unified + Japanese
            or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility
        ):
            cjk_chars += 1
        else:
            ascii_chars += 1

    tokens = (ascii_chars / 4) + (cjk_chars / 2)
    return int(tokens)


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    tokens = estimate_tokens(text)
    usage_pct = (tokens / CONTEXT_WINDOW) * 100

    print(f"Estimated tokens: {tokens:,}")
    print(f"Context window: {CONTEXT_WINDOW:,}")
    print(f"Usage: {usage_pct:.1f}%")

    if usage_pct >= 80:
        print("STATUS: HIGH")
    elif usage_pct >= 60:
        print("STATUS: MODERATE")
    else:
        print("STATUS: OK")


if __name__ == "__main__":
    main()
