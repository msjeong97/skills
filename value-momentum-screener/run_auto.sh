#!/bin/zsh
# Value Momentum Screener — 자동 실행 스크립트 (cron용)

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH"
export CLAUDE_CODE_OAUTH_TOKEN=$(grep -E '^export CLAUDE_CODE_OAUTH_TOKEN=' ~/.zshrc | cut -d'"' -f2)
eval "$(pyenv init --path 2>/dev/null)"
eval "$(pyenv init - 2>/dev/null)"

SCRIPT_DIR="/Users/minseop/repo/skills/value-momentum-screener"
LOG_FILE="$SCRIPT_DIR/results/auto_run.log"

echo "========================================" >> "$LOG_FILE"
echo "실행 시각: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

cd "$SCRIPT_DIR"

/opt/homebrew/bin/claude \
  --dangerously-skip-permissions \
  -p "/value-momentum-screener" \
  >> "$LOG_FILE" 2>&1

echo "완료: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"

# 텔레그램 알림 전송
python "$SCRIPT_DIR/telegram_notify.py" --results-dir "$SCRIPT_DIR/results" >> "$LOG_FILE" 2>&1

# 스크리너 완료 후 다시 잠자기
osascript -e 'tell application "System Events" to sleep'
