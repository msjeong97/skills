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

# 네트워크 대기 (sleep 후 깨어날 때 연결이 늦을 수 있음)
for i in {1..10}; do
  ping -c1 -W2 8.8.8.8 &>/dev/null && break
  echo "[network] 대기 중... (${i}/10)" >> "$LOG_FILE"
  sleep 5
done

# claude 실행 및 출력 임시 캡처
CLAUDE_OUTPUT=$(/opt/homebrew/bin/claude \
  --dangerously-skip-permissions \
  -p "/value-momentum-screener" \
  2>&1)
CLAUDE_EXIT=$?

echo "$CLAUDE_OUTPUT" >> "$LOG_FILE"
echo "[claude exit: $CLAUDE_EXIT, output_len: ${#CLAUDE_OUTPUT}]" >> "$LOG_FILE"

# 출력이 없거나 rate limit 메시지 감지
if [[ -z "$CLAUDE_OUTPUT" ]]; then
  echo "❌ ERROR: claude 출력 없음 (rate limit 또는 인증 실패 가능성)" >> "$LOG_FILE"
  echo "완료(실패): $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
  osascript -e 'tell application "System Events" to sleep'
  exit 1
fi

if echo "$CLAUDE_OUTPUT" | grep -q "hit your limit\|Not logged in"; then
  echo "❌ ERROR: $(echo "$CLAUDE_OUTPUT" | head -1)" >> "$LOG_FILE"
  echo "완료(실패): $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
  osascript -e 'tell application "System Events" to sleep'
  exit 1
fi

echo "완료: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"

# 텔레그램 알림 전송
python "$SCRIPT_DIR/telegram_notify.py" --results-dir "$SCRIPT_DIR/results" >> "$LOG_FILE" 2>&1

# 스크리너 완료 후 다시 잠자기
osascript -e 'tell application "System Events" to sleep'
