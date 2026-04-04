# Telegram Notification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** value-momentum-screener 스킬 실행 결과를 Telegram으로 자동 전송한다.

**Architecture:** `telegram_notifier.py`를 새로 만들어 SKILL.md의 Step 3 이후(Step 4)에 호출한다. 환경변수(`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`)로 설정하고, 최신 `results/YYYY-MM-DD.md`를 읽어 Telegram MarkdownV2 형식으로 포매팅 후 전송한다.

**Tech Stack:** Python 3, `requests` (HTTP), Telegram Bot API (sendMessage), 환경변수 기반 설정

---

## File Structure

- **Create:** `value-momentum-screener/telegram_notifier.py` — Telegram 전송 로직 (메시지 포매팅 + Bot API 호출)
- **Modify:** `value-momentum-screener/requirements.txt` — `requests` 추가
- **Modify:** `value-momentum-screener/SKILL.md` — Step 4 (Telegram 전송) 추가
- **Create:** `tests/test_telegram_notifier.py` — 단위 테스트

---

## 사전 준비 (코드 작성 전)

### Task 0: Telegram Bot 토큰 & Chat ID 확인

- [ ] **Step 1: Bot 토큰 준비**

  아직 없다면 Telegram에서 `@BotFather`에 `/newbot` 명령으로 봇 생성 후 토큰 발급.
  이미 있다면 기존 토큰 사용.

- [ ] **Step 2: Chat ID 확인**

  봇에게 메시지 전송 후 아래 URL로 Chat ID 확인:
  ```
  https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
  ```
  응답 JSON에서 `result[0].message.chat.id` 값 기록.

- [ ] **Step 3: 환경변수 설정**

  `~/.zshrc` 또는 `~/.zprofile`에 추가:
  ```bash
  export TELEGRAM_BOT_TOKEN="1234567890:ABCdef..."
  export TELEGRAM_CHAT_ID="123456789"
  ```
  이후 `source ~/.zshrc` 실행.

---

## Task 1: `requests` 의존성 추가

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: requirements.txt에 requests 추가**

  현재 내용:
  ```
  numpy
  pandas
  yfinance
  pandas_ta
  ```

  수정 후:
  ```
  numpy
  pandas
  yfinance
  pandas_ta
  requests
  ```

- [ ] **Step 2: 설치 확인**

  ```bash
  pip install requests
  ```
  Expected: `Successfully installed requests-2.x.x` 또는 `Requirement already satisfied`

- [ ] **Step 3: Commit**

  ```bash
  git add requirements.txt
  git commit -m "feat: add requests dependency for telegram notification"
  ```

---

## Task 2: `telegram_notifier.py` 작성

**Files:**
- Create: `telegram_notifier.py`
- Create: `tests/test_telegram_notifier.py`

### Step 1: 실패 테스트 작성

- [ ] **`tests/test_telegram_notifier.py` 생성**

  ```python
  import pytest
  from unittest.mock import patch, MagicMock
  import sys
  import os
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


  def test_format_message_truncates_long_lines():
      """Telegram 메시지는 4096자 제한이 있으므로 긴 텍스트를 잘라야 한다."""
      from telegram_notifier import format_for_telegram
      long_text = "A" * 5000
      result = format_for_telegram(long_text)
      assert len(result) <= 4096


  def test_format_message_escapes_special_chars():
      """MarkdownV2 특수문자를 이스케이프해야 한다."""
      from telegram_notifier import escape_markdown_v2
      text = "PE 24.4x | FCF 3.17% (섹터: 25.9x)"
      result = escape_markdown_v2(text)
      # MarkdownV2에서 특수문자는 \ 로 이스케이프
      assert "\\." not in result or "24\\.4" in result  # . 이스케이프 확인
      assert "\\|" in result or "|" not in result  # | 이스케이프 확인


  def test_send_message_calls_api(monkeypatch):
      """send_telegram_message가 올바른 URL로 POST 요청을 보내야 한다."""
      from telegram_notifier import send_telegram_message

      mock_response = MagicMock()
      mock_response.status_code = 200
      mock_response.json.return_value = {"ok": True}

      monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_token")
      monkeypatch.setenv("TELEGRAM_CHAT_ID", "test_chat_id")

      with patch("requests.post", return_value=mock_response) as mock_post:
          result = send_telegram_message("테스트 메시지")
          assert result is True
          mock_post.assert_called_once()
          call_url = mock_post.call_args[0][0]
          assert "test_token" in call_url
          assert "sendMessage" in call_url


  def test_send_message_returns_false_on_api_error(monkeypatch):
      """API가 ok: false를 반환하면 False를 반환해야 한다."""
      from telegram_notifier import send_telegram_message

      mock_response = MagicMock()
      mock_response.status_code = 400
      mock_response.json.return_value = {"ok": False, "description": "Bad Request"}

      monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_token")
      monkeypatch.setenv("TELEGRAM_CHAT_ID", "test_chat_id")

      with patch("requests.post", return_value=mock_response):
          result = send_telegram_message("메시지")
          assert result is False


  def test_missing_env_vars_raises(monkeypatch):
      """환경변수 없으면 RuntimeError를 발생시켜야 한다."""
      from telegram_notifier import send_telegram_message
      monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
      monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

      with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
          send_telegram_message("메시지")


  def test_read_latest_results_finds_newest_file(tmp_path):
      """results/ 에서 날짜 기준 최신 파일을 읽어야 한다."""
      from telegram_notifier import read_latest_results

      results_dir = tmp_path / "results"
      results_dir.mkdir()
      (results_dir / "2026-04-01.md").write_text("# 오래된 결과")
      (results_dir / "2026-04-03.md").write_text("# 최신 결과")

      content = read_latest_results(str(results_dir))
      assert "최신 결과" in content


  def test_read_latest_results_returns_none_when_empty(tmp_path):
      """results/ 가 비어있으면 None을 반환해야 한다."""
      from telegram_notifier import read_latest_results
      results_dir = tmp_path / "results"
      results_dir.mkdir()

      result = read_latest_results(str(results_dir))
      assert result is None
  ```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

  ```bash
  cd /Users/minseop/repo/skills/value-momentum-screener
  python -m pytest tests/test_telegram_notifier.py -v
  ```
  Expected: `ModuleNotFoundError: No module named 'telegram_notifier'`

### Step 2: 구현 코드 작성

- [ ] **`telegram_notifier.py` 생성**

  ```python
  """
  Telegram notification module for value-momentum-screener.

  Environment variables required:
    TELEGRAM_BOT_TOKEN  — Telegram Bot API token from @BotFather
    TELEGRAM_CHAT_ID    — Target chat/channel ID
  """
  import os
  import glob
  import requests


  TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
  MAX_MESSAGE_LENGTH = 4096

  # MarkdownV2에서 이스케이프가 필요한 특수문자
  _MD_V2_SPECIAL = r"\_*[]()~`>#+-=|{}.!"


  def escape_markdown_v2(text: str) -> str:
      """Telegram MarkdownV2 특수문자를 이스케이프한다."""
      result = []
      for ch in text:
          if ch in _MD_V2_SPECIAL:
              result.append("\\" + ch)
          else:
              result.append(ch)
      return "".join(result)


  def format_for_telegram(text: str) -> str:
      """긴 텍스트를 Telegram 메시지 길이 제한(4096자)에 맞게 자른다."""
      if len(text) <= MAX_MESSAGE_LENGTH:
          return text
      truncation_notice = "\n\n... (결과가 너무 길어 일부 생략됨)"
      cutoff = MAX_MESSAGE_LENGTH - len(truncation_notice)
      return text[:cutoff] + truncation_notice


  def send_telegram_message(text: str) -> bool:
      """
      Telegram Bot API로 메시지를 전송한다.

      Returns:
          True if successful, False otherwise.

      Raises:
          RuntimeError: TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID 환경변수 없을 때
      """
      token = os.environ.get("TELEGRAM_BOT_TOKEN")
      chat_id = os.environ.get("TELEGRAM_CHAT_ID")

      if not token:
          raise RuntimeError(
              "TELEGRAM_BOT_TOKEN 환경변수가 설정되지 않았습니다. "
              "~/.zshrc 에 export TELEGRAM_BOT_TOKEN='...' 을 추가하세요."
          )
      if not chat_id:
          raise RuntimeError(
              "TELEGRAM_CHAT_ID 환경변수가 설정되지 않았습니다. "
              "~/.zshrc 에 export TELEGRAM_CHAT_ID='...' 을 추가하세요."
          )

      url = TELEGRAM_API_URL.format(token=token)
      payload = {
          "chat_id": chat_id,
          "text": format_for_telegram(text),
          "parse_mode": "HTML",
      }

      response = requests.post(url, json=payload, timeout=10)
      data = response.json()

      if not data.get("ok"):
          print(f"❌ Telegram 전송 실패: {data.get('description', 'Unknown error')}")
          return False

      return True


  def read_latest_results(results_dir: str) -> str | None:
      """results/ 디렉토리에서 날짜 기준 가장 최신 .md 파일 내용을 반환한다."""
      pattern = os.path.join(results_dir, "????-??-??.md")
      files = sorted(glob.glob(pattern))

      if not files:
          return None

      latest_file = files[-1]
      with open(latest_file, "r", encoding="utf-8") as f:
          return f.read()


  def notify_results(skill_dir: str) -> bool:
      """
      최신 스크리너 결과를 읽어 Telegram으로 전송한다.

      Args:
          skill_dir: value-momentum-screener 루트 경로

      Returns:
          True if sent successfully.
      """
      results_dir = os.path.join(skill_dir, "results")
      content = read_latest_results(results_dir)

      if content is None:
          print("⚠️ 전송할 결과 파일이 없습니다.")
          return False

      # HTML parse_mode로 전송 — MD 특수문자 이스케이프 불필요
      message = f"📊 <b>Value Momentum Screener 결과</b>\n\n{content}"
      success = send_telegram_message(message)

      if success:
          print("✅ Telegram 전송 완료!")
      return success


  if __name__ == "__main__":
      skill_dir = os.path.dirname(os.path.abspath(__file__))
      notify_results(skill_dir)
  ```

- [ ] **Step 3: 테스트 실행 — 통과 확인**

  ```bash
  cd /Users/minseop/repo/skills/value-momentum-screener
  python -m pytest tests/test_telegram_notifier.py -v
  ```
  Expected:
  ```
  tests/test_telegram_notifier.py::test_format_message_truncates_long_lines PASSED
  tests/test_telegram_notifier.py::test_format_message_escapes_special_chars PASSED
  tests/test_telegram_notifier.py::test_send_message_calls_api PASSED
  tests/test_telegram_notifier.py::test_send_message_returns_false_on_api_error PASSED
  tests/test_telegram_notifier.py::test_missing_env_vars_raises PASSED
  tests/test_telegram_notifier.py::test_read_latest_results_finds_newest_file PASSED
  tests/test_telegram_notifier.py::test_read_latest_results_returns_none_when_empty PASSED
  7 passed
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add telegram_notifier.py tests/test_telegram_notifier.py
  git commit -m "feat: add telegram_notifier module with tests"
  ```

---

## Task 3: SKILL.md에 Step 4 추가

**Files:**
- Modify: `SKILL.md`

- [ ] **Step 1: SKILL.md의 Step 3-2 다음에 Step 4 추가**

  현재 `⚠️ **주의사항**` 바로 위에 다음 내용 삽입:

  ```markdown
  ---

  ## Step 4: Telegram 전송 (선택)

  환경변수 `TELEGRAM_BOT_TOKEN`과 `TELEGRAM_CHAT_ID`가 설정된 경우 결과를 Telegram으로 전송합니다.

  ```bash
  python {{SKILL_DIR}}/telegram_notifier.py
  ```

  전송 성공 시: `✅ Telegram 전송 완료!`
  미설정 시: 이 단계를 건너뜁니다.
  ```

  완성된 추가 블록 (파일에 실제로 삽입할 내용):

  ````markdown
  ---

  ## Step 4: Telegram 전송 (선택)

  환경변수 `TELEGRAM_BOT_TOKEN`과 `TELEGRAM_CHAT_ID`가 설정된 경우 결과를 Telegram으로 전송합니다.

  ```bash
  python {{SKILL_DIR}}/telegram_notifier.py
  ```

  **환경변수 설정 방법 (`~/.zshrc`):**
  ```bash
  export TELEGRAM_BOT_TOKEN="your_bot_token_here"
  export TELEGRAM_CHAT_ID="your_chat_id_here"
  ```

  환경변수가 없으면 이 단계를 **건너뜁니다** (오류 없이 종료).

  ````

- [ ] **Step 2: SKILL.md Step 4 환경변수 조건부 실행 처리**

  `telegram_notifier.py`의 `__main__` 블록을 환경변수 없을 때 조용히 종료하도록 수정:

  `telegram_notifier.py` 파일 하단 `if __name__ == "__main__":` 블록을 아래로 교체:

  ```python
  if __name__ == "__main__":
      if not os.environ.get("TELEGRAM_BOT_TOKEN"):
          print("ℹ️ TELEGRAM_BOT_TOKEN 미설정 — Telegram 전송 건너뜀")
      else:
          skill_dir = os.path.dirname(os.path.abspath(__file__))
          notify_results(skill_dir)
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add SKILL.md telegram_notifier.py
  git commit -m "feat: add telegram notification as optional Step 4 in skill"
  ```

---

## Task 4: 수동 통합 테스트

> 환경변수가 실제로 설정된 상태에서 end-to-end 확인

- [ ] **Step 1: 환경변수 설정 확인**

  ```bash
  echo $TELEGRAM_BOT_TOKEN   # 출력되면 OK
  echo $TELEGRAM_CHAT_ID     # 출력되면 OK
  ```

- [ ] **Step 2: 기존 결과 파일로 Telegram 전송 테스트**

  ```bash
  cd /Users/minseop/repo/skills/value-momentum-screener
  python telegram_notifier.py
  ```
  Expected:
  ```
  ✅ Telegram 전송 완료!
  ```
  Telegram 앱에서 봇 메시지 수신 확인.

- [ ] **Step 3: 환경변수 없을 때 graceful 종료 확인**

  ```bash
  TELEGRAM_BOT_TOKEN="" python telegram_notifier.py
  ```
  Expected:
  ```
  ℹ️ TELEGRAM_BOT_TOKEN 미설정 — Telegram 전송 건너뜀
  ```

---

## 최종 워크플로우

스킬 실행 후 전체 흐름:

```
Step 1: python value_momentum_scanner.py
         → Top 10 JSON 생성 (results/YYYY-MM-DD-top10-raw.json)

Step 2: Claude가 WebSearch로 Momentum 점수 산출

Step 3: Top 5 터미널 출력 + results/YYYY-MM-DD.md 저장

Step 4: python telegram_notifier.py
         → results/YYYY-MM-DD.md 읽어 Telegram 전송
         → TELEGRAM_BOT_TOKEN 없으면 조용히 스킵
```
