# Telegram Notification Design — value-momentum-screener

**Date:** 2026-04-18
**Scope:** 평일 자동 실행 후 스크리너 결과를 텔레그램으로 전송

---

## 목적

`run_auto.sh`가 매일 실행하는 `/value-momentum-screener` 결과를 텔레그램 모바일에서 가독성 좋게 받아보기.

---

## 아키텍처

```
run_auto.sh
  1. claude -p "/value-momentum-screener"   → results/YYYY-MM-DD.md 생성
  2. python telegram_notify.py              → 텔레그램 전송 (6개 메시지)
```

`telegram_notify.py`는 스캐너와 완전히 독립된 스크립트. MD 파일 파싱 → 텔레그램 MarkdownV2 변환 → Bot API 전송만 담당.

---

## 파일 구성

| 파일 | 역할 |
|------|------|
| `telegram_notify.py` | 알림 스크립트 (프로젝트 루트) |
| `telegram_config.json` | Bot Token + Chat ID 설정 |
| `.gitignore` | `telegram_config.json` 추가 |
| `run_auto.sh` | 마지막에 `python telegram_notify.py` 한 줄 추가 |

---

## 설정 파일 (`telegram_config.json`)

```json
{
  "bot_token": "1234567890:AAXXXXXX",
  "chat_id": "123456789"
}
```

- `.gitignore`에 추가해 토큰이 커밋되지 않도록 함
- 파일 없으면 알림 전송을 건너뛰고 경고 로그만 출력 (스캐너 실행은 영향 없음)

---

## 메시지 구조 (총 6개)

### 메시지 1 — 헤더 요약

```
📊 *Value Momentum Screener*
YYYY\-MM\-DD | 유니버스 150종목

🏆 Top 5
1\. {TICKER} — {총점}점 \[계량 {quant} \+ 촉매 {catalyst}/30\]
2\. ...
```

### 메시지 2~6 — 종목별 상세

```
*{순위}\. {TICKER} — {회사명}*

💰 현재가: \${price}
📊 총점: {total}/130 \[계량 {quant} \+ 촉매 {catalyst}/30\]

PE: {pe}x \(섹터 {sector_pe}x\) | FCF: {fcf}%
RSI: {rsi} | 52주저점\+{pct}% | MACD {macd_status}

💬 {한줄요약}

🎯 매수: \${buy_low}\~\${buy_high}
🛑 손절: \${stop_loss}

{⚠️ 리스크 있으면 표시, 없으면 생략}
```

---

## 파싱 전략

`results/YYYY-MM-DD.md` 구조에서 추출:

- **헤더 요약 테이블** (`## Top 5 추천 종목` 아래 마크다운 테이블): 종목명, 현재가, 총점, 계량, 촉매, 한줄요약
- **종목별 상세** (`### N. TICKER` 섹션): 점수 내역 bullet들, 매수 포인트, 손절, 리스크

텔레그램 MarkdownV2 이스케이프 필수 문자: `_ * [ ] ( ) ~ > # + - = | { } . !`

---

## 에러 처리

| 상황 | 처리 |
|------|------|
| `telegram_config.json` 없음 | 경고 로그 출력 후 종료 (exit 0) |
| 오늘 날짜 MD 파일 없음 | 경고 로그 출력 후 종료 (exit 0) |
| Telegram API 오류 | 에러 로그 출력, 재시도 없음 |
| 파싱 실패 (종목 데이터 없음) | 해당 종목 건너뜀, 나머지 전송 |

스캐너 자체 실행에는 영향 없도록 모든 예외를 조용히 처리.

---

## run_auto.sh 변경

기존 claude 실행 라인 뒤에 한 줄 추가:

```sh
python "$SCRIPT_DIR/telegram_notify.py" --results-dir "$SCRIPT_DIR/results" >> "$LOG_FILE" 2>&1
```

---

## 의존성

- `requests` (pip) — Telegram Bot API HTTP 호출용
- Python 표준 라이브러리: `re`, `json`, `pathlib`, `datetime`

---

## 테스트 방법

```bash
# 설정 파일 준비 후
python telegram_notify.py --date 2026-04-18
```

`--date` 옵션으로 특정 날짜 MD 파일 지정해서 수동 테스트 가능.
