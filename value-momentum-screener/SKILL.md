---
name: value-momentum-screener
description: >
  미국 시총 상위 150개 우량주 중 저평가 지수 + 상승 신호 점수가 높은 Top 5 종목 추천.
  매수 참고용 단기(1~4주) 스크리너. 트리거: "저평가 주식 추천해줘", "오를 것 같은 주식 찾아줘",
  "value momentum screener", "/value-momentum-screener", "저평가 우량주"
user_invocable: true
---

# Value Momentum Screener

저평가된 미국 우량주 중 단기(1~4주) 반등 가능성이 높은 Top 5 종목을 추천합니다.

- **Quant 점수 (최대 70점)**: Python 스캐너가 계산한 데이터 기반 점수 — 밸류에이션(17.5) + 수익성(14) + 기술 신호(38.5)
- **Momentum 점수 (최대 30점)**: Claude 웹서치로 도출한 정성 점수 — 애널리스트 상향(10) + 실적 호재(10) + 내부자 매수(10)
- **최종 Top 5**: 100점 만점 종합 점수 기준

---

## Step 1: Python 스캐너 실행

```bash
python {{SKILL_DIR}}/value_momentum_scanner.py --json-only
```

약 2~4분 소요. 스캔이 완료되면 Quant 점수 Top 10 목록 JSON이 출력됩니다 (터미널 서식 출력 생략으로 토큰 절감).

---

## Step 2: Momentum 점수 산출 (직접 웹서치)

**Agent 호출 없이** Claude가 직접 WebSearch tool로 **상위 7개 종목**을 조사합니다. (10위권 종목은 Top 5 선정에서 거의 탈락하므로 조사 생략)

### 검색 방법

각 종목에 대해 **2개 쿼리만** 실행합니다. 7개 종목을 병렬로 동시에 날립니다.

**쿼리 A — 상승 촉매 (애널리스트 + 실적 + 내부자 통합)**
```
검색 쿼리: {TICKER} analyst upgrade earnings date insider buying 2025
```
결과에서 아래 3가지를 한 번에 추출합니다:
- **애널리스트 상향 (10점)**: 최근 4주 내 주요 IB Buy/Outperform 상향 → 10점, 중소형 IB → 5점, 없음 → 0점
- **실적 임박 (10점)**: 2주 이내 → 10점, 2~4주 이내 → 5점, 없음 → 0점
- **내부자/기관 매수 (10점)**: 임원급 순매수 또는 주요 기관 신규 포지션 → 10점, 소규모 → 5점, 없음 → 0점

**쿼리 B — 결격 사유 확인**
```
검색 쿼리: {TICKER} SEC investigation class action lawsuit recall 2025
```
- SEC 조사/집단소송/대규모 리콜 → `disqualified: true`
- 판단 모호 → `risk_warning: true` (제외 안 함)

### 처리 방식

7개 종목 × 2개 쿼리 = 총 **14개 검색**을 **모두 병렬로** 실행합니다.

각 검색 결과에서 즉시 구조화된 점수만 추출하고 원문 텍스트는 버립니다:

```
ticker | analyst(0/5/10) | earnings(0/5/10) | insider(0/5/10) | disqualified | catalyst_summary(1문장)
```

---

## Step 3: Top 5 선정, 최종 출력 및 MD 파일 저장

종합 점수(quant_score_70 + momentum_total) 기준 상위 5개를 선정합니다.

### 3-1. 터미널 출력

```
=== Value Momentum Screener 결과 ===
스캔 일시: {날짜} | 대상: 시총 상위 150종목

🏆 Top 5 저평가 단기 반등 후보

{순위}. {TICKER} ({회사명})
   현재가: ${price}
   총점 {total}점 [계량 {quant}/70 + 촉매 {momentum}/30]
   💬 {비전문가도 이해할 수 있는 한 문장 — 왜 지금 사야 하는지}

   [밸류에이션] PE {pe}x (섹터 중위수 {sector_pe}x) | FCF Yield {fcf}%
   [기술 신호]  RSI {rsi} | 52주 저점대비 +{pct}% | MACD {macd} | BB반등 {bb}
   [상승 촉매]  {catalyst_summary}

   💡 매수 포인트: ${현재가×0.99:.2f}~${현재가×1.01:.2f}
      손절 참고: ${52주저점×0.97:.2f}

---
```

**한줄요약 작성 가이드:**
- 투자 비전문가도 이해할 수 있는 일상적인 언어로 작성
- "왜 저평가인지" + "왜 곧 오를 수 있는지" 한 문장에 담기
- 예: "실적 발표 2주 앞두고 Goldman이 목표가 올렸는데 주가는 아직 반응 안 한 상태"
- 예: "섹터 전반이 반등하는데 이 종목만 바닥권에서 거래량 실린 반등 시작"
- 예: "경쟁사 대비 30% 싼 PER에 임원들이 직접 주식 사들이고 있음"

### 3-2. MD 파일 저장 (백테스팅용)

터미널 출력 후, 다음 형식의 마크다운 파일을 **반드시** 저장합니다.

**저장 경로:** `{{SKILL_DIR}}/results/YYYY-MM-DD.md`

**파일 형식:**

```markdown
# Value Momentum Screener — YYYY-MM-DD

스캔: YYYY-MM-DD HH:MM | 유니버스: 시총 상위 150종목 | 점수 체계: 계량 70점 + 촉매 30점 = 100점 만점

## Top 5 추천 종목

| # | 종목 | 현재가 | 총점 | 계량 | 촉매 | 한줄요약 |
|---|------|--------|------|------|------|----------|
| 1 | AAPL | $195.20 | 82점 | 58/70 | 24/30 | 실적 2주 앞두고 Goldman 목표가 상향, 주가 아직 바닥권 |
| 2 | ... | | | | | |

## 종목별 상세

### 1. {TICKER} — {회사명}

**가격 추적**

| 기준일 | 1주 후 | 2주 후 | 4주 후 |
|--------|--------|--------|--------|
| ${price} | | | |

**점수 내역**
- 총점: {total}/100 [계량 {quant}/70 + 촉매 {momentum}/30]
- PE: {pe}x (섹터 {sector_pe}x) | FCF Yield: {fcf}% | ROE: {roe}%
- RSI: {rsi} | 52주저점대비: +{pct}% | MACD: {macd} | BB반등: {bb}
- 촉매: {catalyst_summary}

---
```

파일 저장 후 저장 경로를 터미널에 출력합니다:
```
💾 백테스팅 기록 저장됨: {SKILL_DIR}/results/YYYY-MM-DD.md
```

---

⚠️ **주의사항**
- 이 정보는 매수 참고용이며 투자 조언이 아닙니다
- 모든 투자 결정과 그에 따른 손익은 투자자 본인의 책임입니다
- 실제 매매 전 추가 조사를 권장합니다
