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

- **저평가 지수 (100점)**: 밸류에이션(25) + 수익성(20) + 단기 기술적 반전 신호(55)
- **상승 신호 점수 (7점)**: 애널리스트 상향, 실적 임박, 내부자 매수 (AI 웹서치)
- **최종 Top 5**: 종합 점수(최대 107점) 기준

---

## Step 1: Python 스캐너 실행

```bash
python {{SKILL_DIR}}/value_momentum_scanner.py
```

약 2~4분 소요. 스캔이 완료되면 저평가 지수 Top 20 목록과 AI 웹서치용 JSON이 출력됩니다.

---

## Step 2: 상승 신호 점수 산출 (4개 병렬 subagent)

Step 1 JSON의 상위 20개 종목을 **5개씩 4그룹으로 나눠 병렬 subagent에 동시 dispatch**합니다.

### 그룹 분할

- **Agent A**: rank 1~5
- **Agent B**: rank 6~10
- **Agent C**: rank 11~15
- **Agent D**: rank 16~20

### 각 Agent에게 전달할 프롬프트 템플릿

다음 프롬프트를 4개 Agent에 동시에 dispatch합니다 (단일 메시지에 4개 Agent tool 호출):

```
당신은 주식 리서치 분석가입니다. 아래 종목들의 단기 상승 신호를 웹서치로 확인하고 점수를 매기세요.

## 담당 종목
{해당 그룹의 ticker, name, current_price, undervalue_score}

## 각 종목당 확인 항목 (총 7점)

### 1. 애널리스트 목표가 상향 (3점)
검색: "{TICKER}" analyst upgrade price target raised site:finance.yahoo.com OR marketwatch.com OR benzinga.com
- 최근 4주 내 Goldman Sachs, Morgan Stanley, JPMorgan, BofA, Citi 등 주요 IB 상향/Buy 신규 → 3점
- 없음 → 0점

### 2. 실적 발표 임박 (2점)
검색: "{TICKER}" earnings date next
- 앞으로 2주 이내 실적 발표 예정 → 2점
- 없음 → 0점

### 3. 내부자/기관 매수 (2점)
검색: "{TICKER}" insider buying OR institutional accumulation
- 최근 4주 내 임원급 내부자 순매수 또는 주요 기관 신규 포지션 → 2점
- 없음 → 0점

### 결격 사유 (자동 제외 플래그)
검색: "{TICKER}" SEC investigation OR class action lawsuit OR recall
- SEC 조사/집단소송/대규모 리콜/회계 부정 → disqualified: true
- 모호한 경우 → risk_warning: true (제외하지 않음)

## 출력 형식 (JSON만 출력, 다른 텍스트 없이)

[
  {
    "ticker": "XXX",
    "analyst_upgrade": 0 또는 3,
    "earnings_soon": 0 또는 2,
    "insider_buying": 0 또는 2,
    "signal_total": 합계,
    "disqualified": false,
    "risk_warning": false,
    "signal_summary": "웹서치 결과 한줄 요약 (한국어)"
  },
  ...
]
```

### 4개 Agent 동시 dispatch

Agent tool을 단일 메시지에 4개 동시 호출합니다. 각 Agent가 완료되면 결과 JSON을 수집합니다.

---

## Step 3: Top 5 선정, 최종 출력 및 MD 파일 저장

종합 점수(저평가 지수 + 상승 신호 점수) 상위 5개를 선정합니다.

### 3-1. 터미널 출력

```
=== Value Momentum Screener 결과 ===
스캔 일시: {날짜} | 대상: 시총 상위 150종목

🏆 Top 5 저평가 단기 반등 후보

{순위}. {TICKER} ({회사명})
   현재가: ${price}
   저평가 지수: {score}/100 | 상승 신호 점수: {signal}/7 | 종합: {total:.1f}점
   💬 한줄요약: {비전문가도 이해할 수 있는 한 문장}

   [밸류에이션] PE {pe}x (섹터 중위수 {sector_pe}x) | FCF Yield {fcf}%
   [기술 신호]  RSI {rsi} | 52주 저점대비 +{pct}% | MACD {macd} | BB반등 {bb}
   [상승 신호]  {웹서치 결과 요약}

   💡 매수 포인트: ${현재가×0.99:.2f}~${현재가×1.01:.2f}
      목표가: ${애널리스트 컨센서스 평균, 참고용} | 손절: ${52주저점×0.97:.2f} (참고용)

---
```

**한줄요약 작성 가이드:**
- 투자 비전문가도 이해할 수 있는 일상적인 언어로 작성
- "왜 저평가인지" + "왜 곧 오를 수 있는지" 한 문장에 담기
- 예: "실적 발표 2주 앞두고 애널리스트가 목표가 올렸는데 주가는 아직 반응 안 한 상태"
- 예: "섹터 전반이 반등하는데 이 종목만 바닥권에서 거래량 실린 반등 시작"
- 예: "경쟁사 대비 30% 싼 PER에 임원들이 직접 주식 사들이고 있음"

### 3-2. MD 파일 저장 (백테스팅용)

터미널 출력 후, 다음 형식의 마크다운 파일을 **반드시** 저장합니다.

**저장 경로:** `{{SKILL_DIR}}/results/YYYY-MM-DD.md`

**파일 형식:**

```markdown
# Value Momentum Screener — YYYY-MM-DD

**스캔 일시:** YYYY-MM-DD HH:MM
**유니버스:** 시총 상위 150종목
**추천 기준:** 저평가 지수(100점) + 상승 신호 점수(7점) = 최대 107점

---

## Top 5 추천 종목

### 1. {TICKER} — {회사명}

| 항목 | 값 |
|------|-----|
| 추천일 현재가 | ${price} |
| 52주 저점 | ${52w_low} |
| 52주 고점 | ${52w_high} |
| 저평가 지수 | {undervalue_score}/100 |
| 상승 신호 점수 | {signal_score}/7 |
| **종합 점수** | **{total}점** |

**한줄요약:** {summary}

**밸류에이션**
- PE: {pe}x (섹터 중위수: {sector_pe}x)
- FCF Yield: {fcf}%
- PEG: {peg}

**수익성**
- ROE: {roe}%
- 영업이익률: {op_margin}% (섹터 {op_margin_rank})

**기술 신호**
- RSI(14): {rsi}
- 52주 저점 대비: +{pct_from_low}%
- MACD 골든크로스: {macd}
- 볼린저밴드 반등: {bb}
- 거래량 동반 상승: {vol}

**상승 신호**
- 애널리스트 목표가 상향: {analyst_detail} ({analyst_score}점)
- 실적 발표 임박: {earnings_detail} ({earnings_score}점)
- 내부자/기관 매수: {insider_detail} ({insider_score}점)

**매수 참고**
- 매수 포인트: ${buy_low}~${buy_high}
- 목표가: ${target} (참고용)
- 손절: ${stop} (참고용)

---

### 2. ...

---

## 백테스팅 추적 (나중에 채워넣기)

| 종목 | 추천일 가격 | 1주 후 | 2주 후 | 4주 후 | 결과 |
|------|------------|--------|--------|--------|------|
| {TICKER} | ${price} | | | | |
| ... | | | | | |

---

> ⚠️ 투자 참고용. 손익 책임은 투자자 본인에게 있습니다.
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
