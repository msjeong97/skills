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

## Step 2: 상승 신호 점수 산출 (AI 웹서치)

Step 1 JSON의 상위 20개 종목에 대해 순서대로 웹서치를 수행합니다.

**각 종목당 3가지 확인:**

### 1. 애널리스트 목표가 상향 (3점)
검색: `"{TICKER}" analyst upgrade price target raised 2025 site:finance.yahoo.com OR marketwatch.com OR benzinga.com`

- 최근 4주 내 Goldman Sachs, Morgan Stanley, JPMorgan, BofA, Citi 등 주요 IB가 목표가를 올리거나 Buy/Outperform으로 신규 커버리지 → **3점**
- 없음 → **0점**

### 2. 실적 발표 임박 (2점)
검색: `"{TICKER}" earnings date next quarter`

- 앞으로 2주 이내 실적 발표 예정 → **2점** (어닝 기대감이 주가에 선반영되는 PEAD 효과)
- 없음 → **0점**

### 3. 내부자/기관 매수 (2점)
검색: `"{TICKER}" insider buying OR institutional accumulation 2025`

- 최근 4주 내 임원급 내부자 순매수 또는 주요 기관 신규 포지션 증가 보도 → **2점**
- 없음 → **0점**

### 결격 사유 확인 (자동 제외)
검색: `"{TICKER}" SEC investigation OR class action lawsuit OR massive recall 2025`

- SEC 공식 조사 개시, Class Action 집단소송, 대규모 리콜, 회계 부정 → **해당 종목 제외**
- 판단이 모호한 경우 → 제외하지 않고 출력에 `(⚠️ 리스크 확인 권고)` 추가

---

## Step 3: Top 5 선정 및 최종 출력

종합 점수(저평가 지수 + 상승 신호 점수) 상위 5개를 선정해 다음 형식으로 출력합니다.

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

---

⚠️ **주의사항**
- 이 정보는 매수 참고용이며 투자 조언이 아닙니다
- 모든 투자 결정과 그에 따른 손익은 투자자 본인의 책임입니다
- 실제 매매 전 추가 조사를 권장합니다
