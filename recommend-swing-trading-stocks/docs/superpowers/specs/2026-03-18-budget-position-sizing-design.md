# Design Spec: Budget-Based Position Sizing for recommend-swing-trading-stocks

**Date:** 2026-03-18
**Status:** Approved

---

## Summary

Update the `recommend-swing-trading-stocks` skill (SKILL.md) to ask the user for their available budget before scanning, then calculate and present position sizing — how many shares to buy, at what price, and at what sell targets — for the top conviction stocks.

No changes to `rbi_swing_scanner.py` or any Python code. All new logic lives in SKILL.md as Claude-executed steps.

---

## Workflow Changes

### Step 0: Ask for Budget (NEW)
Before running the scanner, Claude asks:
> "총 투자 가능 예산이 얼마인가요? (예: $10,000 또는 1,000만원)"

- Accept any currency; record as-is.
- Proceed to scan only after receiving the answer.

### Step 1: Run the Scanner (unchanged)

### Step 2: Interpret the Results (unchanged)

### Step 3: Position Sizing (NEW)

After presenting scan results, Claude selects top stocks by Conviction score and allocates budget:

| # of passing stocks | Allocation |
|---|---|
| ≥ 3 | Top 3: 50% / 30% / 20% |
| 2 | Top 2: 50% / 50% |
| 1 | 100% to the single stock |
| 0 | "오늘은 진입 기회 없음" — skip position sizing |

For each selected stock, output:
```
📌 {TICKER} — Conviction {score}/10
   배분 금액:   {amount} ({pct}%)
   매수 추천가: {ideal_entry}  →  약 {shares}주
   최대 진입가: {max_entry}
   손절가:     {stop_loss} ({sl_pct}%)
   목표 매도가: {target} ({tp_pct}%)
   R:R Ratio:  1 : {rr}
```

- `shares` = floor(allocated_amount / ideal_entry)
- All prices come from `trade_prices` already computed by the scanner.
- Currency/unit matches what the user provided.

### Step 4: Provide Context (was Step 3, number only changes)

---

## What Does NOT Change

- `rbi_swing_scanner.py` — no modifications
- RBI filter conditions (5/5 required)
- Backtest logic (fixed TP+10%/SL-5% baseline)
- Near-miss output section
- Disclaimer wording

---

## Edge Cases

- **0 passing stocks:** Print "오늘은 진입 기회 없음" and skip position sizing. Still show Near-miss list.
- **Tie in Conviction score:** Break ties by R:R ratio (higher = better).
- **Budget too small for even 1 share:** Warn the user — "예산이 {ticker} 1주 매수가({price})보다 적습니다."

---

## Files Modified

| File | Change |
|---|---|
| `SKILL.md` (in `.agents/skills/`) | Add Step 0, add Step 3, renumber old Step 3 → Step 4 — **edit this first** |
| `SKILL.md` (in `repo/skills/`) | Copy from `.agents/skills/` version after editing |

---

## Notes

- **통화 단위:** 스캐너는 USD 기준 가격을 반환합니다. 사용자가 KRW로 예산을 입력한 경우, Step 0에서 통화를 확인하고 USD 기준인지 명시적으로 안내합니다. 단위 불일치 시 환산하거나 사용자에게 USD로 재입력 요청.
- **출력 포맷:** Step 3의 출력 예시는 참고용 템플릿이며, 정렬/공백 등 사소한 차이는 허용됩니다.
