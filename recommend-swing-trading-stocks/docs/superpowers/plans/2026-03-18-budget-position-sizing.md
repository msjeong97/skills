# Budget-Based Position Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `recommend-swing-trading-stocks` SKILL.md to ask the user for their budget before scanning, then present per-stock position sizing (shares, entry/stop/target prices) for the top 2-3 conviction stocks.

**Architecture:** SKILL.md-only change — no Python code modified. Claude asks for budget in Step 0, runs the scanner in Step 1 (unchanged), interprets results in Step 2 (unchanged), then in the new Step 3 selects top stocks by Conviction score, allocates budget at 50/30/20%, and calculates share counts. Old Step 3 becomes Step 4.

**Tech Stack:** Markdown (SKILL.md). Two file copies must be kept in sync: `.agents/skills/` is edited first, then copied to `repo/skills/`.

---

## File Map

| File | Change |
|---|---|
| `/Users/minseop/.agents/skills/recommend-swing-trading-stocks/SKILL.md` | Primary edit — add Step 0, Step 3, renumber Step 3→4 |
| `/Users/minseop/repo/skills/recommend-swing-trading-stocks/SKILL.md` | Secondary — overwrite with the edited version |

---

### Task 1: Add Step 0 — Budget Question

**Files:**
- Modify: `/Users/minseop/.agents/skills/recommend-swing-trading-stocks/SKILL.md`

- [ ] **Step 1: Read the current SKILL.md**

  Open `/Users/minseop/.agents/skills/recommend-swing-trading-stocks/SKILL.md` and confirm the current structure.

- [ ] **Step 2: Insert Step 0 before Step 1**

  Add the following section immediately before `### Step 1: Run the Scanner`:

  ```markdown
  ### Step 0: Ask for Budget

  Before running the scanner, ask the user:

  > "총 투자 가능 예산이 얼마인가요? (예: $10,000 또는 1,000만원)"

  - Wait for the user's answer before proceeding.
  - Record the amount and currency unit as provided.
  - If the user inputs KRW (원), note that the scanner returns USD prices. Ask:
    > "스캐너는 USD 기준 가격을 반환합니다. USD 환산 금액으로 입력해주시겠어요? 또는 현재 환율을 알려주시면 제가 환산합니다."
  - Proceed to Step 1 only after the budget is confirmed.
  ```

- [ ] **Step 3: Verify insertion looks correct**

  Re-read the file and confirm Step 0 appears before Step 1 with correct heading levels.

- [ ] **Step 4: Commit**

  ```bash
  git -C /Users/minseop/.agents add skills/recommend-swing-trading-stocks/SKILL.md
  git -C /Users/minseop/.agents commit -m "feat: add Step 0 budget question to swing trading skill"
  ```
  *(If `.agents/` is not a git repo, skip the commit and proceed.)*

---

### Task 2: Add Step 3 — Position Sizing

**Files:**
- Modify: `/Users/minseop/.agents/skills/recommend-swing-trading-stocks/SKILL.md`

- [ ] **Step 1: Insert Step 3 after Step 2**

  Add the following section after `### Step 2: Interpret the Results` and before `### Step 3: Provide Context`:

  ```markdown
  ### Step 3: Position Sizing

  Using the budget from Step 0 and the Conviction scores from Step 2, select the top stocks and calculate position sizes.

  **Stock selection:**
  - ≥ 3 passing stocks → pick top 3 by Conviction score → allocate 50% / 30% / 20%
  - 2 passing stocks → pick both → allocate 50% / 50%
  - 1 passing stock → allocate 100%
  - 0 passing stocks → print "오늘은 진입 기회 없음 — 포지션 계산을 건너뜁니다." and skip to Step 4. (Near-miss 목록은 Step 2에서 이미 출력되므로 별도 처리 불필요)

  **Tie-breaking:** If two stocks share the same Conviction score, rank by R:R Ratio (higher = better).

  **Budget too small:** If the allocated amount for a stock is less than 1 share at `ideal_entry`, warn:
  > "예산이 {TICKER} 1주 매수가({ideal_entry})보다 적습니다. 해당 종목은 건너뜁니다."

  **Per-stock output format:**

  ```
  📌 {TICKER} — Conviction {score}/10
     배분 금액:   {allocated_amount} ({pct}%)
     매수 추천가: {ideal_entry}  →  약 {shares}주
     최대 진입가: {max_entry}
     손절가:     {stop_loss} ({sl_pct}%)
     목표 매도가: {target} ({tp_pct}%)
     R:R Ratio:  1 : {rr}
  ```

  - `shares` = floor(allocated_amount ÷ ideal_entry)
  - All prices come from `trade_prices` computed by the scanner in Step 1.
  - Use the same currency unit the user provided in Step 0.
  ```

- [ ] **Step 2: Verify insertion looks correct**

  Re-read the file and confirm Step 3 appears between Step 2 and the old Step 3 (now to be renumbered).

- [ ] **Step 3: Commit**

  ```bash
  git -C /Users/minseop/.agents add skills/recommend-swing-trading-stocks/SKILL.md
  git -C /Users/minseop/.agents commit -m "feat: add Step 3 position sizing to swing trading skill"
  ```

---

### Task 3: Renumber Old Step 3 → Step 4

**Files:**
- Modify: `/Users/minseop/.agents/skills/recommend-swing-trading-stocks/SKILL.md`

- [ ] **Step 1: Rename heading**

  Find `### Step 3: Provide Context` and change it to `### Step 4: Provide Context`.

- [ ] **Step 2: Verify no other Step 3 references remain**

  Search the file for any remaining references to "Step 3" that still point to the old context section. Update if found.

- [ ] **Step 3: Commit**

  ```bash
  git -C /Users/minseop/.agents add skills/recommend-swing-trading-stocks/SKILL.md
  git -C /Users/minseop/.agents commit -m "chore: renumber Step 3→4 in swing trading skill"
  ```

---

### Task 4: Sync to repo/skills/ Copy

**Files:**
- Modify: `/Users/minseop/repo/skills/recommend-swing-trading-stocks/SKILL.md`

- [ ] **Step 1: Overwrite the repo copy**

  ```bash
  cp /Users/minseop/.agents/skills/recommend-swing-trading-stocks/SKILL.md \
     /Users/minseop/repo/skills/recommend-swing-trading-stocks/SKILL.md
  ```

- [ ] **Step 2: Verify both files are identical**

  ```bash
  diff /Users/minseop/.agents/skills/recommend-swing-trading-stocks/SKILL.md \
       /Users/minseop/repo/skills/recommend-swing-trading-stocks/SKILL.md
  ```

  Expected output: *(empty — no differences)*

- [ ] **Step 3: Commit repo copy**

  ```bash
  git -C /Users/minseop/repo add skills/recommend-swing-trading-stocks/SKILL.md
  git -C /Users/minseop/repo commit -m "chore: sync SKILL.md with budget position sizing update"
  ```

---

### Task 5: Manual Smoke Test

**Files:** None modified.

- [ ] **Step 1: Invoke the skill and verify Step 0 fires**

  Run `/recommend-swing-trading-stocks` and confirm Claude asks for the budget before executing the scanner.

- [ ] **Step 2: Provide a test budget and verify position sizing output**

  Enter a budget (e.g., `$10,000`). After the scan completes, confirm:
  - Top 2-3 stocks are selected by Conviction score.
  - Budget is split 50/30/20% (or 50/50 for 2 stocks).
  - Each stock shows: allocated amount, shares, ideal_entry, max_entry, stop_loss, target, R:R.

- [ ] **Step 3: Test edge case — 0 passing stocks**

  Invoke the skill normally with a ticker that is very unlikely to pass all 5 conditions (e.g., a low-volatility ETF like `BIL`):
  > `/recommend-swing-trading-stocks BIL`

  Confirm Claude prints "오늘은 진입 기회 없음" and skips position sizing, while still showing the Near-miss list from Step 2.

- [ ] **Step 4: Test edge case — budget too small**

  Provide a very small budget (e.g., `$1`) and confirm the per-stock warning appears correctly.
