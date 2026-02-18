# Session Context: 동적 매매가격 산출 기능 추가
> Archived on 2026-02-18

## Core Structure
- `/Users/minseop/repo/skills/recommend-swing-trading-stocks/rbi_swing_scanner.py` — 메인 스캐너 (수정 대상)
- `SKILL.md` — 스킬 설명 문서
- `requirements.txt` — 의존성

## Architecture & Decisions
- **백테스트는 기존 고정 TP+10%/SL-5% 유지** — 일관된 기준으로 과거 성과 측정
- **동적 가격은 "현재 시점 매매 계획"에만 적용** — `calc_trade_prices()` 함수가 담당
- 구매추천가: MA5 눌림목 vs 현재가 (0.99 기준), 최대진입가 = 현재가 * 1.01
- 손절가: 3가지 기술적 기준(10일 스윙로우, MA20, ATR) 중 **가장 높은 값** 채택, -3%~-8% 클램핑
- 목표매도가: 3가지 기준(저항선, ATR x3, 피보나치 1.618) 중 **+8% 이상 후보 중 가장 낮은 값** 채택

## Current Progress
### 완료된 작업 (Step 1~4)
1. **Step 1: `calc_trade_prices(stock, df)` 함수 추가** — 라인 ~490 부근, Phase 2 Backtest 섹션 뒤에 삽입
   - ATR(14) 계산
   - ideal_entry, max_entry, entry_reason 산출
   - stop_loss, sl_pct, sl_reason 산출 (3기준 + 클램핑)
   - target, tp_pct, tp_reason 산출 (3기준 + 기본 +10% 폴백)
   - rr_ratio 계산
   - dict 반환

2. **Step 2: `main()` 수정** — backtest 후, df 삭제 전에 `r["trade_prices"] = calc_trade_prices(r, r["df"])` 호출

3. **Step 3: Phase 1 테이블 업데이트** — 기존 컬럼에 Entry($), SL($), Target($), R:R 추가

4. **Step 4: Phase 2 상세 분석 TRADE PLAN 섹션 교체** — 고정 TP/SL → 동적 가격 + 산출근거 출력
   - 구매추천가, 최대진입가, 손절가(+%), 목표매도가(+%), R:R Ratio, ATR, Swing Low 표시
   - RISK 섹션의 sl_price도 동적 값으로 교체

### 구문 검증
- `py_compile.compile()` 통과 — 문법 오류 없음

## Pending TODOs
- [x] **Step 5: 실행 테스트** — 완료 (2026-02-18)
  - 오늘 시장: 5조건 통과 종목 0개 (정상 — 엄격한 필터)
  - LRCX (거래량 기준 완화 테스트): TRADE PLAN 섹션 동적 가격 + 산출근거 정상 출력 확인
  - calc_trade_prices() LRCX/AVGO/VRTX 모두 정상 반환
  - 손절가 클램핑(-3%~-8%), 목표가 저항선/ATR/피보나치 분기 모두 정상

## 모든 작업 완료 ✓

## Key Code References
- `calc_trade_prices(stock, df)` — rbi_swing_scanner.py, Phase 2 Backtest 뒤 "Dynamic Trade Price Calculator" 섹션
- `main()` — rbi_swing_scanner.py 하단, `r["trade_prices"] = calc_trade_prices(r, r["df"])` 라인
- Phase 1 테이블 출력 — `print_results()` 함수 내 "Phase 1 Research" 블록
- Phase 2 TRADE PLAN — `print_results()` 함수 내 종목별 상세 분석 루프

## Notes
- 테스트 명령: `python /Users/minseop/repo/skills/recommend-swing-trading-stocks/rbi_swing_scanner.py AAPL NVDA`
- 계획 원본: `/Users/minseop/.claude/projects/-Users-minseop/7fbaf91a-b33a-4629-8475-57f6fb91312a.jsonl` (plan mode transcript)
- 백테스트 엔진은 `backtesting.py` 라이브러리 우선, 미설치 시 수동 폴백 (`_USE_BT_LIB` 플래그)
