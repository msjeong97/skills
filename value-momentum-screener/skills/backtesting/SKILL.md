---
name: value-momentum-screener:backtesting
description: >
  value-momentum-screener 과거 추천 종목의 실제 수익률을 추적하고,
  팩터별 예측력 분석을 통해 스코어링 가중치 개선을 제안합니다.
  트리거: "/value-momentum-screener:backtesting", "백테스팅 실행", "스크리너 성과 분석"
user_invocable: true
---

# Value Momentum Screener — Backtesting

과거 추천 종목의 실제 수익률을 채우고, 어떤 팩터가 유효했는지 분석합니다.

---

## Step 1: 가격 업데이트 스크립트 실행

```bash
python {{SKILL_DIR}}/backtest_updater.py
```

스크립트가 다음을 수행합니다:
- `results/*.json`의 모든 픽에 대해 +7/+14/+28일 실제 종가 조회 (yfinance)
- `results/YYYY-MM-DD.md`의 빈 가격 추적 칸 자동 기입
- 분석용 JSON을 stdout으로 출력

스크립트 출력 JSON을 아래 분석에 사용합니다.

---

## Step 2: 팩터 유효성 분석

스크립트 출력의 `picks` 배열에서 `returns`가 null이 아닌 항목만 사용합니다.

아래 팩터별로 **수익률이 양수인 픽의 비율**과 **평균 수익률**을 계산합니다:

| 팩터 | 분석 방법 |
|------|-----------|
| 52주 저점 대비 5~20% 구간 여부 | 해당/비해당 그룹 수익률 비교 |
| RSI 30~45 구간 여부 | 해당/비해당 그룹 수익률 비교 |
| MACD 골든크로스 있/없 | 있/없 그룹 수익률 비교 |
| 볼린저밴드 %B < 0.3 | 해당/비해당 그룹 수익률 비교 |
| FCF Yield > 5% | 해당/비해당 그룹 수익률 비교 |
| PEG < 1.0 | 해당/비해당 그룹 수익률 비교 |
| 촉매 점수 15점 이상 | 해당/비해당 그룹 수익률 비교 |

샘플 수가 30 미만이면 모든 수치 옆에 `⚠️ 샘플 부족 (N=X)` 경고를 표시합니다.

---

## Step 3: 스코어링 개선 제안

분석 결과를 바탕으로 아래 형식으로 개선 제안을 작성합니다:

```
## 스코어링 개선 제안

### 유효 팩터 (현행 유지 또는 가중치 ↑)
- [팩터명]: 해당 그룹 평균 수익률 X.X% vs 비해당 Y.Y%

### 무효 팩터 (가중치 ↓ 또는 제거 고려)
- [팩터명]: 유의미한 수익률 차이 없음 (X.X% vs Y.Y%)

### 현행 vs 권장 가중치
| 팩터 | 현행 | 권장 |
|------|------|------|
| 52주 저점 | 15점 | ?점 |
| RSI | 12점 | ?점 |
| MACD | 12점 | ?점 |
| 볼린저밴드 | 8점 | ?점 |
| FCF Yield | 10점 | ?점 |
| PEG | 5점 | ?점 |
```

---

## Step 4: 리포트 저장

분석 결과를 아래 경로에 저장합니다:

**저장 경로:** `{{SKILL_DIR}}/../../results/backtest-report-YYYY-MM-DD.md`

```markdown
# Backtest Report — YYYY-MM-DD

분석 기간: YYYY-MM-DD ~ YYYY-MM-DD | 총 스캔: N회 | 수익률 확인된 픽: M개

## 팩터 유효성 분석
[Step 2 결과]

## 스코어링 개선 제안
[Step 3 결과]

⚠️ 이 분석은 투자 조언이 아닙니다. 샘플 수가 충분하지 않으면 통계적 신뢰도가 낮습니다.
```

저장 후 경로를 출력합니다:
```
💾 백테스팅 리포트 저장됨: results/backtest-report-YYYY-MM-DD.md
```
