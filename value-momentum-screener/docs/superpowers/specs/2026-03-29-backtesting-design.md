# Backtesting Skill Design — value-momentum-screener:backtesting

**Date:** 2026-03-29
**Skill:** `value-momentum-screener:backtesting`
**Location:** `~/repo/skills/value-momentum-screener/skills/backtesting/SKILL.md`

---

## 목적

`/value-momentum-screener`가 매일 실행되며 `results/`에 데이터가 쌓이면, 백테스팅 스킬이 이를 분석해:

1. 과거 추천 종목의 실제 수익률을 MD 파일에 자동 기입
2. 각 팩터별 예측력(수익률 상관관계)을 분석
3. 스코어링 가중치 개선 제안을 리포트로 출력

---

## 아키텍처

### 입력
- `results/*.json` — raw quant 점수 데이터 (팩터별 세부 점수 포함)
- `results/*.md` — 촉매 점수 + 현재가 기준일

### 출력
- `results/YYYY-MM-DD.md` — 빈 가격 추적 칸 업데이트 (1주/2주/4주 후 실제가)
- `results/backtest-report-YYYY-MM-DD.md` — 종합 분석 리포트

---

## 실행 흐름 (5단계)

### Step 1: 결과 스캔
- `results/*.json` 파일 전체 읽기
- 각 날짜별 Top 10 종목, 팩터별 점수, 기준 가격 수집
- 이미 가격이 채워진 셀은 건너뜀 (중복 업데이트 방지)

### Step 2: 가격 수집
- 각 픽에 대해 스캔일 기준 +7거래일/+14거래일/+28거래일 가격 조회
- yfinance(Python) 또는 WebSearch로 과거 주가 조회
- 미래 날짜(아직 도래하지 않은 구간)는 "—"로 처리

### Step 3: MD 파일 업데이트
- `results/YYYY-MM-DD.md`의 "가격 추적" 테이블 빈 칸 채우기
- 수익률(%) 자동 계산해서 칸에 함께 표기: `$XXX (+Y%)`

### Step 4: 팩터별 수익률 분석

분석 대상 팩터:

| 팩터 | 현행 배점 | 분석 방법 |
|------|-----------|-----------|
| 52주 저점 대비 위치 (5~20%) | 15점 | 구간별 평균 수익률 비교 |
| RSI (30~45) | 12점 | RSI 구간별 수익률 |
| MACD 골든크로스 | 12점 | 있/없 수익률 차이 |
| 볼린저밴드 %B | 8점 | 저점 터치 후 반등 유효성 |
| FCF Yield | 10점 | 높을수록 수익률 높은지 |
| PEG | 5점 | <1.0 구간 유효성 |
| 촉매 총점 (30점) | — | 계량 vs 촉매 예측력 비교 |

샘플 수 < 30일 때는 모든 분석에 "참고용 (샘플 부족)" 경고 명시.

### Step 5: 리포트 생성

리포트 구성:

```
## 백테스팅 리포트 — YYYY-MM-DD

분석 기간: N일 | 총 픽 수: M개 | 완료된 수익률 구간: X개

### 📊 팩터 유효성 분석
- 유효 팩터 Top 3 (수익률 상관관계 ↑)
- 무효 팩터 (높은 점수 대비 낮은 예측력)
- 계량(70점) vs 촉매(30점) 기여도 비교

### 🔧 스코어링 개선 제안
- 현행 가중치 vs 데이터 기반 권장 가중치 표
- 팩터 제거/추가 제안

### ⚠️ 신뢰도
- 샘플 수 및 통계적 유의미성 명시
```

---

## 제약 사항

- 유의미한 분석을 위해 최소 **4주(~20개 스캔일) 이상** 데이터 필요
- 샘플 부족 시에도 가격 채우기(Step 1~3)는 항상 실행
- 시장 휴장일(주말/공휴일)은 yfinance가 자동 처리

---

## 스킬 메타데이터

```yaml
name: value-momentum-screener:backtesting
description: >
  value-momentum-screener 과거 추천 종목의 실제 수익률을 추적하고,
  팩터별 예측력 분석을 통해 스코어링 가중치 개선을 제안합니다.
  트리거: "/value-momentum-screener:backtesting", "백테스팅 실행", "스크리너 성과 분석"
user_invocable: true
```
