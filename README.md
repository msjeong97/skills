# skills

Claude Code custom skills 모음.

## Skills

### session-archiver

컨텍스트 윈도우 사용량을 추정하고, 80% 이상일 때 세션 컨텍스트를 마크다운 파일로 아카이브하여 작업 연속성을 유지하는 스킬.

- 트리거: `/session-archiver`
- 기능:
  - 컨텍스트 사용량 퍼센트 보고
  - 80% 이상 시 `ctx_[주제]_[YYYYMMDD].md` 아카이브 파일 자동 생성
  - "Yes, clear and reload" 선택 시 `/clear` + 아카이브 파일 로드 안내 (복붙 명령어 제공)

### recommend-swing-trading-stocks

RBI-Scan: Moon Dev의 R-B-I(Research-Backtest-Implement) 시스템 기반 30일 단기 급등주 포착 스캐너.

- 트리거: `/recommend-swing-trading-stocks`
- 기능:
  - NASDAQ 100 + S&P 500 주요 101종목 자동 스캔 (커스텀 티커 지원, KOSPI `.KS` 가능)
  - 5가지 기술적 조건 필터링 (수익 기여 우선순위순):
    - **P1 ★★★★★ 거래량**: Volume > 2x 5D avg (엔진의 시동, 세력/기관 개입 신호)
    - **P2 ★★★★☆ 매물대**: 3개월 최고가 +5% 이내 저항 없음 (장애물 유무)
    - **P3 ★★★☆☆ 추세**: Close > MA20 우상향 (진입 타이밍, 과거 데이터 기반)
    - **P3 ★★★☆☆ 모멘텀**: MA5/MA20 골든크로스 (진입 타이밍, 추세 확인 보조)
    - **P4 ★★☆☆☆ RSI**: 40~60 구간 (보조 지표)
  - backtesting.py 기반 1년 백테스트 (TP +10% / SL -5%)
  - 종목별 상세 분석: 우선순위별 매수 사유, 종합 확신도(10점), 목표가/손절가, 리스크 경고
  - Near-miss 종목: 3+/5 조건 충족 종목도 참고용 표시 (실패 조건 우선순위 라벨 포함)
