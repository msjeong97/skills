# skills

Claude Code custom skills 모음.

## Skills

### blog-auto-poster

사진 + 네이버 지도 링크 + 메모를 받아 네이버 상위 노출 SEO + 전문 카피라이터 스타일의 블로그 글을 작성하는 스킬. humanizer 패턴 내재화로 AI 냄새 없는 자연스러운 한국어 작성. `[그림N]` 마커로 사진 배치 위치 명시.

- 트리거: 사진이나 방문 메모를 보내며 "블로그 글 써줘" 라고 할 때
- 기능:
  - 네이버 지도 링크에서 영업시간·가격대·메뉴 등 자동 추출
  - SEO 최적화 제목 + 본문 작성
  - 사진 배치 위치 `[그림N]` 마커로 명시 (포스팅은 `naver-blog-uploader` 사용)

### naver-blog-uploader

Playwright를 이용해 네이버 블로그에 글과 사진을 자동으로 업로드하는 스킬. `blog-auto-poster`가 작성한 초안을 받아 SmartEditor에 텍스트 붙여넣기 + 사진 업로드 + 발행까지 처리.

- 트리거: 블로그 초안이 확정되어 네이버에 올릴 준비가 됐을 때
- 기능:
  - 네이버 로그인 (쿠키 캐시)
  - 본문 클립보드 붙여넣기 + `[그림N]` 순서에 맞게 사진 자동 업로드
  - 발행까지 자동화

### recommend-swing-trading-stocks

Moon Dev의 R-B-I(Research-Backtest-Implement) 시스템 기반 30일 단기 급등주 포착 스캐너.

- 트리거: `/recommend-swing-trading-stocks`
- 기능:
  - NASDAQ 100 + S&P 500 주요 101종목 자동 스캔 (커스텀 티커 지원, KOSPI `.KS` 가능)
  - 5가지 기술적 조건 필터링 (수익 기여 우선순위순):
    - **P1 ★★★★★ 거래량**: Volume > 2x 5D avg (세력/기관 개입 신호)
    - **P2 ★★★★☆ 매물대**: 3개월 최고가 +5% 이내 저항 없음
    - **P3 ★★★☆☆ 추세**: Close > MA20 우상향
    - **P3 ★★★☆☆ 모멘텀**: MA5/MA20 골든크로스
    - **P4 ★★☆☆☆ RSI**: 40~60 구간
  - backtesting.py 기반 1년 백테스트 (TP +10% / SL -5%)
  - 종목별 상세 분석: 매수 사유, 종합 확신도(10점), 목표가/손절가, 리스크 경고

### session-archiver

컨텍스트 윈도우 사용량을 추정하고, 80% 이상일 때 세션 컨텍스트를 마크다운 파일로 아카이브하여 작업 연속성을 유지하는 스킬.

- 트리거: `/session-archiver`
- 기능:
  - 컨텍스트 사용량 퍼센트 보고
  - 80% 이상 시 `ctx_[주제]_[YYYYMMDD].md` 아카이브 파일 자동 생성
  - `/clear` + 아카이브 파일 로드 안내 (복붙 명령어 제공)

### smart-kg-researcher

지식 그래프 기반 리서처. `~/knowledge/**/*.md` 파일과 MCP knowledge graph(server-memory)를 검색해 답변하고, 새 리서치를 YAML frontmatter `.md` 파일로 저장하며 MCP 그래프에 동기화.

- 트리거: "지식 그래프에서 찾아줘", "kg에서 검색", "내 노트에서", `/smart-kg-researcher`
- 기능:
  - `~/knowledge` 파일 + MCP 그래프 통합 검색
  - 새 리서치 자동 저장 및 그래프 동기화
  - 대화 내용을 지식 그래프에 추가

### value-momentum-screener

미국 시총 상위 150개 우량주 중 저평가 지수 + 상승 신호 점수가 높은 Top 5 종목 추천. 매수 참고용 단기(1~4주) 스크리너.

- 트리거: "저평가 주식 추천해줘", "오를 것 같은 주식 찾아줘", `/value-momentum-screener`
- 기능:
  - **Quant 점수 (최대 70점)**: Python 스캐너 — 밸류에이션(17.5) + 수익성(14) + 기술 신호(38.5)
  - **Momentum 점수 (최대 30점)**: Claude 웹서치 — 애널리스트 상향(10) + 실적 호재(10) + 내부자 매수(10)
  - 결과 `results/` 디렉토리에 MD + JSON으로 저장

#### value-momentum-screener:backtesting

과거 추천 종목의 실제 수익률을 추적하고 팩터별 예측력을 분석해 스코어링 개선을 제안하는 서브 스킬.

- 트리거: "백테스팅 실행", "스크리너 성과 분석", `/value-momentum-screener:backtesting`
- 기능:
  - yfinance로 +7/+14/+28일 실제 종가 자동 조회 및 MD 파일 업데이트
  - 팩터별(52주 저점, RSI, MACD, 볼린저밴드, FCF Yield, PEG, 촉매 점수) 수익률 유효성 분석
  - 스코어링 가중치 개선 제안 리포트 저장
