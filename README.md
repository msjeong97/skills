# skills

Claude Code custom skills 모음.

## Skills

### session-archiver

컨텍스트 윈도우 사용량을 추정하고, 80% 이상일 때 세션 컨텍스트를 마크다운 파일로 아카이브하여 작업 연속성을 유지하는 스킬.

- 트리거: `/session-archiver`
- 기능:
  - 컨텍스트 사용량 퍼센트 보고
  - 80% 이상 시 `ctx_[주제]_[YYYYMMDD].md` 아카이브 파일 자동 생성
  - `/clear` 후 새 세션에서 아카이브 파일로 작업 재개 안내
