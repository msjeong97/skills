# Phase 0: 초기 설정

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
```

---

## 섹션 A: MCP 설정

```bash
claude mcp add -s user memory -- npx -y @modelcontextprotocol/server-memory@latest
claude mcp list
```

`memory` 항목이 나타나면 성공입니다. Claude Code를 재시작한 후 다시 질문해 주세요.

---

## 섹션 B: 위키 초기화 + Migration

### B-1: 디렉토리 생성

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
mkdir -p "$VAULT/wiki/concepts"
mkdir -p "$VAULT/wiki/entities"
mkdir -p "$VAULT/wiki/sources"
mkdir -p "$VAULT/wiki/analyses"
mkdir -p "$VAULT/raw"
```

### B-2: Migration 수행

`migrate.md`를 Read 도구로 읽어 기존 `$VAULT/research/` 파일들을 새 구조로 복사합니다.

### B-3: index.md 생성

Write 도구로 `$VAULT/index.md`를 아래 내용으로 생성합니다:

```markdown
# Knowledge Wiki Index

마지막 업데이트: {오늘 날짜}

---

## Concepts

```dataview
TABLE title, tags, date_ingested AS "수집일"
FROM "wiki/concepts"
SORT date_ingested DESC
```

## Sources

```dataview
TABLE title, tags, date_ingested AS "수집일"
FROM "wiki/sources"
SORT date_ingested DESC
```

## Entities

```dataview
TABLE title, tags, source_count AS "소스 수"
FROM "wiki/entities"
SORT source_count DESC
```

## Analyses

```dataview
TABLE title, tags, date_ingested AS "수집일"
FROM "wiki/analyses"
SORT date_ingested DESC
```
```

> Obsidian에서 Dataview 플러그인이 활성화되면 위 코드블록이 동적 테이블로 렌더링됩니다.
> Dataview 없이 사용할 때는 LLM이 Glob으로 파일을 직접 수집해 탐색합니다.

### B-4: log.md 생성

Write 도구로 `$VAULT/log.md`를 아래 내용으로 생성합니다:

```markdown
# Knowledge Wiki Activity Log

append-only 로그. 파싱: `grep "^## \[" log.md | tail -10`

---

## [{오늘 날짜}] migrate | Initial wiki setup

- 기존 research/ 파일들을 wiki/ 구조로 migration
- wiki/concepts/paimon, wiki/sources/iceberg-kafka-connect, wiki/analyses/paimon-iceberg 생성
- index.md, log.md 초기 생성
- 원본 파일 보존: $VAULT/research/
```

---

## 섹션 C: 그래프 워밍 (cold start 시)

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
find "$VAULT/wiki" -name "*.md" 2>/dev/null
```

Glob 도구로 `$VAULT/wiki/**/*.md` 전체 수집.

**스킵 조건:** `last_synced_with_graph`가 오늘 날짜인 파일은 건너뜁니다.

각 파일 처리:
1. Read 도구로 읽기
2. 본문에서 핵심 엔티티 추출 (기술명, 시스템명, 개념명)
3. MCP `create_entities` 생성:
   ```json
   [{"name": "엔티티명", "entityType": "Technology", "observations": ["핵심 설명 1~3문장"]}]
   ```
   `entityType`: Technology / Concept / System
4. 2개 이상 엔티티면 MCP `create_relations` 생성
5. frontmatter `related_nodes`, `last_synced_with_graph` 업데이트 후 Write 저장

완료 메시지:
> "그래프 워밍 완료: {N}개 파일 처리, {M}개 엔티티 생성"
