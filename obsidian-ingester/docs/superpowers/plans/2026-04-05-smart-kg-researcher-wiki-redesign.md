# smart-kg-researcher Wiki Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LLM Wiki 패턴에 따라 smart-kg-researcher 스킬을 재설계하고, iCloud에 위치한 Obsidian vault를 지식 베이스로 사용하며, 기존 `research/**/*.md` 데이터를 무손실 migration한다.

**Architecture:**
세 레이어(Raw Sources / Wiki / Schema)와 세 오퍼레이션(Ingest / Query / Lint)으로 위키를 점진적으로 축적한다. Obsidian vault가 iCloud에 있어 Mac Mini / MacBook 양쪽에서 접근 가능하다. LLM이 파일을 수정하면 iCloud가 동기화하고 Obsidian이 실시간 반영한다. 기존 MCP `server-memory` 그래프는 유지하되, 파일 레이어 구조와 frontmatter를 Obsidian/Dataview 호환으로 표준화한다.

**Tech Stack:** Claude Code skills (markdown), MCP `server-memory`, Obsidian (iCloud vault), Dataview plugin, `[[wikilink]]` 형식, YAML frontmatter

---

## Vault 경로

```
Mac Mini:  /Users/minseop/Library/Mobile Documents/com~apple~CloudDocs/Obsidian
MacBook:   ~/Library/Mobile Documents/com~apple~CloudDocs/Obsidian
```

> **경로에 공백이 포함**되어 있어 모든 bash 명령에서 따옴표 필수.
> 스킬 파일에서는 `VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"` 변수를 사용한다.

## 신규 디렉토리 구조

```
$VAULT/                         ← Obsidian vault root = Claude Code 작업 디렉토리
├── .obsidian/                  ← Obsidian 설정 (LLM 절대 수정 금지)
├── index.md                    ← 위키 카탈로그 (Dataview 쿼리 포함)
├── log.md                      ← 활동 로그 (append-only)
├── wiki/
│   ├── concepts/<topic>/       ← 개념·기능 페이지
│   ├── entities/               ← 기술·시스템 엔티티 페이지
│   ├── sources/<topic>/        ← 소스 요약 페이지
│   └── analyses/               ← 종합·비교·분석 페이지
└── raw/                        ← 원시 소스 (사용자 큐레이션, LLM 읽기 전용)
```

**Migration 매핑 (현재 `$VAULT/research/` 기준):**

`research/` 아래 모든 서브폴더를 자동 감지하여 migration. 현재 확인된 폴더:

| research 폴더 | wiki 경로 | type | 파일 수 |
|---|---|---|---|
| `paimon/` | `wiki/concepts/paimon/` | concept | 31개 |
| `iceberg-kafka-connect-research/` | `wiki/sources/iceberg-kafka-connect/` | source | 6개 |
| `paimon-iceberg/` | `wiki/analyses/paimon-iceberg/` | analysis | 1개 |
| `claude-code/` | `wiki/concepts/claude-code/` | concept | 1개 |
| `eks-logging/` | `wiki/sources/eks-logging/` | source | 1개 |
| `starrocks/` | `wiki/concepts/starrocks/` | concept | 1개 |
| **신규 폴더** | `wiki/concepts/<폴더명>/` | concept | 자동 |

**type 판단 규칙:**
- 폴더명에 `-research`, `-study` 포함 → `source`
- 폴더명에 두 기술명 조합 (예: `paimon-iceberg`) → `analysis`
- 그 외 → `concept`

**태그 자동 생성:** 파일 본문에서 기술명·시스템명 3~6개 추출하여 설정

## 신규 스킬 파일

| 파일 | 역할 | 기존 대응 |
|---|---|---|
| `SKILL.md` | 엔트리 + 라우팅 | 전면 재작성 |
| `setup.md` | MCP + 위키 초기화 | 재작성 |
| `migrate.md` | 기존 데이터 migration | 신규 |
| `query.md` | Query 오퍼레이션 | search.md + answer.md 대체 |
| `ingest.md` | Ingest 오퍼레이션 | research.md 대체 |
| `lint.md` | Lint 오퍼레이션 | 신규 |
| `save-session.md` | 세션 저장 | 업데이트 |
| `refine.md` | MCP 그래프 업데이트 | 소폭 업데이트 |

삭제: `search.md`, `answer.md`, `research.md`

---

## Task 1: SKILL.md 재작성

**Files:**
- Modify: `SKILL.md`

- [ ] **Step 1: SKILL.md 작성**

```markdown
---
name: smart-kg-researcher
description: >
  LLM Wiki 패턴 기반 개인 지식 베이스 관리자.
  iCloud Obsidian vault의 wiki/**/*.md 파일과 MCP 지식 그래프를
  점진적으로 축적하는 영속적 위키를 구축·유지관리합니다.
  Ingest / Query / Lint / Session-Save 네 가지 오퍼레이션을 지원합니다.
user_invocable: true
triggers:
  - "지식 그래프에서 찾아줘"
  - "리서치 파일에서 찾아줘"
  - "kg에서 검색"
  - "내 노트에서"
  - "obsidian"
  - "what do I know about"
  - "지식 그래프에 추가해줘"
  - "이 대화 저장해줘"
  - "세션 저장"
  - "KG에 추가"
  - "위키에 추가"
  - "수집해줘"
  - "위키 점검"
  - "lint"
---

# Smart Knowledge-Graph Based Researcher (Wiki Edition)

## Vault 경로

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
```

이 변수를 모든 파일 접근 명령에 사용합니다.
**`.obsidian/` 디렉토리는 절대 수정하지 않습니다.**

---

## 진입 시 상태 체크

### Step 1: MCP 설정 확인

```bash
claude mcp list 2>/dev/null | grep -q "^memory" && echo "EXISTS" || echo "NOT_FOUND"
```

NOT_FOUND → `setup.md` 섹션 A 수행 후 중단.

### Step 2: 위키 초기화 확인

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
test -f "$VAULT/index.md" && echo "READY" || echo "NOT_INITIALIZED"
```

NOT_INITIALIZED → `setup.md` 섹션 B 수행 (위키 초기화 + Migration).

### Step 3: 그래프 워밍 확인

MCP `search_nodes`("")로 빈 쿼리 실행.
결과 없음 (cold start) → `setup.md` 섹션 C 수행.

---

## 워크플로우 라우팅

### 세션 저장 요청
"이 대화 저장해줘" / "세션 저장" / "KG에 추가" / "이 내용 위키에" 등
→ `save-session.md` 실행

### Ingest 요청
"이 URL 수집해줘" / "위키에 추가해줘" / "이 파일 정리해서 저장해줘" 등
→ `ingest.md` 실행

### Lint 요청
"위키 점검해줘" / "lint 돌려줘" / "고아 페이지 찾아줘" 등
→ `lint.md` 실행

### 일반 질문 (Query)
위 이외의 모든 경우 → `query.md` 실행
```

- [ ] **Step 2: 저장 확인**

```bash
head -5 /Users/minseop/repo/skills/smart-kg-researcher/SKILL.md
# "---" frontmatter로 시작하면 성공
```

- [ ] **Step 3: Commit**

```bash
git add SKILL.md
git commit -m "feat: rewrite SKILL.md for iCloud Obsidian + LLM Wiki redesign"
```

---

## Task 2: setup.md 재작성

**Files:**
- Modify: `setup.md`

- [ ] **Step 1: setup.md 작성**

```markdown
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

`memory` 항목 확인 후 Claude Code 재시작.

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

Write 도구로 `$VAULT/index.md` 생성:

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

Write 도구로 `$VAULT/log.md` 생성:

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
2. 본문에서 핵심 엔티티 추출
3. MCP `create_entities` 생성:
   ```json
   [{"name": "엔티티명", "entityType": "Technology", "observations": ["핵심 설명 1~3문장"]}]
   ```
4. 2개 이상 엔티티면 MCP `create_relations` 생성
5. frontmatter `related_nodes`, `last_synced_with_graph` 업데이트 후 Write 저장

완료 메시지:
> "그래프 워밍 완료: {N}개 파일 처리, {M}개 엔티티 생성"
```

- [ ] **Step 2: 저장 확인**

```bash
grep 'VAULT=' /Users/minseop/repo/skills/smart-kg-researcher/setup.md | head -3
# VAULT 변수 정의가 있으면 성공
```

- [ ] **Step 3: Commit**

```bash
git add setup.md
git commit -m "feat: rewrite setup.md with iCloud vault path and wiki init"
```

---

## Task 3: migrate.md 신규 작성

**Files:**
- Create: `migrate.md`

- [ ] **Step 1: migrate.md 작성**

```markdown
# Migration: research/ → wiki/ 구조 전환

기존 `$VAULT/research/**/*.md` 파일을 새 위키 구조로 복사하고
frontmatter를 Obsidian/Dataview 호환 형식으로 표준화합니다.
**원본 파일은 삭제하지 않습니다.**

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
```

---

## Migration 매핑

| 기존 경로 | 신규 경로 | type |
|---|---|---|
| `research/paimon/*.md` | `wiki/concepts/paimon/` | concept |
| `research/paimon-iceberg/*.md` | `wiki/analyses/paimon-iceberg/` | analysis |
| `research/iceberg-kafka-connect-research/*.md` | `wiki/sources/iceberg-kafka-connect/` | source |
| 기타 `research/<topic>/*.md` | `wiki/concepts/<topic>/` | concept |

---

## 처리 절차

### 1. 기존 파일 목록 수집

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
find "$VAULT/research" -name "*.md" | sort
```

### 2. 각 파일 처리

**a) 대상 경로 결정** — 매핑 규칙 적용, 파일명 유지.

**b) 디렉토리 생성**
```bash
mkdir -p "$VAULT/<신규_폴더>"
```

**c) 파일 읽기** — Read 도구로 원본 파일 읽기.

**d) frontmatter 표준화**

frontmatter가 **없는 경우** — 파일 맨 앞에 추가 (본문 수정 금지):
```yaml
---
title: "{첫 번째 # 헤딩, 없으면 파일명에서 추출}"
type: "{매핑 규칙 type}"
date_ingested: "{파일 내 날짜 주석 또는 오늘 날짜}"
tags: [tag1, tag2]
related: []
related_nodes: []
last_synced_with_graph: ""
source_count: 1
migrated_from: "research/{원본_상대경로}"
---

```

frontmatter가 **있는 경우** — 누락 필드만 추가:
- `type`, `date_ingested`, `source_count`, `migrated_from` 추가
- `related` 없으면 추가
- 기존 `title`, `tags`, `related_nodes`, `last_synced_with_graph` 유지

**e) 내부 링크 변환**

본문의 `[텍스트](상대경로.md)` 형태를 `[[파일명]]`으로 변환.
외부 URL(`http://`, `https://`)은 변환하지 않음.

**f) 파일 저장** — Write 도구로 `$VAULT/<신규 경로>` 에 저장.

### 3. 특수 파일 처리

- `research/paimon/README.md` → `wiki/concepts/paimon/README.md` (type: index)
- `research/paimon/FINAL.md` → `wiki/concepts/paimon/FINAL.md` (type: synthesis)

### 4. 완료 보고

```
Migration 완료:
- 처리된 파일: {N}개
- wiki/concepts: {N}개 / wiki/sources: {N}개 / wiki/analyses: {N}개
- frontmatter 추가: {N}개
- 원본 보존: $VAULT/research/
```

---

## 롤백

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
rm -rf "$VAULT/wiki/"
rm -f "$VAULT/index.md" "$VAULT/log.md"
```

원본 `$VAULT/research/`는 그대로 보존됩니다.
```

- [ ] **Step 2: 저장 확인**

```bash
test -f /Users/minseop/repo/skills/smart-kg-researcher/migrate.md && echo "OK"
```

- [ ] **Step 3: Commit**

```bash
git add migrate.md
git commit -m "feat: add migrate.md for research/ to wiki/ migration with iCloud path"
```

---

## Task 4: query.md 신규 작성

**Files:**
- Create: `query.md`
- Delete: `search.md`, `answer.md`

- [ ] **Step 1: query.md 작성**

```markdown
# Query 오퍼레이션

index.md를 우선 탐색하고 관련 위키 페이지와 MCP 그래프로 답변합니다.

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
```

---

## Q-1: index.md 탐색

Read 도구로 `$VAULT/index.md`를 읽습니다.

사용자 질문에서 핵심 키워드 1~3개 추출.
index.md에서 키워드와 관련된 파일 경로 수집.

**index.md가 없거나 매칭 실패 시** — Grep으로 직접 검색:
```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
grep -r "{키워드}" "$VAULT/wiki/" -l 2>/dev/null
```

## Q-2: MCP 그래프 검색

MCP `search_nodes` 도구로 각 키워드 검색.
관련 엔티티와 엣지(관계) 정보 수집.

## Q-3: 충분성 판단

| 조건 | 결과 | 다음 단계 |
|---|---|---|
| 위키 파일 ≥1 AND MCP 결과 ≥1 | 충분 | Q-4 답변 |
| 둘 중 하나만 | 부분 | Q-4 답변 (부분 커버리지 명시) |
| 둘 다 없음 | 불충분 | Q-5 웹 리서치 제안 |

## Q-4: 위키 페이지 읽기 + 답변

수집된 파일을 Read 도구로 읽어 내용 확인.

**답변 형식:**

출처 (첫 줄):
- 충분: `> 위키 기반 답변입니다.`
- 부분: `> 위키 기반 답변입니다. (일부 정보만 존재합니다)`

본문: 위키 페이지 + MCP 그래프 노드 종합.

그래프 통찰 (엣지 있을 때):
`> 이 주제는 [[엔티티 A]]와 {관계 유형}으로 연결되어 있습니다.`

참조 파일 (마지막):
```
참조: [[wiki/concepts/paimon/01-concepts-overview]]
```

**답변 후**: 중요한 통찰이 있으면 위키 저장 제안:
> "이 답변을 위키에 저장할까요? (`wiki/analyses/{slug}.md`)"
> 동의 시 `ingest.md`의 파일 저장 절차 실행.

## Q-5: 웹 리서치 제안

> "위키에 **{질문 주제}** 정보가 부족합니다. 웹 리서치를 수행하고 위키에 추가할까요?"

거절 → 부분 정보와 함께 "정보 부족" 답변.
동의 → `ingest.md` 실행.
```

- [ ] **Step 2: 기존 파일 삭제**

```bash
rm /Users/minseop/repo/skills/smart-kg-researcher/search.md
rm /Users/minseop/repo/skills/smart-kg-researcher/answer.md
```

- [ ] **Step 3: 확인**

```bash
test -f /Users/minseop/repo/skills/smart-kg-researcher/query.md && echo "query.md OK"
test ! -f /Users/minseop/repo/skills/smart-kg-researcher/search.md && echo "removed OK"
test ! -f /Users/minseop/repo/skills/smart-kg-researcher/answer.md && echo "removed OK"
```

- [ ] **Step 4: Commit**

```bash
git add query.md && git rm search.md answer.md
git commit -m "feat: add query.md (index-first), remove search.md + answer.md"
```

---

## Task 5: ingest.md 신규 작성

**Files:**
- Create: `ingest.md`
- Delete: `research.md`

- [ ] **Step 1: ingest.md 작성**

```markdown
# Ingest 오퍼레이션

새로운 소스를 읽고 위키에 통합합니다.
하나의 소스가 여러 위키 페이지에 영향을 줄 수 있습니다.

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
```

---

## I-1: 소스 읽기

| 유형 | 방법 |
|---|---|
| URL | WebFetch 도구로 내용 수집 |
| 로컬 파일 (`$VAULT/raw/`) | Read 도구로 읽기 |
| 웹 리서치 필요 | WebSearch 2~3회 → WebFetch 상위 2~3개 |
| 대화 내용 | 현재 세션 분석 |

## I-2: 핵심 내용 추출

- 주요 기술/시스템/개념 이름
- 핵심 사실과 인사이트 3~7개
- index.md 대조: 기존 위키 어떤 페이지와 관련되는가
- 새 엔티티가 있는가 (위키에 없는 기술명)

## I-3: 저장 경로 결정

**기존 관련 페이지가 있는 경우** — 기존 페이지에 내용 추가할지 별도 페이지 생성할지 결정.

**새 주제인 경우:**
- 개념·기능 설명 → `wiki/concepts/<topic>/`
- 특정 소스 요약 → `wiki/sources/<topic>/`
- 기술·시스템 엔티티 → `wiki/entities/`
- 종합·비교 분석 → `wiki/analyses/`

파일명: 영어 kebab-case, zero-padding 인덱스
```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
COUNT=$(ls "$VAULT/wiki/<type>/<topic>/"*.md 2>/dev/null | wc -l)
INDEX=$(printf "%02d" $((COUNT + 1)))
# 예: 03-paimon-compaction-deep-dive.md
```

## I-4: 파일 저장

Write 도구로 저장. frontmatter 형식:

```yaml
---
title: "{소스/주제 제목}"
type: "{source | concept | entity | analysis}"
source_url: "{URL, 없으면 생략}"
date_ingested: "{오늘 날짜 YYYY-MM-DD}"
tags: [tag1, tag2]
related: ["[[관련-페이지-제목]]"]
related_nodes: []
last_synced_with_graph: ""
source_count: 1
---

# {제목}

{마크다운 본문}
```

**내부 링크는 반드시 `[[wikilink]]` 형식으로 작성합니다.**
Mermaid 다이어그램은 3개 이상 컴포넌트/흐름이 있을 때만 포함합니다.

## I-5: 기존 페이지 업데이트

새 소스가 기존 위키 페이지 내용을 보완하는 경우:
1. Read 도구로 해당 페이지 읽기
2. 새 내용을 관련 섹션에 추가
3. `source_count` 1 증가
4. `related`에 새 소스 `[[링크]]` 추가
5. Write 도구로 저장

## I-6: log.md 업데이트

log.md 끝에 append:

```markdown
## [{오늘 날짜}] ingest | {소스 제목}

- 저장: `wiki/<type>/<topic>/파일명.md`
- 업데이트된 기존 페이지: {N}개
- 새 엔티티: {기술명 목록}
```

## I-7: MCP 그래프 업데이트

`refine.md`를 Read 도구로 읽어 새로 저장된 파일들의 MCP 그래프 업데이트.

---

## 완료 메시지

```
Ingest 완료: {소스 제목}
- 새 파일: $VAULT/{경로}.md  ← Obsidian에서 실시간 확인 가능
- 업데이트된 기존 페이지: {N}개
- log.md 업데이트 ✓
- MCP 엔티티 {N}개 추가/업데이트
```
```

- [ ] **Step 2: research.md 삭제**

```bash
rm /Users/minseop/repo/skills/smart-kg-researcher/research.md
```

- [ ] **Step 3: 확인**

```bash
test -f /Users/minseop/repo/skills/smart-kg-researcher/ingest.md && echo "OK"
test ! -f /Users/minseop/repo/skills/smart-kg-researcher/research.md && echo "removed OK"
```

- [ ] **Step 4: Commit**

```bash
git add ingest.md && git rm research.md
git commit -m "feat: add ingest.md (LLM Wiki ingest op), remove research.md"
```

---

## Task 6: lint.md 신규 작성

**Files:**
- Create: `lint.md`

- [ ] **Step 1: lint.md 작성**

```markdown
# Lint 오퍼레이션: 위키 상태 점검

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
```

---

## 점검 항목

### L-1: 깨진 `[[wikilink]]`

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
grep -r "\[\[" "$VAULT/wiki/" --include="*.md" -h | \
  grep -oP '\[\[([^\]|]+)' | sed 's/\[\[//' | sort -u | \
  while read link; do
    find "$VAULT/wiki" -name "${link}.md" 2>/dev/null | grep -q . || echo "BROKEN: [[${link}]]"
  done
```

### L-2: 고아 페이지

다른 페이지에서 한 번도 `[[링크]]`되지 않은 페이지:

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
find "$VAULT/wiki" -name "*.md" | while read f; do
  name=$(basename "$f" .md)
  count=$(grep -r "\[\[$name" "$VAULT/wiki/" --include="*.md" -l | grep -v "^$f$" | wc -l)
  [ "$count" -eq 0 ] && echo "ORPHAN: $f"
done
```

### L-3: 빈약한 페이지

`source_count ≥ 3`이지만 본문이 300자 미만:

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
find "$VAULT/wiki" -name "*.md" | while read f; do
  sc=$(grep "^source_count:" "$f" 2>/dev/null | awk '{print $2}')
  [ "${sc:-0}" -ge 3 ] || continue
  chars=$(wc -c < "$f")
  [ "$chars" -lt 300 ] && echo "THIN: $f (source_count=$sc)"
done
```

### L-4: 개념 누락

MCP `search_nodes` 도구로 엔티티 목록 조회.
위키에 자체 페이지가 없는 주요 엔티티 목록 출력.

### L-5: 오래된 내용

`date_ingested`가 6개월 이상 지나고 `source_count ≥ 5`인 중요 페이지:

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
CUTOFF=$(date -v-6m +%Y-%m-%d 2>/dev/null || date -d "6 months ago" +%Y-%m-%d)
grep -r "^date_ingested:" "$VAULT/wiki/" --include="*.md" -l | while read f; do
  d=$(grep "^date_ingested:" "$f" | awk '{print $2}')
  sc=$(grep "^source_count:" "$f" | awk '{print $2}')
  [ "${sc:-0}" -ge 5 ] && [ "$d" \< "$CUTOFF" ] && echo "STALE: $f ($d)"
done
```

---

## 점검 결과 보고

각 항목: 발견 수 + 파일 목록 + 수정 제안.

"자동 수정해줘" 시 L-1(깨진 링크 제거), 고아 페이지 index 추가만 자동 수정.
나머지는 내용 판단이 필요하므로 사용자 확인 후 수정.
```

- [ ] **Step 2: 확인**

```bash
test -f /Users/minseop/repo/skills/smart-kg-researcher/lint.md && echo "OK"
```

- [ ] **Step 3: Commit**

```bash
git add lint.md
git commit -m "feat: add lint.md wiki health check with iCloud vault path"
```

---

## Task 7: save-session.md 업데이트

**Files:**
- Modify: `save-session.md`

- [ ] **Step 1: 파일 상단에 VAULT 변수 추가**

save-session.md 맨 위에 추가:

```markdown
```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
```
```

- [ ] **Step 2: 저장 경로 업데이트**

기존 `~/knowledge/research/<주제명>/` → `$VAULT/wiki/<type>/<주제명>/`

- [ ] **Step 3: frontmatter 형식 교체**

기존:
```yaml
---
title: "주제 제목"
tags: [tag1, tag2]
related_nodes: []
last_synced_with_graph: ""
session_date: "YYYY-MM-DD"
---
```

신규:
```yaml
---
title: "주제 제목"
type: concept
date_ingested: "YYYY-MM-DD"
tags: [tag1, tag2]
related: []
related_nodes: []
last_synced_with_graph: ""
source_count: 1
---
```

- [ ] **Step 4: 완료 섹션 뒤에 log.md 업데이트 추가**

"### 6. 완료 메시지 출력" 뒤에 추가:

```markdown
### 7. log.md 업데이트

log.md 끝에 append:

```markdown
## [{오늘 날짜}] session-save | {주요 주제 1~2개}

- 저장: `{파일 경로 또는 "파일 저장 생략"}`
- KG 엔티티 추가/업데이트: {N}개
```
```

- [ ] **Step 5: 확인**

```bash
grep 'VAULT=' /Users/minseop/repo/skills/smart-kg-researcher/save-session.md | wc -l
# 1 이상이면 성공
```

- [ ] **Step 6: Commit**

```bash
git add save-session.md
git commit -m "feat: update save-session.md with iCloud vault path and log.md sync"
```

---

## Task 8: refine.md 소폭 업데이트

**Files:**
- Modify: `refine.md`

- [ ] **Step 1: 파일 상단에 VAULT 변수 추가 + frontmatter 보존 규칙 추가**

refine.md 맨 위에 추가:
```markdown
```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
```
```

6번 항목 (`related_nodes`, `last_synced_with_graph` 업데이트) 뒤에 추가:
```markdown
   - `source_count`: 이미 있으면 그대로 유지 (refine은 변경하지 않음)
   - `related`: 기존 `[[링크]]` 목록 그대로 유지
```

- [ ] **Step 2: 확인**

```bash
grep 'VAULT=' /Users/minseop/repo/skills/smart-kg-researcher/refine.md
```

- [ ] **Step 3: Commit**

```bash
git add refine.md
git commit -m "fix: refine.md add iCloud vault path and preserve source_count/related"
```

---

## Task 9: 실제 Migration 실행

> **주의:** 실제 사용자 데이터 변경. 원본 파일은 삭제하지 않음.

`$VAULT/research/` 아래 **모든 서브폴더**를 자동 감지하여 migration.

**현재 폴더 목록 (총 41개 파일):**
```
research/paimon/                      → wiki/concepts/paimon/          (31개, concept)
research/iceberg-kafka-connect-research/ → wiki/sources/iceberg-kafka-connect/ (6개, source)
research/paimon-iceberg/              → wiki/analyses/paimon-iceberg/   (1개, analysis)
research/claude-code/                 → wiki/concepts/claude-code/      (1개, concept)
research/eks-logging/                 → wiki/sources/eks-logging/       (1개, source)
research/starrocks/                   → wiki/concepts/starrocks/        (1개, concept)
```

**type 판단 규칙 (신규 폴더에도 적용):**
- 폴더명에 `-research`, `-study` 포함 → `source`
- 폴더명이 두 기술명 조합 (예: `paimon-iceberg`, `kafka-iceberg`) → `analysis`
- 그 외 → `concept`

**Files:**
- Create: `$VAULT/wiki/**/*.md` (research/ 하위 전체)
- Create: `$VAULT/index.md`
- Create: `$VAULT/log.md`

- [ ] **Step 1: 디렉토리 생성**

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
mkdir -p "$VAULT/wiki/concepts/paimon"
mkdir -p "$VAULT/wiki/concepts/claude-code"
mkdir -p "$VAULT/wiki/concepts/starrocks"
mkdir -p "$VAULT/wiki/sources/iceberg-kafka-connect"
mkdir -p "$VAULT/wiki/sources/eks-logging"
mkdir -p "$VAULT/wiki/analyses/paimon-iceberg"
mkdir -p "$VAULT/wiki/entities"
mkdir -p "$VAULT/raw"
```

- [ ] **Step 2: paimon 파일 복사 + frontmatter 표준화 (31개)**

`$VAULT/research/paimon/` 전체 파일을 Glob으로 수집.
`paimon/org/` 서브폴더(`.class` 파일)는 건너뜁니다.

`README.md` → type: `index`, `FINAL.md` → type: `synthesis`, 나머지 → type: `concept`.

각 파일 Read → frontmatter 추가 → `$VAULT/wiki/concepts/paimon/{파일명}` Write.

frontmatter 없는 파일 기준 (tags는 본문에서 실제 기술명 추출):
```yaml
---
title: "{첫 번째 # 헤딩}"
type: concept
date_ingested: "2026-03-01"
tags: [apache-paimon, lakehouse, data-engineering]
related: []
related_nodes: []
last_synced_with_graph: ""
source_count: 1
migrated_from: "research/paimon/{파일명}"
---
```

- [ ] **Step 3: iceberg-kafka-connect 파일 복사 + frontmatter 표준화 (6개)**

`$VAULT/research/iceberg-kafka-connect-research/*.md` 처리.

```yaml
---
title: "{헤딩}"
type: source
date_ingested: "2026-03-21"
tags: [apache-iceberg, kafka-connect, data-engineering]
related: ["[[wiki/concepts/paimon/01-concepts-overview]]"]
related_nodes: []
last_synced_with_graph: ""
source_count: 1
migrated_from: "research/iceberg-kafka-connect-research/{파일명}"
---
```

저장: `$VAULT/wiki/sources/iceberg-kafka-connect/{파일명}`

- [ ] **Step 4: paimon-iceberg 파일 복사 + frontmatter 표준화 (1개)**

`$VAULT/research/paimon-iceberg/*.md` 처리.

```yaml
---
title: "{헤딩}"
type: analysis
date_ingested: "2026-04-01"
tags: [apache-paimon, apache-iceberg, compatibility, analysis]
related: ["[[wiki/concepts/paimon/01-concepts-overview]]", "[[wiki/sources/iceberg-kafka-connect/03-apache-iceberg]]"]
related_nodes: []
last_synced_with_graph: ""
source_count: 1
migrated_from: "research/paimon-iceberg/{파일명}"
---
```

저장: `$VAULT/wiki/analyses/paimon-iceberg/{파일명}`

- [ ] **Step 5: claude-code 파일 복사 + frontmatter 표준화 (1개)**

`$VAULT/research/claude-code/*.md` 처리.

```yaml
---
title: "{헤딩}"
type: concept
date_ingested: "2026-04-05"
tags: [claude-code, developer-tools, ai]
related: []
related_nodes: []
last_synced_with_graph: ""
source_count: 1
migrated_from: "research/claude-code/{파일명}"
---
```

저장: `$VAULT/wiki/concepts/claude-code/{파일명}`

- [ ] **Step 6: eks-logging 파일 복사 + frontmatter 표준화 (1개)**

`$VAULT/research/eks-logging/*.md` 처리.

```yaml
---
title: "{헤딩}"
type: source
date_ingested: "2026-04-05"
tags: [eks, kubernetes, logging, filebeat]
related: []
related_nodes: []
last_synced_with_graph: ""
source_count: 1
migrated_from: "research/eks-logging/{파일명}"
---
```

저장: `$VAULT/wiki/sources/eks-logging/{파일명}`

- [ ] **Step 7: starrocks 파일 복사 + frontmatter 표준화 (1개)**

`$VAULT/research/starrocks/*.md` 처리.

```yaml
---
title: "{헤딩}"
type: concept
date_ingested: "2026-04-05"
tags: [starrocks, olap, data-engineering]
related: []
related_nodes: []
last_synced_with_graph: ""
source_count: 1
migrated_from: "research/starrocks/{파일명}"
---
```

저장: `$VAULT/wiki/concepts/starrocks/{파일명}`

- [ ] **Step 5: index.md 생성 (Dataview 방식)**

Write 도구로 `$VAULT/index.md` 생성 — setup.md B-3 형식 사용.

- [ ] **Step 6: log.md 생성**

Write 도구로 `$VAULT/log.md` 생성 — setup.md B-4 형식, migration 첫 항목 포함.

- [ ] **Step 7: 결과 확인**

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
find "$VAULT/wiki" -name "*.md" | wc -l
# Expected: ≥ 35

test -f "$VAULT/index.md" && echo "index.md OK"
test -f "$VAULT/log.md" && echo "log.md OK"

# frontmatter 확인 (샘플)
head -12 "$VAULT/wiki/concepts/paimon/01-concepts-overview.md"
# Expected: --- 로 시작하는 YAML frontmatter
```

---

## Task 10: 최종 검증 + 스킬 동기화

- [ ] **Step 1: 스킬 파일 목록 확인**

```bash
ls /Users/minseop/repo/skills/smart-kg-researcher/
# Expected: SKILL.md setup.md migrate.md query.md ingest.md lint.md save-session.md refine.md docs/
```

삭제 확인:
```bash
test ! -f /Users/minseop/repo/skills/smart-kg-researcher/search.md && echo "OK"
test ! -f /Users/minseop/repo/skills/smart-kg-researcher/answer.md && echo "OK"
test ! -f /Users/minseop/repo/skills/smart-kg-researcher/research.md && echo "OK"
```

- [ ] **Step 2: 설치된 스킬과 repo 동기화**

```bash
diff /Users/minseop/.claude/skills/smart-kg-researcher/SKILL.md \
     /Users/minseop/repo/skills/smart-kg-researcher/SKILL.md
```

차이가 있으면:
```bash
cp /Users/minseop/repo/skills/smart-kg-researcher/*.md \
   /Users/minseop/.claude/skills/smart-kg-researcher/
```

- [ ] **Step 3: 최종 commit**

```bash
git add -A
git commit -m "feat: complete smart-kg-researcher Obsidian + LLM Wiki redesign (iCloud vault)

- SKILL.md: Ingest/Query/Lint/Session-Save 라우팅, VAULT 변수, .obsidian/ 보호
- setup.md: iCloud vault 초기화 + Dataview index.md
- migrate.md: research/ → wiki/ 무손실 migration (wikilink 변환 포함)
- query.md: index.md 우선 탐색, wikilink 참조 답변
- ingest.md: LLM Wiki ingest 오퍼레이션
- lint.md: wikilink 기반 고아 페이지 감지
- save-session.md: Dataview frontmatter + log.md 연동
- refine.md: source_count/related wikilink 보존"
```

---

## 구현 완료 기준

- [ ] 스킬 파일 8개 존재, 구 파일 3개(search/answer/research) 삭제
- [ ] 모든 스킬 파일에서 `VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"` 사용
- [ ] `$VAULT/wiki/` 에 기존 research 파일 ≥35개 (frontmatter 포함)
- [ ] `$VAULT/index.md` Dataview 쿼리 방식으로 생성
- [ ] `$VAULT/log.md` migration 항목 포함하여 생성
- [ ] 원본 `$VAULT/research/` 파일 보존
- [ ] 모든 새 위키 파일 내부 링크가 `[[wikilink]]` 형식
- [ ] `/Users/minseop/.claude/skills/smart-kg-researcher/` 와 repo 동기화
