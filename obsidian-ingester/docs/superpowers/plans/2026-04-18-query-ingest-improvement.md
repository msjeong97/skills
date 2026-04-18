# Query & Ingest 개선 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Query 탐색을 grep 우선으로 바꾸고, Ingest 경로 결정에 판단 테이블을 추가해 "파일이 있는데 못 찾는" 문제와 "경로 결정 애매함" 문제를 해결한다.

**Architecture:** `query.md`의 Q-1~Q-4를 재작성해 Grep을 1번 탐색 경로로 승격하고 MCP를 관계 보완용으로 재정의한다. `ingest.md`의 I-3에 카테고리 판단 테이블과 모호할 때의 결정 규칙을 추가한다.

**Tech Stack:** Markdown skill files, Grep tool, MCP memory server (`search_nodes`)

---

### Task 1: query.md — Q-1~Q-4 재작성

**Files:**
- Modify: `/Users/minseop/.claude/skills/obsidian-ingester/query.md`

- [ ] **Step 1: 현재 파일 읽기**

Read 도구로 `/Users/minseop/.claude/skills/obsidian-ingester/query.md` 읽기.
(내용 숙지 후 Step 2 진행)

- [ ] **Step 2: Q-1~Q-4 전면 교체**

`query.md`의 `## Q-1: index.md 탐색` 섹션부터 `## Q-5:` 직전까지를 아래 내용으로 교체한다.

```markdown
## Q-1: Grep 검색

사용자 질문에서 핵심 키워드 1~3개 추출.

각 키워드에 대해 Grep 도구로 검색:
- path: `$VAULT/wiki/`
- glob: `*.md`
- output_mode: `files_with_matches`

결과: 키워드가 포함된 파일 경로 목록 수집.
중복 제거 후 최대 10개 파일 후보 유지.

**결과가 없는 경우** → Q-3 불충분으로 이동.

## Q-2: MCP 그래프 검색

MCP `search_nodes` 도구로 각 키워드 검색.
관련 엔티티와 관계(엣지) 정보 수집.

**목적:** 파일 탐색이 아닌 "연관 개념·관계" 파악용.
Q-1 결과와 무관하게 항상 실행.

## Q-3: 충분성 판단

| 조건 | 결과 | 다음 단계 |
|---|---|---|
| Grep 파일 ≥1 AND MCP 결과 ≥1 | 충분 | Q-4 답변 |
| Grep 파일 ≥1, MCP 없음 | 부분 | Q-4 답변 (파일 기반) |
| Grep 없음, MCP ≥1 | 부분 | Q-4 답변 (KG 기반) |
| 둘 다 없음 | 불충분 | Q-5 웹 리서치 제안 |

## Q-4: 위키 페이지 읽기 + 답변

Q-1에서 수집된 파일을 Read 도구로 읽어 내용 확인.

**답변 형식:**

출처 (첫 줄):
- 충분: `> 위키 기반 답변입니다. ({N}개 파일 참조)`
- 부분: `> 위키 기반 답변입니다. (일부 정보만 존재합니다)`

본문: 위키 페이지 내용 종합.

연관 개념 (MCP 엣지가 있을 때만):
`> 이 주제는 [[엔티티 A]]와 {관계 유형}으로 연결되어 있습니다.`

참조 파일 (마지막):
\```
참조: [[wiki/concepts/paimon/01-concepts-overview]]
\```

**답변 후**: 중요한 통찰이 있으면 위키 저장 제안:
> "이 답변을 위키에 저장할까요? (`wiki/analyses/{slug}.md`)"
> 동의 시 `ingest.md`를 Read 도구로 읽어 파일 저장 절차 실행.
```

- [ ] **Step 3: 파일 저장 확인**

수정된 `query.md`를 Read 도구로 다시 열어 Q-1~Q-4가 올바르게 교체됐는지 확인.
Q-5 섹션(웹 리서치 제안)이 그대로 남아있는지 확인.

- [ ] **Step 4: 커밋**

```bash
cd /Users/minseop/.claude/skills/obsidian-ingester
git add query.md
git commit -m "refactor(query): promote grep to primary search, MCP as relation supplement"
```

---

### Task 2: ingest.md — I-3 경로 결정 로직 강화

**Files:**
- Modify: `/Users/minseop/.claude/skills/obsidian-ingester/ingest.md`

- [ ] **Step 1: 현재 파일 읽기**

Read 도구로 `/Users/minseop/.claude/skills/obsidian-ingester/ingest.md` 읽기.

- [ ] **Step 2: I-3 섹션에 판단 테이블 삽입**

I-3의 `**새 주제인 경우:**` 줄 바로 앞에 아래 내용을 삽입한다.

```markdown
**카테고리 판단 테이블:**

| 카테고리 | 핵심 질문 | 예시 주제 |
|----------|-----------|-----------|
| `wiki/concepts/<topic>/` | "X가 **무엇이고 어떻게 작동**하는가?" | LSM-tree 원리, Kafka 파티셔닝, MVCC |
| `wiki/sources/<topic>/` | "이 **특정 글·문서·영상**에서 뭘 배웠는가?" | 공식문서 섹션 요약, 블로그 포스트 정리 |
| `wiki/entities/` | "X라는 기술의 **기본 정보 카드**가 필요한가?" | Apache Flink, Apache Paimon, Debezium |
| `wiki/analyses/` | "여러 소스를 **종합·비교한 내 판단**인가?" | Paimon vs Iceberg 비교, CDC 툴 선택 |

**모호할 때 판단 규칙 (위에서부터 순서대로 확인):**
1. 소스 URL이 있다 → `sources/`
2. 기술 이름 하나가 주제다 → `entities/`
3. "X란 무엇인가" 형태의 설명이다 → `concepts/`
4. 두 개 이상 기술을 비교한다 → `analyses/`

```

- [ ] **Step 3: 파일 저장 확인**

수정된 `ingest.md`를 Read 도구로 다시 열어 판단 테이블이 I-3 섹션 내 올바른 위치에 삽입됐는지 확인.
I-1, I-2, I-4~I-7 섹션이 그대로 남아있는지 확인.

- [ ] **Step 4: 커밋**

```bash
cd /Users/minseop/.claude/skills/obsidian-ingester
git add ingest.md
git commit -m "feat(ingest): add category decision table and disambiguation rules to I-3"
```

---

### Task 3: 검증

- [ ] **Step 1: 두 파일 최종 확인**

Read 도구로 `query.md`, `ingest.md` 순서대로 읽어 변경사항 최종 확인.

체크리스트:
- `query.md`: Q-1이 Grep 검색으로 시작하는가?
- `query.md`: Q-3 충분성 판단 테이블에 4가지 조건이 있는가?
- `query.md`: Q-5 웹 리서치 제안 섹션이 그대로인가?
- `ingest.md`: I-3에 판단 테이블(4행)이 있는가?
- `ingest.md`: I-3에 모호할 때 판단 규칙(4개)이 있는가?
- `ingest.md`: I-4~I-7 섹션이 그대로인가?

- [ ] **Step 2: SKILL.md 라우팅 확인**

Read 도구로 `SKILL.md` 읽어 "일반 질문(Query)" 라우팅이 `query.md`를 가리키는지, "Ingest 요청" 라우팅이 `ingest.md`를 가리키는지 확인. (변경 불필요, 확인만)
