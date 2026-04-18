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

아래 테이블로 카테고리를 결정한다. 모호하면 판단 규칙을 순서대로 적용.

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

**기존 관련 페이지가 있는 경우** — 기존 페이지에 내용 추가할지 별도 페이지 생성할지 결정.

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
