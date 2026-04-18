# obsidian-ingester Query & Ingest 개선 설계

Date: 2026-04-18  
Scope: `query.md`, `ingest.md` 2개 파일  
Wiki size assumption: <100 files

---

## 문제 정의

### Query
- **현상:** 문서가 존재하는데도 MCP `search_nodes`가 찾지 못해 grep fallback으로 이어짐
- **원인:** 탐색 순서가 index.md(Dataview 미실행, stale) → MCP(warmup 품질에 의존) → grep 순이라 실제로 신뢰할 수 있는 grep이 항상 마지막 수단이 됨

### Ingest
- **현상:** 저장 경로(concepts/entities/sources/analyses) 결정이 매번 애매함
- **원인:** I-3 경로 결정 기준이 텍스트 설명만 있고 판단 예시가 없음

---

## 설계

### Query 재설계

탐색 순서를 아래와 같이 변경합니다.

**변경 전:**
```
Q-1: index.md 읽기 → 키워드 매칭
Q-2: MCP search_nodes
Q-3: (실패 시) grep -r fallback
```

**변경 후:**
```
Q-1: Grep으로 wiki/**/*.md 키워드 검색 → 파일 후보 수집
Q-2: MCP search_nodes → 엔티티·관계 정보 보완
Q-3: 충분성 판단
Q-4: 파일 읽기 + 답변 (MCP 관계 정보를 "연관 개념" 섹션으로 포함)
Q-5: 웹 리서치 제안 (기존 동일)
```

**핵심 변경 이유:**
- grep은 <100 파일 규모에서 항상 신뢰할 수 있는 O(N×F) 탐색
- MCP는 파일 탐색이 아닌 "관계 통찰" 역할로 재정의 (warmup 품질과 무관하게 동작)
- index.md 의존성 제거 (Dataview가 없으면 LLM이 활용 불가)

**답변 형식 변경:**
- Grep 결과 파일 수 명시: `> {N}개 파일에서 키워드 발견`
- MCP 관계 정보가 있을 때만 "연관 개념" 섹션 추가

---

### Ingest 경로 결정 개선

I-3 섹션에 아래 판단 테이블과 예시를 추가합니다.

| 카테고리 | 핵심 질문 | 예시 주제 |
|----------|-----------|-----------|
| `wiki/concepts/<topic>/` | "X가 **무엇이고 어떻게 작동**하는가?" | LSM-tree 원리, Kafka 파티셔닝, MVCC |
| `wiki/sources/<topic>/` | "이 **특정 글·문서·영상**에서 뭘 배웠는가?" | 공식문서 특정 섹션 요약, 블로그 포스트 정리 |
| `wiki/entities/` | "X라는 기술의 **기본 정보 카드**가 필요한가?" | Apache Flink, Apache Paimon, Debezium |
| `wiki/analyses/` | "여러 소스를 **종합·비교한 내 판단**인가?" | Paimon vs Iceberg, CDC 툴 비교 |

**모호할 때 판단 규칙:**
- 소스 URL이 있으면 → `sources/`
- 기술 이름 하나가 주제면 → `entities/`
- "X란 무엇인가" 형태의 설명이면 → `concepts/`
- 두 개 이상 기술을 비교하면 → `analyses/`

---

## 변경 범위

| 파일 | 변경 내용 |
|------|-----------|
| `query.md` | Q-1~Q-4 전면 재작성, Q-5 유지 |
| `ingest.md` | I-3에 판단 테이블·규칙 추가, 나머지 유지 |

**변경하지 않는 파일:** `setup.md`, `refine.md`, `save-session.md`, `lint.md`, `SKILL.md`

---

## 성공 기준

- Query 시 grep이 1번 경로로 동작해 "파일이 있는데 못 찾는" 상황이 없어짐
- Ingest 시 경로 결정에 판단 테이블을 참조해 애매함이 줄어듦
