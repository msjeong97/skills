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
> 동의 시 `ingest.md`를 Read 도구로 읽어 파일 저장 절차 실행.

## Q-5: 웹 리서치 제안

> "위키에 **{질문 주제}** 정보가 부족합니다. 웹 리서치를 수행하고 위키에 추가할까요?"

거절 → 부분 정보와 함께 "정보 부족" 답변.
동의 → `ingest.md`를 Read 도구로 읽어 실행.
