# Phase 4: 지식 그래프 업데이트

Phase 3에서 저장된 파일 경로를 받아 MCP 그래프를 업데이트합니다.

---

## 절차

1. Read 도구로 저장된 파일을 읽습니다.

2. 본문에서 핵심 엔티티를 추출합니다 (기술명, 시스템명, 개념명).
   - 엔티티가 없으면 frontmatter의 `title` 값을 단일 엔티티로 사용합니다.

3. MCP `create_entities` 도구로 엔티티를 생성합니다:
   ```json
   [
     {
       "name": "엔티티명",
       "entityType": "Technology",
       "observations": ["핵심 설명 1~3문장"]
     }
   ]
   ```
   `entityType`: Technology / Concept / System 중 선택.

4. 같은 파일에 2개 이상의 엔티티가 있으면 MCP `create_relations` 도구로 관계를 생성합니다:
   ```json
   [
     {
       "from": "엔티티 A",
       "to": "엔티티 B",
       "relationType": "relates_to"
     }
   ]
   ```
   - 기존 관계는 절대 삭제하지 않습니다. 새 관계만 추가합니다.
   - 명확한 관계 유형이 있으면 구체적으로 표현합니다 (예: "integrates_with", "builds_on", "extends").

5. 생성된 엔티티 이름 목록을 수집합니다.

6. 파일의 frontmatter를 업데이트하여 Write 도구로 저장합니다:
   - `related_nodes`: 엔티티 이름 목록
   - `last_synced_with_graph`: 오늘 날짜 (YYYY-MM-DD)

---

## 실패 처리

어느 단계에서든 오류가 발생하면:
- 오류 내용을 사용자에게 알립니다.
- 파일은 Phase 3에서 저장된 상태 그대로 둡니다 (`related_nodes: []`, `last_synced_with_graph: ""`).
- 다음 cold start 시 warm-up이 이 파일을 자동으로 임포트합니다.
