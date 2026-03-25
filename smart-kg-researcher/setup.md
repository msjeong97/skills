# Phase 0: MCP 설정 및 그래프 워밍

이 파일은 두 가지 경우에 읽힙니다:
1. MCP가 설정되지 않은 경우 → 섹션 A 수행
2. 그래프가 비어있는 경우 → 섹션 B 수행

---

## 섹션 A: MCP 설정 주입

Claude Code에서 MCP 서버는 `~/.claude/settings.json`이 아닌 `~/.claude.json`에 등록합니다.
가장 안전한 방법은 `claude mcp add` CLI 명령어를 사용하는 것입니다.

다음 Bash 명령어를 실행합니다:

```bash
claude mcp add -s user memory -- npx -y @modelcontextprotocol/server-memory@latest
```

등록 확인:
```bash
claude mcp list
```

`memory` 항목이 나타나면 성공입니다. Claude Code를 재시작한 후 다시 질문해 주세요.

---

## 섹션 B: 그래프 워밍 (cold start 시)

`~/knowledge/` 내 모든 `.md` 파일 경로를 Glob 도구로 수집합니다.
파일이 없으면 "워밍할 파일이 없습니다."를 출력하고 종료합니다.

### 스킵 조건

- **cold start (그래프 완전히 비어있음):** 모든 파일을 처리합니다. 날짜 무관.
- **세션 중 재실행 (그래프에 이미 노드가 있음):** `last_synced_with_graph`가 오늘 날짜인 파일은 건너뜁니다.

### 파일별 처리 절차

각 `.md` 파일에 대해 순서대로 수행합니다:

1. Read 도구로 파일 내용을 읽습니다.
2. 파일 첫 줄이 `---`로 시작하는 frontmatter가 없으면:
   - 첫 번째 `#` 헤딩을 `title`로 사용합니다. 없으면 파일명(확장자 제외)을 사용합니다.
   - 본문에서 기술명/개념명을 최대 5개 추출해 `tags`로 사용합니다.
   - 파일 맨 앞에 아래 frontmatter를 prepend합니다 **(본문은 절대 수정하지 않음)**:
     ```yaml
     ---
     title: "추출된 제목"
     tags: [tag1, tag2]
     related_nodes: []
     last_synced_with_graph: ""
     ---

     ```
3. 본문에서 핵심 엔티티를 추출합니다 (기술명, 시스템명, 개념명).
   - 추출된 엔티티가 없으면 `title` 값을 단일 엔티티로 사용합니다.
4. MCP `create_entities` 도구로 엔티티를 생성합니다:
   ```json
   [
     {
       "name": "엔티티명",
       "entityType": "Technology",
       "observations": ["핵심 설명 1~3문장"]
     }
   ]
   ```
   `entityType`은 Technology / Concept / System 중 가장 적합한 것을 선택합니다.
5. 같은 파일에 2개 이상의 엔티티가 있으면 MCP `create_relations` 도구로 관계를 생성합니다:
   ```json
   [
     {
       "from": "엔티티 A",
       "to": "엔티티 B",
       "relationType": "relates_to"
     }
   ]
   ```
   명확한 관계(예: "integrates_with", "builds_on", "replaces")가 있으면 더 구체적으로 표현합니다.
6. 생성된 엔티티 이름 목록을 수집합니다.
7. 파일의 frontmatter에서 `related_nodes`와 `last_synced_with_graph`를 업데이트한 후 Write 도구로 저장합니다:
   - `related_nodes`: 엔티티 이름 목록
   - `last_synced_with_graph`: 오늘 날짜 (YYYY-MM-DD)

### 완료 메시지

모든 파일 처리 후:
> "그래프 워밍 완료: {N}개 파일 처리, {M}개 엔티티 생성"
