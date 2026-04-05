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

NOT_FOUND → `setup.md`를 Read 도구로 읽어 **섹션 A** 수행 후 중단.

### Step 2: 위키 초기화 확인

```bash
VAULT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Obsidian"
test -f "$VAULT/index.md" && echo "READY" || echo "NOT_INITIALIZED"
```

NOT_INITIALIZED → `setup.md`를 Read 도구로 읽어 **섹션 B** 수행.

### Step 3: 그래프 워밍 확인

MCP `search_nodes`("")로 빈 쿼리 실행.
결과 없음 (cold start) → `setup.md`를 Read 도구로 읽어 **섹션 C** 수행.

---

## 워크플로우 라우팅

### 세션 저장 요청
"이 대화 저장해줘" / "세션 저장" / "KG에 추가" / "이 내용 위키에" 등
→ `save-session.md`를 Read 도구로 읽어 실행

### Ingest 요청
"이 URL 수집해줘" / "위키에 추가해줘" / "이 파일 정리해서 저장해줘" 등
→ `ingest.md`를 Read 도구로 읽어 실행

### Lint 요청
"위키 점검해줘" / "lint 돌려줘" / "고아 페이지 찾아줘" 등
→ `lint.md`를 Read 도구로 읽어 실행

### 일반 질문 (Query)
위 이외의 모든 경우 → `query.md`를 Read 도구로 읽어 실행
