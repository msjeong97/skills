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
