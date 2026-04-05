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
