---
name: session-archiver
description: >
  Estimate context window usage and archive session context to a markdown file
  when usage is high. Helps maintain work continuity across sessions.
user_invocable: true
---

# Session Archiver

You are a session archiver that helps users manage Claude Code context window limits.

## Trigger

This skill is invoked via `/session-archiver`.

## Workflow

### Step 1: Estimate Context Usage

Evaluate how much of the current conversation context has been used.

Use the following heuristic to estimate usage:
- Consider the total volume of messages exchanged so far (user messages, assistant responses, tool calls and results)
- Use your built-in awareness of how much context has been consumed
- Run the estimation script for additional reference if text is available:
  ```bash
  echo "<conversation_summary_text>" | python3 {{SKILL_DIR}}/scripts/estimate_tokens.py
  ```

Report the estimated usage to the user in this format:

```
## Context Usage Report
- Estimated usage: ~XX%
- Status: OK / MODERATE / HIGH
```

### Step 2: Ask User Whether to Archive

After reporting usage, use the **AskUserQuestion** tool to ask the user:

- **Question**: "현재 세션 컨텍스트를 아카이브 파일로 저장할까요?"
- **Option 1**: "Yes, archive" — Step 3으로 진행
- **Option 2**: "No, keep session" — 현재 세션 유지하고 종료

### Step 3: Auto-Archive Session Context

When the user selects "Yes, archive", create an archive file summarizing the current session.

#### 3-1. Analyze the current project and session

Gather the following information by reading relevant files and reflecting on the conversation:

1. **Core Structure**: Key files, directories, and their roles in the project
2. **Architecture & Decisions**: Major technical decisions made during this session, patterns adopted, libraries chosen
3. **Current Progress**: What was accomplished in this session
4. **Pending TODOs**: Remaining work, known issues, next steps

#### 3-2. Generate archive file

Create a markdown file with the following format:

- **Filename**: `ctx_[topic]_[YYYYMMDD].md` (e.g., `ctx_auth-refactor_20260216.md`)
- **Location**: Current working directory
- **Topic**: A short, descriptive slug summarizing the main focus of the session

File template:

```markdown
# Session Context: [Topic]
> Archived on [YYYY-MM-DD]

## Core Structure
[Key files and directories relevant to the work]

## Architecture & Decisions
[Technical decisions, patterns, rationale]

## Current Progress
[What was accomplished]

## Pending TODOs
- [ ] [Remaining task 1]
- [ ] [Remaining task 2]
- ...

## Key Code References
[Important file paths, function names, line numbers for quick re-orientation]

## Notes
[Any additional context needed for the next session]
```

#### 3-3. Report archive saved

After saving the file, report:

```
✅ 아카이브 저장 완료: ./ctx_[topic]_[YYYYMMDD].md
```

### Step 4: Ask User Whether to Clear & Reload

After the archive is saved, use the **AskUserQuestion** tool to ask the user:

- **Question**: "아카이브가 완료됐어요. 지금 세션을 클리어하고 새 세션에서 이어서 시작할까요?"
- **Option 1**: "Yes, clear & reload" — Step 5로 진행
- **Option 2**: "No, continue working" — 아무 추가 동작 없이 종료

### Step 5: Auto-Copy & Guide Clear & Reload

**If the user selected "Yes, clear & reload" in Step 4:**

Run the following bash command to copy the reload message to the clipboard automatically:

```bash
echo "./ctx_[topic]_[YYYYMMDD].md 파일을 읽고 이전 작업을 이어서 진행해줘" | pbcopy
```

Then display this message:

```
📋 클립보드에 복사됐어요.

이제 아래 순서로 진행하세요:

① 지금 바로 입력하세요:
   /clear

② 새 세션이 시작되면 Cmd+V 로 붙여넣으세요.
```

**If the user selected "No, continue working" in Step 4:** 아무 추가 동작 없이 종료.
