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

### Step 2: Decide Next Action

- **Below 80%**: Report usage and stop. No further action needed.
- **80% or above**: Inform the user that context is running high, then proceed to Step 3.

### Step 3: Auto-Archive Session Context

When usage is 80% or above, create an archive file summarizing the current session.

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

#### 3-3. Ask user to clear and reload

After saving the file, report:

```
Archive saved: ./ctx_[topic]_[YYYYMMDD].md
```

Then use the **AskUserQuestion** tool to ask the user:

- **Question**: "아카이브가 저장되었습니다. 세션을 클리어하고 아카이브에서 컨텍스트를 다시 로드할까요?"
- **Option 1**: "Yes, clear and reload" — `/clear` 실행 후 아카이브 파일을 읽어서 컨텍스트 복원
- **Option 2**: "No, keep session" — 현재 세션 유지

### Step 4: Handle User Response

**If the user selects "Yes, clear and reload":**

Note the archive file path (e.g., `./ctx_auth-refactor_20260216.md`), then display the following message:

```
✅ 아카이브 완료: ./ctx_[topic]_[YYYYMMDD].md

아래 단계를 따라주세요:
1. /clear 를 입력해서 세션을 초기화하세요
2. 새 세션에서 아래 명령어를 붙여넣으세요:

./ctx_[topic]_[YYYYMMDD].md 파일을 읽고 이전 작업을 이어서 진행해줘
```

**If the user selects "No, keep session":** 아무 추가 동작 없이 종료.
