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

#### 3-3. Post-archive guidance

After saving the file, present the user with:

```
Archive saved: ./ctx_[topic]_[YYYYMMDD].md

To continue in a fresh session:
1. Run /clear to reset context
2. In the new session, ask Claude to read the archive file:
   "Read ./ctx_[topic]_[YYYYMMDD].md and continue from where we left off"
```

Ask the user if they want to proceed with `/clear` now.
