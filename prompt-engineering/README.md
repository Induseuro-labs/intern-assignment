# Prompt engineering intern assignment

| Question | Spec | Notes |
|----------|------|--------|
| [Question 3](questions/question-3-prompt-engineering.md) | **Primary:** adversarial prompt engineering (38 visible + 20 blind) — [scenario list](questions/question-3-scenarios.md) | Candidate implements `prompts/`, `runner/`, `validate/` |
| [Question 4](questions/question-4-llm-call-testing.md) | LLM call regression, consistency, CI-friendly eval | Optional second task; not required if only Q3 was assigned |

## Setup

No reference implementation in this repo. Use any stack you prefer; document model, API, and how to reproduce results.

Optional: `OPENROUTER_API_KEY` or your provider’s key in `.env` at the **repository root**.

## Submission

Follow the checklist in each question spec (`REPORT.md`, frozen `prompts/final/`, examples).

**Mandatory:** submit the **coding agent transcript** — the full conversation log from the IDE agent you used to write and iterate on this assignment.

This means tools like **Cursor** (Agent / Composer chats), **Claude Code**, **GitHub Copilot** agent/chat in the editor, **Windsurf**, **Cline**, or similar **code-aware agents**. Using them is allowed and expected.

| Requirement | Detail |
|-------------|--------|
| Location | `submission/agent-transcript/` in your solution repo (or PR) |
| What to include | Exported or saved **agent chat sessions** for this take-home (prompt edits, validator/runner work, debugging). One file per session or one export per tool is fine. |
| What is not enough | Pasted prompts only, a generic web ChatGPT thread with no repo context, or **`examples/*.json` / API logs from your eval runner** (those are separate deliverables) |
| Format | Whatever your tool exports (`.md`, `.json`, `.html`, plain text). If there is no export, copy the full thread into a `.md` file in chronological order. |
| Index | `submission/agent-transcript/README.md`: agent product + version (e.g. Cursor 0.x), model if shown, date range, filename → Q3 vs Q4 |
| Redaction | Remove API keys, employer secrets, and private holdout material if you were given any |

**Examples (pick what matches your setup):**

- **Cursor:** export or save each Agent/Composer chat used on the assignment.
- **Claude Code:** include the session log(s) from your project directory or terminal history export, with user and assistant turns intact.
- **Copilot / other:** IDE chat export or full thread copy into `submission/agent-transcript/`.
