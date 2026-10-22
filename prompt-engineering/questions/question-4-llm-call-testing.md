# Question 4: Testing LLM Calls for Consistency and Accuracy

## 1. Problem Statement

A service issues **many kinds** of LLM calls (generation, classification, grading, review). Manual review of every combination does not scale. Build a **regression harness** that measures accuracy and consistency across fixtures, prompt versions, and models.

Independent of Questions 1–3. No access to our production codebase required.

## 2. Requirements

1. **≥3 call kinds** with separate prompts or functions (e.g. generate, classify duplicate, accept/reject review).
2. **≥30 hand-authored fixtures** plus a documented way to expand to **100+** cases (combinatorial tags or generator).
3. **Layered checks:** JSON schema, deterministic rules, gold labels; optional LLM-as-judge only with caching and documented limitations.
4. **Consistency mode:** same fixture **N≥3** at temperature 0; report flake rate and field stability.
5. **CI:** default `pytest` / offline path with **no network**; live run behind env API key.
6. **Report:** pass/fail by case and tag, failure excerpts, model + prompt hash + timestamp.

## 3. Starter fixtures (extend heavily)

**Generator (`gold_rules`):** MCQ has 4 options and 1 correct; chart brief contains all author-supplied numbers; passage sets share identical `sharedStimulus`.

**Classifier (`gold`):** paraphrase → `duplicate`; different concept → `distinct`.

**Reviewer (`gold`):** five-option MCQ → `reject`; stem contradicts brief → `reject`; valid item → `accept`.

## 4. Submission

CLI (`eval run`, `eval report`), `data/fixtures/`, `examples/REPORT.md`, README on scaling to 500+ cases without reviewing every output manually.

**Mandatory:** `submission/agent-transcript/` — full **IDE coding agent** chat export(s) (Cursor, Claude Code, Copilot agent, etc.) for work on this question, plus index README ([details](../README.md#submission)). Not a substitute: web-only chat with no code context, or logs from your harness’s LLM API calls.
