# Question 3: Prompt Engineering Under Adversarial Constraints

## Problem

Replace a naive “write exam questions from these notes” prompt with a **deliberate prompt system** (plus minimal code) that yields **review-ready structured items** in **one model call per request**. Grade on **validator pass rates** on visible and hidden scenarios, not README prose.

## Hard rules (violations fail)

1. **Single-shot** — One user message per scenario (fixed system message OK). No retry loops in the eval harness.
2. **Prompt budget** — System + filled user template ≤ **4,000 tokens** (your model’s tokenizer). Document counts.
3. **≤3 prompt variants** total; behavior via structured fields in the payload, not per-scenario system prompts.
4. **No answer key in the prompt** — Do not tell the model the correct answers to generate (transcription tasks excepted).
5. **JSON only** — Must parse against the schema below; unparseable = fail.
6. **Blind holdout** — **20 unseen scenarios**; visible pass rate alone is not enough.

## Deliverables


| Path               | Purpose                                                                                         |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| `prompts/`         | Versioned templates + `CHANGELOG.md`                                                            |
| `schema/`          | Output JSON Schema                                                                              |
| `runner/`          | `run --scenario <id>`, `run --suite visible`, `run --suite all`                                 |
| `validate/`        | Same deterministic checks for visible and holdout                                               |
| `report/REPORT.md` | Pass rates by tag, failure taxonomy, ablation (≥3 prompt versions on 10 scenarios), token table |
| `examples/`        | Raw JSON for every **visible** scenario                                                         |
| `submission/agent-transcript/` | **Required.** Full **Cursor / Claude Code / IDE agent** chat export(s) + index README ([details](../README.md#submission)) |


Document model, temperature (eval default **0**), top_p, seed. Freeze `prompts/final/` in git.

## Output contract (summary)

Top level: `{ "items": [...], "warnings": [] }`.

Each item: `itemType` (`MCQ` | `MCQ_MULTI` | `SHORT_ANSWER`), `language` (BCP-47), `stem`, `options`, `visualBrief`, `sharedStimulus`, `stimulusIndex`.

**Type rules (validator-enforced):**

- **MCQ:** 4 options, exactly one correct; no “All/None of the above”.
- **MCQ_MULTI:** 4 options, ≥2 correct; stem includes **select all that apply**.
- **SHORT_ANSWER:** one correct option (reference answer text).

Options: `{ "text", "isCorrect" }`. Preserve source `**bold**` / `_italic_` as `<b>`, `<i>`, or `<em>` when transcribing.

Full field table and edge cases: same rules as in code/schema you ship; scenario intent is in [question-3-scenarios.md](question-3-scenarios.md).

## Scenario suite

**38 visible** cases (IDs **A1–G4**), grouped by tag: type discipline, visual brief fidelity, passage grouping, language/tone, transcribe vs generate, distractor quality, batch/consistency. Target **≥34/38** on your validator plus spot-check failures. **≥32/38 visible** and **≥14/20 holdout** minimum bar; no visible tag category below **60%**.

Scenario text, traps, and tags: **[question-3-scenarios.md](question-3-scenarios.md)**.

## Scoring


| Component                    | Weight |
| ---------------------------- | ------ |
| Visible pass rate            | 25%    |
| Holdout pass rate            | 35%    |
| Failure taxonomy + ablation  | 15%    |
| Token budget + variant limit | 10%    |
| Reproducibility              | 15%    |


We do not provide production prompts, worked solutions for adversarial cases (e.g. stale numbers, injection, key leak), or per-scenario prompt tuning.

## Submission checklist

- [ ] `prompts/final/` tagged in git
- [ ] `REPORT.md` (pass matrix, ablation, tokens)
- [ ] `examples/*.json` for all visible IDs
- [ ] `submission/agent-transcript/` (Cursor, Claude Code, or equivalent agent chats — full sessions, not snippets; secrets redacted)
- [ ] One command reproduces visible suite with only `API_KEY` in env