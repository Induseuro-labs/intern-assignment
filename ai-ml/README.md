# AI / ML intern assignment

| Question | Spec | Reference solution |
|----------|------|-------------------|
| [Question 1](questions/question-1-image-catalog.md) | Image catalog, detection, metadata, composites | [question1/](question1/README.md) |
| [Question 2](questions/question-2-duplicate-questions.md) | Generation + duplicate prevention + detection | [question2/](question2/README.md) |

## Setup

From **`ai-ml/`** (or use paths below from the repo root):

```bash
cd ai-ml
uv venv ../.venv --python 3.11
uv pip install --python ../.venv/bin/python -r question1/requirements.txt -r question2/requirements.txt
```

On Windows use `../.venv/Scripts/python.exe` instead of `../.venv/bin/python`.

Put `OPENROUTER_API_KEY=...` in `.env` at the **repository root**. Tests run offline without it.

## Run

```bash
cd question1 && python -m pytest -q tests && python scripts/run_scenarios.py
cd ../question2 && python -m pytest -q tests && python scripts/run_demo.py
```

## Submission

Working code per question attempted, architecture notes, and scenario reports (`question1/examples/REPORT.md`, `question2/examples/REPORT.md`).
