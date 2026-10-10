# Intern Assignment

This repository contains two independent assignment questions. Each solution lives in its own directory with its own README that explains the architecture, algorithms, technology choices and trade-offs, plus a generated report covering every required scenario.

| Question | Topic | Solution | Scenario run |
|----------|--------|----------|--------------|
| [Question 1](questions/question-1-image-catalog.md) | Image catalog, object detection, metadata, composite image generation | [question1/](question1/README.md) | [question1/examples/REPORT.md](question1/examples/REPORT.md) |
| [Question 2](questions/question-2-duplicate-questions.md) | AI question generation and duplicate prevention | [question2/](question2/README.md) | [question2/examples/REPORT.md](question2/examples/REPORT.md) |

## Setup (both questions)

Python 3.11. One virtual environment at the repository root serves both questions:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu torch torchvision   # or a CUDA build
uv pip install --python .venv/Scripts/python.exe -r question1/requirements.txt -r question2/requirements.txt
```

(On Linux/macOS use `.venv/bin/python`.) Hosted LLM steps use OpenRouter. Put `OPENROUTER_API_KEY=...` in the environment or in a `.env` file at the repository root; it is git-ignored. Both solutions also run without a key, with deterministic fallbacks, and their test suites never call the network LLM.

```bash
cd question1 && python -m pytest -q tests && python scripts/run_scenarios.py
cd question2 && python -m pytest -q tests && python scripts/run_demo.py
```

## Submission

- Provide working code for each question you attempt.
- Include explanations of architecture, algorithms, technology choices, and trade-offs.
- Include examples that demonstrate the required scenarios listed in each question.
