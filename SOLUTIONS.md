# Intern Assignment: Solutions

My solutions to both questions in [Induseuro-labs/intern-assignment](https://github.com/Induseuro-labs/intern-assignment).
The original prompts are in [`questions/`](questions/).

| | Question | Solution | Design doc |
|---|---|---|---|
| 1 | Image catalog, object detection, metadata, composite generation | [`q1_image_catalog/`](q1_image_catalog/) | [README](q1_image_catalog/README.md) |
| 2 | AI question generation and duplicate prevention | [`q2_duplicate_questions/`](q2_duplicate_questions/) | [README](q2_duplicate_questions/README.md) |

## Quick start

```bash
pip install -r requirements.txt          # core: numpy, pillow, opencv, scikit-learn, pytest
pip install -r requirements-ai.txt       # optional: sentence-transformers + anthropic

python -m pytest -q                      # all tests (MiniLM variants are skipped if not installed)
python -m q1_image_catalog.demo          # -> examples/q1/
python -m q2_duplicate_questions.demo    # -> examples/q2/DEMO_LOG.md
python -m q2_duplicate_questions.evaluate  # -> examples/q2/EVALUATION.md
```

**Works offline by default; AI is optional.** Every hosted-AI component has a deterministic local version. Tests and
demos run with no API key and give the same output every time. If `ANTHROPIC_API_KEY` is set, Claude (`claude-opus-5-5`)
is used for the parts where it adds value: labelling detected regions in images, parsing free-form composite
requests, generating questions, and settling borderline duplicate cases. All Claude calls go through
[`common/llm.py`](common/llm.py), which uses structured JSON output so replies always parse.

## Highlights

**Q1**
- Detects every object in the section-6 images (apple, orange, shapes, colour swatches, overlapping shapes, cartoon
  people). It reports `no_objects` for an empty image and `failed`, with a reason, for a cluttered photo; it does not
  guess.
- Manual corrections are the source of truth. An image with corrections is skipped by batch re-detection, and a
  single re-run is refused unless the user explicitly asks (`force` / `keep_manual`). The detector's original values
  are kept in `auto_original`.
- Composites use *only* pixels from the catalog. A missing asset raises `MissingAssetError` and nothing is
  substituted. Each composite is saved as a new catalog entry with full provenance: source object, scale, placement,
  and z-order for every layer.

| `"one red apple and one orange"` | `"a boy, a girl and colored shapes"` |
|---|---|
| ![](examples/q1/composite_apple_orange.png) | ![](examples/q1/composite_boy_girl_shapes.png) |

**Q2**
- Detection in layers: normalised hash → (intent, concept) signature → MiniLM embeddings → optional LLM judge. Guards
  stop same-topic questions that ask different things (laws, angle, numerical, difference) from being merged.
- Reproduces all 10 expected classifications in section 6, including Q8 → `needs_review`. Scores 22/22 on a separate
  held-out set that includes hard negatives such as *refraction* vs *reflection*. No unique question is
  auto-rejected.
- Generation uses a fixed-size coverage brief (nearest questions, cluster exemplars, covered angles, rejected
  candidates), so the prompt does not grow with the bank. The independent detector checks every candidate, and
  rejected ones feed into the next round.

## Layout

```
common/llm.py               shared Claude wrapper (optional)
q1_image_catalog/           catalog.py · detect.py · compose.py · testdata.py · demo.py · __main__.py (CLI) · tests/
q2_duplicate_questions/     core.py · detector.py · generator.py · data.py · evaluate.py · demo.py · tests/
examples/                   generated outputs: logs, annotated detections, composites, evaluation tables
questions/                  original assignment prompts
```
