# Induseuro-Labs Intern Assignment: Advanced Multi-Modal Catalog & AI Question Platform

> **Candidate:** Shivshankar Mali ([@shiv123-coder](https://github.com/shiv123-coder))  
> **Repository Fork:** [github.com/shiv123-coder/intern-assignment](https://github.com/shiv123-coder/intern-assignment)  
> **Official Pull Request:** [Induseuro-labs/intern-assignment #4](https://github.com/Induseuro-labs/intern-assignment/pull/4)  
> **Release Tag:** [v1.0.0](https://github.com/shiv123-coder/intern-assignment/releases/tag/v1.0.0)  
> **Cloud Deployment (Render):** [intern-assignment.onrender.com](https://intern-assignment.onrender.com)  

A comprehensive, production-grade implementation of both assignment challenges:
1. **Question 1:** Multi-Modal Image Catalog, Occlusion-Aware Object Detection, Source-of-Truth Metadata Management, and Natural Language Compositor with Closed-Loop Verification.
2. **Question 2:** AI Question Generation with Agglomerative Concept Avoidance, Token Budgeting, and Independent Multi-Stage Safeguard Duplicate Detection.
3. **Engineering Console:** Tactile Skeuomorphic and Claymorphic Web Console running natively on FastAPI (port 8000), Streamlit (port 8501), and Docker on Render.

---

## 🌟 Key Highlights & Standout Features

| Capability | Question 1: Image Catalog & Compositing | Question 2: Question Avoidance & Detection |
|---|---|---|
| **Core Architecture** | Dual-layer schema (`auto` detector outputs vs `user` overrides) + SQLite/JSON persistence | Two-model independence (Avoidance via Concept Clustering + Safeguard Cascade via intent discriminators) |
| **Object Detection** | Multi-modal: Photos (fruits), cartoons (clothing colors), geometric shapes (circle, square, triangle, rect), swatches | Content-driven multi-stage cascade: exact canonical keys, numeric givens, intent taxonomy, dense vector cosine |
| **Edge Cases Handled** | Overlapping shapes with `occluded_by` stacking & `visible_fraction`, blank images (`no_objects_found`) | Subtle paraphrases, hard negatives (convex vs concave, 30° vs 55°, law 1 vs law 2), scope ambiguities (`NEEDS_REVIEW`) |
| **Compositing & Realism** | Calibrated soft drop shadows, authentic asset provenance (`generative_fill=False`), size priors | Context-budgeted prompt generation (MMR + Agglomerative Clustering) + in-loop multi-round rejection feedback |
| **Verification Loop** | Closed-loop `--verify` runs detector on composite output with automated IoU alignment scoring | Held-out Q1–Q10 dataset (10/10 PASS) + 60-pair calibration benchmark (100% duplicate precision, 0 silent misses) |
| **Interfaces Provided** | Python API + Unified CLI (`python cli.py q1`) + Interactive Streamlit Web UI | Python API + Unified CLI (`python cli.py q2`) + Interactive Streamlit Web UI |

---

## 🏛️ System Architecture

### Question 1: Catalog, Detection, and Compositor Architecture

```
                            ┌────────────────────────────┐
                            │      CLI / API / UI        │
                            └──────────────┬─────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
       ┌────────────────────────┐                    ┌────────────────────────┐
       │   ImageCatalog Storage │                    │  Hybrid Object Detector│
       │   - models.py          │                    │  - detector.py         │
       │   - auto vs user schema│◄───────────────────┤  - CIE Lab color naming│
       │   - ProtectedObjectsErr│                    │  - Multi-modal geometry│
       │   - Audit & tombstones │                    │  - Occlusion & depth   │
       └────────────┬───────────┘                    └────────────────────────┘
                    │
                    ▼
       ┌────────────────────────┐                    ┌────────────────────────┐
       │  Composite Engine      │                    │ Closed-Loop Verifier   │
       │  - Prompt parser       │                    │ - Runs detector on crop│
       │  - Size priors         │───────────────────►│ - Bounding box IoU     │
       │  - Soft contact shadow │                    │ - Provenance audit log │
       │  - Generative fill: No │                    │                        │
       └────────────────────────┘                    └────────────────────────┘
```

### Question 2: Duplicate Prevention & Independent Safeguard Pipeline

```
                [Target Subject / Concept]
                            │
                            ▼
     ┌─────────────────────────────────────────────────────────┐
     │ 1. PREVENTION: QuestionGeneratorWithAvoidance           │
     │    - Agglomerative Concept Clustering (Cosine >= 0.80)  │
     │    - Bounded Token Budgeting (~1,200 tokens O(1))       │
     │    - MMR Diversity Exemplars & Exhausted Facet Warning  │
     │    - LLM / Deterministic High-Fidelity Generation       │
     │    - In-Loop Rejection Feedback Loop (Up to 3 Rounds)   │
     └──────────────────────────────┬──────────────────────────┘
                                    │ (Candidate Questions)
                                    ▼
     ┌─────────────────────────────────────────────────────────┐
     │ 2. SAFEGUARD: IndependentDuplicateDetector              │
     │    Stage 1: Exact Key Normalization (NFKC, punctuation) │
     │    Stage 2: Numeric Given Invariants (30° vs 55°)       │
     │    Stage 3: Pedagogical Intent Classifier (Laws vs Def) │
     │    Stage 4: Dense Semantic Cosine + Scope Ambiguity     │
     └──────────────────────────────┬──────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       [ACCEPTED / UNIQUE QUESTION]     [REJECTED DUPLICATE / REVIEW]
       - Stored with Provenance Tags    - Logged with Failure Rationale
```

---

## 🚀 Quickstart & Usage

### 1. Interactive Web Application (Streamlit)

Launch the full interactive browser interface featuring real-time detection previews, bounding box overlays, visual composite generator, and duplicate check gauges:

```bash
python cli.py serve
# Open browser at http://localhost:8501
```

### 2. Unified Command-Line Interface (`cli.py`)

You can access all functions via the top-level `cli.py` or directly through Python module syntax:

#### Question 1 Commands
```bash
# List all catalog entries and detected objects
python cli.py q1 list

# Add new image to catalog with auto-detection
python cli.py q1 add question_1_image_catalog/data/apple.png --name apple

# Run detection on catalog entry (enforces manual edit protection)
python cli.py q1 detect apple

# Force re-detection (archives previous manual edits in detection_runs)
python cli.py q1 detect apple --force

# Manually edit an object's label or confidence
python cli.py q1 edit apple obj_id --label "golden_delicious_apple" --conf 0.99

# Generate a composite image from natural language with closed-loop verification
python cli.py q1 compose "Create an image containing one red apple and one orange" --verify
python cli.py q1 compose "Create a composite containing a boy, a girl, and colored shapes" --verify

# Compile comprehensive visual report for Question 1
python cli.py q1 report
```

#### Question 2 Commands
```bash
# Compare two questions directly (instant pairwise check)
python cli.py q2 check "Define the reflection of light." "What is reflection of light?"
python cli.py q2 check "What is a concave mirror?" "What is a convex mirror?"

# Check a candidate against the entire question bank
python cli.py q2 check "What is reflection?"

# Generate genuinely new questions with avoidance & safeguard gate
python cli.py q2 generate "Reflection of light" -n 5

# Evaluate on assignment ground-truth Q1-Q10 corpus
python cli.py q2 eval

# Run vector bank retrieval scale benchmark (1,000 to 100,000 vectors)
python cli.py q2 benchmark

# Compile comprehensive evaluation report for Question 2
python cli.py q2 report
```

#### Test Runners
```bash
# Run all test suites across both questions
python cli.py test

# Compile both comprehensive Markdown reports (REPORT.md in each folder)
python cli.py report
```

---

## 📊 Empirical Evaluation & Benchmark Results

### Question 1: All 7 Required Scenarios
The complete run is documented in [`question_1_image_catalog/REPORT.md`](question_1_image_catalog/REPORT.md):
- **Scenario 1 (Multi-modal detection):** 100% accuracy on photos (apple, orange), cartoons (boy clothing: blue/navy; girl clothing: pink/yellow), shapes (circle, square, rect, triangle), and swatches (5 colors).
- **Scenario 2 (Manual addition):** Undetected leaf stem added manually with origin tagged as `user`.
- **Scenario 3 (Manual editing):** Modified label and confidence; read-only `auto` fields preserved intact.
- **Scenario 4 (Persistence & Protection):** Rerun without `--force` raises `ProtectedObjectsError`. Forced rerun archives previous snapshot in `detection_runs`.
- **Scenario 5 (Apple + Orange composite):** Composite created with soft contact shadows; detector verifies all placed items (**Average IoU alignment: 0.891**).
- **Scenario 6 (Boy + Girl + Shapes composite):** Proportional layout using real-world size priors (person ~120, apple ~8, shapes ~36).
- **Scenario 7 (Overlapping & Empty canvas):** Overlapping shapes resolved with `occluded_by` stacking order; uniform white canvas gracefully returns `no_objects_found`.

### Question 2: Ground Truth & Calibration Benchmarks
The complete run is documented in [`question_2_duplicate_prevention/REPORT.md`](question_2_duplicate_prevention/REPORT.md):
- **Held-Out Assignment Q1–Q10 Dataset:** **10 / 10 Correct (100.0% Accuracy)**.
  - Precision: **1.000** | Recall: **1.000** | F1-Score: **1.000**
- **60-Pair Multi-Disciplinary Calibration Benchmark:** **98.3% Accuracy (59 / 60)**.
  - Duplicate Precision: **100.0%** (0 false positives; no valid questions wrongly deleted).
  - Silent Failure Rate: **0 silent misses** (0 duplicates slipped through undetected).
  - Ambiguous / Review Cases: Safely routed to review queue (e.g. Q8 *"What is reflection?"*).
- **Vector Retrieval Scale Benchmark:**
  - 1,000 questions: **0.12 ms** (8,082 queries/sec)
  - 10,000 questions: **1.46 ms** (683 queries/sec)
  - 50,000 questions: **5.98 ms** (167 queries/sec)
  - 100,000 questions: **11.08 ms** (90 queries/sec)
- **Ablation Comparison:**
  - Avoidance Architecture: **0% Duplicates (100% Unique Novel Questions)**.
  - Naive Prompting: **37.5% Duplicates / Review Cases**.

---

## 🛠️ Technology Choices and Trade-offs

| Component | Choice Made | Why This Choice Was Selected | Architectural Trade-Off |
|---|---|---|---|
| **Storage & Schema** | Dual-dictionary JSON / SQLite (`auto` vs `user`) | Enforces clear data ownership and easy field rollback. | Slightly larger storage footprint than flat schemas; well worth auditability. |
| **Object Detection** | Classical segmentation + Contour geometry + YOLO fusion | High precision and zero false positives on plain backgrounds; recovers amodal shapes and exact bounds. | Highly busy/cluttered backgrounds fall back to deep learning object detection (YOLO/zero-shot). |
| **Compositing** | Deterministic rule parser + authentic asset cutouts + soft drop shadows | Strictly guarantees **NO generative hallucination** (`generative_fill: false`). Assets are authentic. | Composites are clean cutouts with lighting shadows rather than relit generative fills. |
| **Question Avoidance** | Agglomerative Clustering + MMR Diversity | Collapses paraphrase families to keep context bounded within ~1,200 tokens even with 100k questions. | Concepts beyond the token budget rely on the in-loop safeguard filter. |
| **Safeguard Detection** | Multi-stage cascade (exact key, numeric given, intent taxonomy, dense cosine) | Explainable verdicts with sub-millisecond inference and 0 external network dependencies. | Requires rule patterns for specific exam taxonomies, backed by semantic embeddings. |

---

## 📁 Repository Directory Structure

```
Induseuro-labs-intern-assignment/
├── cli.py                                   # Unified command-line interface
├── run_all_demos.py                         # Master verification demo runner
├── requirements.txt                         # Python dependencies
├── Dockerfile                               # Production Docker container specification
├── render.yaml                              # Render 1-click cloud blueprint
├── Procfile                                 # PaaS process definition
├── app/                                     # Engineering Console Web Application
│   ├── server.py                            # FastAPI backend REST API
│   ├── static/                              # Tactile Skeuomorphic & Claymorphic frontend
│   │   ├── index.html                       # Tactile console markup
│   │   ├── styles.css                       # Skeuomorphic & claymorphic CSS tokens
│   │   ├── app.js                           # Interactive JavaScript controller
│   │   └── favicon.svg                      # Custom optical lens favicon
│   ├── streamlit_app.py                     # Streamlit entry point (port 8501)
│   ├── pages/                               # Streamlit secondary pages
│   └── utils/
│       └── visualizer.py                    # Bounding box & annotation renderer
├── question_1_image_catalog/
│   ├── __main__.py                          # Q1 CLI executable module
│   ├── generate_report.py                   # Automated 7-scenario visual report builder
│   ├── REPORT.md                            # Comprehensive visual evaluation report
│   ├── run_tests.py                         # Q1 test suite
│   ├── catalog/
│   │   ├── models.py                        # Pydantic models (ProtectedObjectsError, audit log)
│   │   └── storage.py                       # Catalog storage & source-of-truth persistence
│   ├── detector/
│   │   └── detector.py                      # Multi-modal object & shape detector
│   ├── compositor/
│   │   └── engine.py                        # Natural language compositor with verification
│   └── data/                                # Test assets & procedural generator
└── question_2_duplicate_prevention/
    ├── __main__.py                          # Q2 CLI executable module
    ├── generate_report.py                   # Automated evaluation & benchmark report builder
    ├── REPORT.md                            # Comprehensive Q2 evaluation report
    ├── run_tests.py                         # Q2 test suite
    ├── corpus/
    │   ├── models.py                        # Question data models & normalization
    │   └── repository.py                    # Question store & tag filtering
    ├── detector/
    │   └── detector.py                      # Multi-stage independent safeguard detector
    ├── generator/
    │   └── generator.py                     # Context-budgeted generator with avoidance
    ├── evaluator/
    │   └── evaluator.py                     # Evaluation suite & scale benchmark
    └── data/
        └── calibration_pairs.json           # 60 labelled calibration pairs
```

---

## 🏆 Summary

This project delivers **100% compliance** with all requirements across both assignment questions, backed by:
- Complete programmatic verification passing with zero errors across all test suites.
- Tactile Skeuomorphic & Claymorphic Engineering Console (FastAPI on port 8000 & Streamlit on port 8501).
- Unified CLI (`python cli.py test`, `python cli.py serve`, `python cli.py q1`, `python cli.py q2`).
- Academic-grade evaluation reports with ground-truth confusion matrices and scale benchmarks.
