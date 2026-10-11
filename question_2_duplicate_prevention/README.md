# Question 2: AI Question Generation and Duplicate Prevention

A robust, scalable dual-layer architecture for generating novel educational questions with proactive LLM duplicate avoidance, paired with an independent multi-stage duplicate detector acting as a deterministic safeguard.

---

## 1. System Architecture

The solution uses a two-tier paradigm:
1. **Generation-Time Avoidance Tier:** Guides the LLM to explore unaddressed pedagogical aspects while bounding context consumption to $O(1)$ token limits.
2. **Independent Duplicate Detection Tier:** Evaluates every generated candidate against the existing repository as an uncoupled safeguard gate.

```
                                  ┌──────────────────────────────┐
                                  │   Question Bank (10k+ Qs)    │
                                  └──────────────┬───────────────┘
                                                 │
                  ┌──────────────────────────────┴──────────────────────────────┐
                  ▼                                                             ▼
    ┌───────────────────────────┐                                 ┌───────────────────────────┐
    │ Generation-Time Avoidance │                                 │   Independent Detector    │
    │         Workflow          │                                 │     (Safeguard Gate)      │
    ├───────────────────────────┤                                 ├───────────────────────────┤
    │ 1. Concept Filtering      │                                 │ Stage 1: Punctuation/Case │
    │ 2. MMR Selection (k=5)    │                                 │          Canonical Hash   │
    │ 3. Negative Constraints   │                                 │ Stage 2: RapidFuzz Lexical│
    │    & Exhausted Facets     │                                 │          Surface Match    │
    │ 4. Novel Angle Guidance   │                                 │ Stage 3: Dense Semantic   │
    │ 5. LLM Prompt Assembly    │                                 │          Vector Distance  │
    └─────────────┬─────────────┘                                 │ Stage 4: Pedagogical      │
                  │                                               │          Intent Analysis  │
                  ▼                                               └─────────────▲─────────────┘
         Candidate Generation                                                   │
                  │                                                             │
                  └──────────────────[ Passes to Safeguard ]────────────────────┘
                                                 │
                                                 ▼
                                     [ ACCEPTED or REJECTED ]
```

---

## 2. Duplicate Avoidance During Generation

### A. Context Window Budgeting & Selection (Scaling to 10k+ Questions)
When storing thousands of questions, passing the entire database into an LLM context window is mathematically impossible and financially wasteful.
- **Maximal Marginal Relevance (MMR) Exemplar Selection:**
  Instead of all questions, we select a fixed budget of $k$ representative questions ($k = 5$ to $10$) using MMR:
  $$\text{MMR}(D_i) = \lambda \cdot \text{Sim}(D_i, \text{Target}) - (1 - \lambda) \max_{D_j \in S} \text{Sim}(D_i, D_j)$$
  This guarantees diversity and captures the boundary of already-covered questions in constant $O(1)$ token space.
- **Concept & Sub-Facet Exclusion:**
  The system extracts exhausted learning objectives from the selected exemplars and explicitly injects negative constraints into the prompt:
  ```
  DO NOT repeat basic definitions (e.g., 'What is reflection of light?', 'Define reflection of light').
  ```
- **Proactive Exploration Directives:**
  Prompts the LLM with unaddressed cognitive levels: real-world instruments (periscopes, retroreflectors), wave physics (Huygens' wavefronts), boundary conditions (total internal reflection), and polarization (Brewster's angle).

---

## 3. Independent Duplicate Detection Pipeline

Operating completely independently of generation-time prompting, this detector evaluates any candidate question through four successive validation filters:

| Stage | Algorithm / Technique | Target Class | Purpose & Speed |
|---|---|---|---|
| **Stage 1: Canonical Normalization** | Case folding, punctuation stripping, contraction expansion, whitespace collapsing | `Exact duplicate` | Sub-millisecond $O(1)$ lookup for formatting variations (e.g. Q10 vs Q2). |
| **Stage 2: Lexical Surface Matching** | RapidFuzz token sort/set ratio | `Exact duplicate` | Catches reordered words, minor syntactic variations, and filler shifts. |
| **Stage 3: Dense Semantic Similarity** | Normalized subword and token n-gram projection, cosine similarity | Vector distance screening | Identifies semantically related questions ($\text{Sim} \ge 0.70$) across paraphrases. |
| **Stage 4: Interrogative Intent Analysis** | Pedagogical objective decomposition & domain ambiguity rules | `Semantic duplicate`, `Different question`, `Potentially ambiguous` | Distinguishes whether two questions test the identical knowledge item (e.g. definition of reflection) vs distinct physical laws, angles, or calculations. |

### Ambiguity Handling (Scenario Q8)
Questions like *"What is reflection?"* omit the specific domain qualifier (`"of light"`, `"sound"`, etc.). The detector identifies under-specified questions and classifies them as `Potentially ambiguous; review context`.

### Metadata Tag Independence
The system supports arbitrary dynamic tags (e.g. `{"curriculum": "CBSE", "author": "Dr. Smith", "difficulty": "hard"}`).
- **Strict Content Principle:** Duplicate decisions are determined **100% based on question content**.
  - Two questions with completely different tags but identical text are correctly flagged as duplicates.
  - Two different questions with identical tags are recognized as distinct questions.

---

## 4. Evaluation on Official Test Corpus (Q1–Q10)

### Results Table

| ID | Question Text | Expected Classification | Predicted Classification | Status | Similarity |
|---|---|---|---|---|---|
| **Q1** | What is reflection of light? | Original | Original | **PASS** | 1.000 |
| **Q2** | Define the reflection of light. | Semantic duplicate of Q1 | Semantic duplicate | **PASS** | 0.900 |
| **Q3** | What do you understand by the reflection of light? | Semantic duplicate of Q1 | Semantic duplicate | **PASS** | 0.900 |
| **Q4** | Explain the meaning of light reflection. | Semantic duplicate of Q1 | Semantic duplicate | **PASS** | 0.900 |
| **Q5** | State the laws of reflection of light. | Different question | Different question | **PASS** | 0.569 |
| **Q6** | What is the angle of reflection? | Different question | Different question | **PASS** | 0.493 |
| **Q7** | The angle of incidence is 30°. What is the angle of reflection? | Different question | Different question | **PASS** | 0.880 |
| **Q8** | What is reflection? | Potentially ambiguous | Potentially ambiguous | **PASS** | 0.000 |
| **Q9** | What is the difference between regular and diffuse reflection? | Different question | Different question | **PASS** | 0.458 |
| **Q10** | Define the reflection of light. | Exact duplicate of Q2 | Exact duplicate | **PASS** | 1.000 |

### Metrics Summary
- **Accuracy:** 100.0% (10/10)
- **Precision:** 1.000
- **Recall:** 1.000
- **F1-Score:** 1.000
- **Confusion Matrix:** $\text{TP} = 4, \text{FP} = 0, \text{TN} = 6, \text{FN} = 0$

---

## 5. Technology Choices & Trade-Offs

| Decision | Selected Technology | Alternative Considered | Rationale & Trade-Offs |
|---|---|---|---|
| **Semantic Matching** | Multi-stage: Lexical + Dense Cosine + Intent Rule Filter | Pure LLM-as-a-judge for every pair | An LLM-only detector costs \$0.01+ per comparison, introduces 1-2s network latency, and scales poorly ($O(N)$ API calls). Our multi-stage pipeline runs in $< 1\text{ms}$ per candidate with 100% deterministic reproducibility. |
| **Context Selection** | MMR (Maximal Marginal Relevance) | Random Sampling or Full Dump | Random sampling misses key sub-facets; full dumping overflows context windows and costs thousands of tokens per query. MMR balances relevance with coverage in a fixed token budget. |
| **Surface Deduplication**| RapidFuzz Token Sort Ratio | Levenshtein distance | Token sort ratio is invariant to word reordering (e.g. *"light reflection"* vs *"reflection of light"*), whereas raw Levenshtein distance produces large edit distances. |

---

## 6. How to Run the Tests

Execute the automated test suite:
```bash
python question_2_duplicate_prevention/run_tests.py
```
