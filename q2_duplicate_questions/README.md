# Question 2: AI Question Generation and Duplicate Prevention

```bash
python -m q2_duplicate_questions.demo            # detection examples + a generation run -> examples/q2/DEMO_LOG.md
python -m q2_duplicate_questions.evaluate        # accuracy + latency -> examples/q2/EVALUATION.md
python -m pytest q2_duplicate_questions -q
python -m q2_duplicate_questions.demo --claude   # real generation with Claude and the LLM judge (needs ANTHROPIC_API_KEY)
```

## Architecture

```
                         ┌──────────────────────── QuestionBank (core.py) ───────────────────────┐
                         │ questions + tags │ exact-hash index │ float32 embedding matrix         │
                         └──────┬───────────────────────────────────────────────┬────────────────┘
                                │ bounded context                               │ top-k neighbours
 topic, n, tag scope            ▼                                               ▼
 ───────────────► build_context -> prompt -> LLM ─► candidates ─► DuplicateDetector (independent)
                        ▲                                                │
                        └──────── rejected candidates fed back ◄────────┤ duplicate -> reject
                                                                         ├ needs_review -> human queue
                                                                         └ unique -> add to bank
```

The two layers are **independent** by construction. The generator only shapes the prompt. The detector receives a
candidate string and a read-only bank, and shares no state with the generator. It is the same detector you would run
on questions imported from anywhere else.

## Content representations (tags never enter any of them)

| Representation | Catches | Example |
|---|---|---|
| **Normalised text hash**: NFKC, lowercase, punctuation stripped, `°`→`degrees`, number words → digits | Exact duplicates, including formatting differences | `define the REFLECTION of light` = Q2 |
| **Signature `(intent, concept terms)`**: intent from ordered patterns (numerical, difference, laws, examples, reason, process, application, definition); concepts = stemmed words minus question-framing words and numbers | Rewordings with a known framing; questions that are the same apart from the numbers | Q1–Q4 all → `(definition, {light, reflect})`; *"incidence is 45°…"* = Q7's template |
| **Dense embedding**: `all-MiniLM-L6-v2` (384-d, CPU); TF-IDF word+char n-grams if it is not installed | Paraphrases the rules don't anticipate | *"Briefly explain what reflection of light is."* (cos 0.95) |

## Independent detection (`detector.py`)

For each candidate, take the k=8 nearest questions and decide one pair at a time:

1. Normalised hash hit → **exact_duplicate**
2. Same intent **and** same concept set → **semantic_duplicate**
3. **Guards:** different known intents (*definition* vs *laws* vs *numerical*) → not a duplicate. No shared concept
   term (*reflection* vs *refraction*) → not a duplicate. These guards keep same-topic, different-question pairs from
   being merged: Q5, Q6, Q7 and Q9 have high cosine with Q1 but are different questions.
4. Cosine ≥ 0.92 → **semantic_duplicate**; ≥ 0.80 → **needs_review**
5. Optional **LLM judge** (Claude, structured output) settles `needs_review` as duplicate, different, or still ambiguous.

The final verdict is the most severe one across all neighbours. Every verdict carries the matched question and a
human-readable reason.

**Q8 "What is reflection?"** is more general than Q1 and has a cosine of 0.86 with it, so it lands in `needs_review`.
That matches the "potentially ambiguous" label in the brief. When a judge is configured, it decides; otherwise a
human does.

## Generation-time avoidance (`generator.py`)

**Choosing what goes in the prompt without exceeding context.** The prompt holds a fixed-size *coverage brief*,
never the whole bank:
- the **k nearest** existing questions to the topic, which is where duplicates come from;
- one **exemplar per k-means cluster** of the scoped pool, which shows the breadth of the bank and not only its
  nearest corner;
- the most common **covered angles** (`definition: light reflect`, `laws: law light reflect`, …). This is a compact
  summary of hundreds of questions in a few lines;
- the most recent **rejected candidates**, so the model doesn't repeat them.

Tags (`{"topic": "reflection"}`) only choose which part of the bank is used for context. They never decide whether
two questions are duplicates. The demo and tests show the same text with different tags still detected as a duplicate.

**Avoiding questions with the same meaning.** The prompt asks for a declared intent and concept for each question,
which pushes the model toward new angles rather than new wording. It also over-generates by ×1.5, and every candidate
goes through the detector. Accepted questions are added to the bank immediately, so later candidates in the same
batch are checked against them too (the demo shows *"What is meant by a plane mirror?"* rejected against *"What is a
plane mirror?"*, which was accepted a moment earlier). Rejected candidates go back into the next round's prompt, for
up to 3 rounds.

**Scaling.** The prompt size is O(k) whatever the bank size. Exact lookup is an O(1) hash. Neighbour search is a
brute-force matrix product: **~6–20 ms for top-10 over 100k × 384 vectors** on a laptop CPU. Beyond about 10⁶ questions,
replace `QuestionBank.search` with FAISS/HNSW or pgvector (same interface), and replace the k-means clusters with
precomputed, incrementally updated clusters.

## Evaluation

Full tables: [examples/q2/EVALUATION.md](../examples/q2/EVALUATION.md)

| Embedder | Section-6 corpus (10) | Held-out (22) | Duplicate recall | Duplicate precision | Unique questions wrongly rejected |
|---|---|---|---|---|---|
| MiniLM | 10/10 | 22/22 | 1.00 | 1.00 | 0 |
| TF-IDF fallback | 10/10 | 20/22 | 1.00 | 0.89 | 0 (2 sent to review) |

**How to evaluate this properly at scale:**
- **Accuracy.** Build a labelled set of pairs per subject, made by sampling neighbour pairs at several cosine levels
  and having teachers label them, with extra *hard negatives* (same topic, different question). Report
  precision/recall per verdict, keeping *false auto-rejects* separate because they cost the most (a good question is
  lost), and the size of the review queue, which is the human cost.
- **Threshold tuning.** Sweep `dup_threshold` and `review_threshold` on a validation split. Pick the review threshold
  for recall and the dup threshold for precision.
- **Generation quality.** Track novelty rate (accepted ÷ generated), rejections per round, and the coverage gain
  (new signatures per accepted question).
- **Performance.** Measure p50/p95 `check()` latency and index build time against bank size; the evaluation script
  includes both.

Caveat: the held-out set is small and was written by me, so the 100% figure shows the design works on this subject.
It is not a general accuracy claim. The intent patterns are written for English exam phrasing, and a new subject
may need its framing vocabulary extended. The embedding and judge layers are what make it general.

## Trade-offs

| Choice | Benefit | Cost |
|---|---|---|
| Rule signature before embeddings | Explainable, exact on common phrasings; numeric variants grouped by template | Patterns are specific to English; needs tuning for new languages |
| Intent guard | Stops "laws of X" from being merged with "what is X" despite cosine 0.8 | A paraphrase with a wrongly detected intent can be missed; the embedding and review layers limit that |
| Small local embedding model | Free, runs offline, ~30–70 ms per check on CPU | Weaker than large hosted embedding models |
| LLM judge only on the review band | LLM cost only where it is needed | Adds latency for borderline cases |
| Brute-force vectors | No infrastructure | Needs an ANN index past about 10⁶ questions |
