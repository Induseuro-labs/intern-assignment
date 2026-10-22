"""Generation with duplicate prevention.

1. Context selection (bounded, whatever the collection size)
   * retrieve the existing questions most relevant to the requested topic
     (MiniLM embeddings, optional tag filter to scope the pool);
   * cluster them into *concepts* (agglomerative, cosine ≥ 0.80 = same concept)
     — a paraphrase family collapses into one line;
   * show the LLM one representative per concept, largest/most relevant first,
     with its variant count, until a token budget is reached.
   Prompt size therefore depends on the number of distinct concepts that fit
   the budget, not on the number of stored questions.
2. Steering
   * the prompt lists covered concepts, explains that rewording does not make
     a question new, asks the model to first name uncovered concepts and then
     write one question per concept, varying the question type.
3. In-loop novelty filter (still prevention, not detection)
   * each candidate is embedded and compared with the relevant pool and with
     questions accepted earlier in this run; too-close candidates are rejected
     and fed back to the next round as "do not write questions like these".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .embeddings import GEN_MODEL, Embedder
from .llm import GEN_LLM, chat_json
from .text import exact_key


@dataclass
class Concept:
    representative: str
    size: int
    members: list[str]
    relevance: float


@dataclass
class Context:
    topic: str
    pool_size: int           # questions considered (after tag filter)
    relevant: int            # questions retrieved as relevant to the topic
    concepts: list[Concept]  # all clusters
    shown: int               # clusters that fit the budget
    prompt_block: str
    est_tokens: int


@dataclass
class Candidate:
    question: str
    concept: str = ""
    qtype: str = ""
    accepted: bool = False
    reason: str = ""
    nearest: str | None = None
    similarity: float = 0.0
    round: int = 0


@dataclass
class GenerationResult:
    topic: str
    context: Context
    accepted: list[Candidate]
    rejected: list[Candidate]
    rounds: int
    prompts: list[str] = field(default_factory=list)


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def cluster(vecs: np.ndarray, threshold: float = 0.80) -> list[list[int]]:
    """Average-linkage agglomerative clustering on cosine similarity."""
    n = len(vecs)
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    from sklearn.cluster import AgglomerativeClustering

    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1 - threshold).fit_predict(vecs)
    groups: dict[int, list[int]] = {}
    for i, l in enumerate(labels):
        groups.setdefault(int(l), []).append(i)
    return list(groups.values())


class QuestionGenerator:
    SYSTEM = (
        "You write new exam questions for an existing question bank. The bank already covers the concepts "
        "listed by the user (one representative question per concept, with how many variants exist). Your "
        "job is to add questions that are genuinely NEW.\n"
        "Rules:\n"
        "- Do not paraphrase or reword any covered question. Changing 'What is X?' into 'Define X', 'What do "
        "you understand by X', 'Explain the meaning of X' or 'X ... meaning' is the SAME question.\n"
        "- A question is new only if it tests a different concept, sub-concept, skill or application, or asks "
        "for a different kind of answer (law, reason, comparison, numerical calculation with new values, "
        "real-life application, diagram/ray reasoning, misconception check, experiment design).\n"
        "- Every new question must also differ from every other new question you write.\n"
        "- Questions must be self-contained, unambiguous and answerable at the level of the existing bank.\n"
        "First list the uncovered concepts you will target, then write exactly one question per concept.\n"
        "Return only JSON: {\"uncovered_concepts\": [str], \"questions\": [{\"question\": str, \"concept\": str, "
        "\"type\": str}]}"
    )

    def __init__(self, store, model: str = GEN_LLM, budget_tokens: int = 1500, novelty_threshold: float = 0.82,
                 relevance_floor: float = 0.35, max_pool: int = 400, max_rounds: int = 3, llm=None):
        self.store = store
        self.model = model
        self.embedder = Embedder(GEN_MODEL, store)
        self.budget_tokens = budget_tokens
        self.novelty_threshold = novelty_threshold
        self.relevance_floor = relevance_floor
        self.max_pool = max_pool
        self.max_rounds = max_rounds
        self.llm = llm or (lambda system, user: chat_json(system, user, model=self.model, temperature=0.7))

    # ----------------------------------------------------------- context
    def select_context(self, topic: str, tags: dict | None = None):
        pool = self.store.all(tags=tags, include_review=True)
        if not pool:
            return Context(topic, 0, 0, [], 0, "(the bank has no questions on this topic yet)", 0), [], np.zeros((0, 384))
        vecs = self.embedder.encode_questions(pool)
        tvec = self.embedder.encode([topic])[0]
        rel = vecs @ tvec
        order = np.argsort(-rel)[: self.max_pool]
        order = [i for i in order if rel[i] >= self.relevance_floor] or list(order[:20])
        rel_qs = [pool[i] for i in order]
        rel_vecs = vecs[order]
        concepts = []
        for members in cluster(rel_vecs):
            sub = rel_vecs[members]
            centroid = sub.mean(0)
            medoid = members[int(np.argmax(sub @ centroid))]
            concepts.append(Concept(rel_qs[medoid].text, len(members), [rel_qs[m].text for m in members],
                                    float(rel[order[medoid]])))
        concepts.sort(key=lambda c: -c.relevance)  # most on-topic concepts first: likeliest to be regenerated
        lines, used, shown = [], 0, 0
        header = f"Topic: {topic}\nCovered concepts ({len(concepts)} concepts from {len(rel_qs)} existing questions):"
        used = est_tokens(header)
        examples = 0
        for c in concepts:
            rep = c.representative if len(c.representative) <= 140 else c.representative[:137] + "..."
            line = f"- {rep}"
            if c.size > 1:
                other = next((m for m in c.members if m != c.representative), None)
                if other and examples < 3 and len(other) <= 140:  # show a few rewordings so the model sees what "same" means
                    line += f"  [{c.size} variants, e.g. \"{other}\"]"
                    examples += 1
                else:
                    line += f"  [{c.size} variants]"
            if used + est_tokens(line) > self.budget_tokens:
                break
            lines.append(line)
            used += est_tokens(line)
            shown += 1
        if shown < len(concepts):
            lines.append(f"- ... and {len(concepts) - shown} smaller concepts not listed (also covered)")
        block = header + "\n" + "\n".join(lines)
        return Context(topic, len(pool), len(rel_qs), concepts, shown, block, est_tokens(block)), rel_qs, rel_vecs

    # ---------------------------------------------------------- generate
    def generate(self, topic: str, n: int = 5, tags: dict | None = None) -> GenerationResult:
        ctx, rel_qs, rel_vecs = self.select_context(topic, tags)
        ref_texts = [q.text for q in rel_qs]
        ref_vecs = rel_vecs
        existing_keys = {exact_key(q.text) for q in self.store.all()}
        accepted: list[Candidate] = []
        rejected: list[Candidate] = []
        prompts = []
        rnd = 0
        while len(accepted) < n and rnd < self.max_rounds:
            rnd += 1
            want = n - len(accepted)
            user = ctx.prompt_block
            if accepted:
                user += "\n\nAlready written in this session (also covered now):\n" + \
                        "\n".join(f"- {c.question}" for c in accepted)
            if rejected:
                user += "\n\nRejected because they duplicated covered questions — do not write anything similar:\n" + \
                        "\n".join(f"- {c.question}  (too close to: {c.nearest})" for c in rejected[-12:])
            user += f"\n\nWrite {want + 2} new questions on the topic."
            prompts.append(user)
            data = self.llm(self.SYSTEM, user)
            items = data.get("questions", []) if isinstance(data, dict) else data
            cands = [Candidate(str(i["question"]).strip(), str(i.get("concept", "")), str(i.get("type", "")),
                               round=rnd) for i in items if str(i.get("question", "")).strip()]
            if not cands:
                continue
            cvecs = self.embedder.encode([c.question for c in cands])
            for c, v in zip(cands, cvecs):
                if len(accepted) >= n:
                    break
                if exact_key(c.question) in existing_keys:
                    c.reason, c.similarity = "identical to an existing question", 1.0
                    rejected.append(c)
                    continue
                sims = ref_vecs @ v if len(ref_vecs) else np.zeros(0)
                j = int(np.argmax(sims)) if len(sims) else -1
                c.similarity = float(sims[j]) if j >= 0 else 0.0
                c.nearest = ref_texts[j] if j >= 0 else None
                if c.similarity >= self.novelty_threshold:
                    c.reason = f"too similar ({c.similarity:.2f}) to existing/new question"
                    rejected.append(c)
                    continue
                c.accepted, c.reason = True, f"novel (nearest {c.similarity:.2f})"
                accepted.append(c)
                ref_texts.append(c.question)  # later candidates must also differ from this one
                ref_vecs = np.vstack([ref_vecs, v[None]]) if len(ref_vecs) else v[None]
        return GenerationResult(topic, ctx, accepted, rejected, rnd, prompts)


def naive_generate(topic: str, n: int, model: str = GEN_LLM) -> list[str]:
    """Baseline without any knowledge of the bank (for the ablation in the report)."""
    data = chat_json("You write exam questions. Return only JSON {\"questions\": [str]}.",
                     f"Write {n} exam questions on: {topic}", model=model, temperature=0.7)
    return [str(q) for q in data["questions"]][:n]


def to_json(result: GenerationResult) -> str:
    return json.dumps({
        "topic": result.topic, "rounds": result.rounds,
        "context": {"pool_size": result.context.pool_size, "relevant": result.context.relevant,
                    "concepts": len(result.context.concepts), "shown": result.context.shown,
                    "est_tokens": result.context.est_tokens},
        "accepted": [c.__dict__ for c in result.accepted], "rejected": [c.__dict__ for c in result.rejected],
    }, indent=2)
