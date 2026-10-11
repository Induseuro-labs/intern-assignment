"""
Evaluation Suite for Question 2.
Evaluates the Independent Duplicate Detector against the official ground-truth test corpus (Q1-Q10).
Computes performance metrics (Precision, Recall, F1, Accuracy, Confusion Matrix)
and validates Generation-Time Duplicate Avoidance.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple



from ..corpus.models import (
    Question, ClassificationCategory, DuplicateCheckResult
)
from ..corpus.repository import QuestionRepository
from ..detector.detector import IndependentDuplicateDetector
from ..generator.generator import QuestionGeneratorWithAvoidance, QuestionGenerationRequest

# Official ground-truth labels from assignment prompt
GROUND_TRUTH_DATASET = [
    ("Q1", "What is reflection of light?", ClassificationCategory.ORIGINAL, None),
    ("Q2", "Define the reflection of light.", ClassificationCategory.SEMANTIC_DUPLICATE, "Q1"),
    ("Q3", "What do you understand by the reflection of light?", ClassificationCategory.SEMANTIC_DUPLICATE, "Q1"),
    ("Q4", "Explain the meaning of light reflection.", ClassificationCategory.SEMANTIC_DUPLICATE, "Q1"),
    ("Q5", "State the laws of reflection of light.", ClassificationCategory.DIFFERENT_QUESTION, None),
    ("Q6", "What is the angle of reflection?", ClassificationCategory.DIFFERENT_QUESTION, None),
    ("Q7", "The angle of incidence is 30°. What is the angle of reflection?", ClassificationCategory.DIFFERENT_QUESTION, None),
    ("Q8", "What is reflection?", ClassificationCategory.POTENTIALLY_AMBIGUOUS, None),
    ("Q9", "What is the difference between regular and diffuse reflection?", ClassificationCategory.DIFFERENT_QUESTION, None),
    ("Q10", "Define the reflection of light.", ClassificationCategory.EXACT_DUPLICATE, "Q2"),
]

class QuestionCorpusEvaluator:
    def __init__(self, detector: IndependentDuplicateDetector):
        self.detector = detector

    def evaluate_official_corpus(self) -> Dict[str, Any]:
        """
        Runs evaluation incrementally across Q1-Q10 exactly as questions enter the repository.
        """
        accumulated_corpus: List[Question] = []
        eval_rows = []

        correct_count = 0
        total_count = len(GROUND_TRUTH_DATASET)

        # For binary duplicate classification metrics: (Is Duplicate vs Not Duplicate)
        tp = fp = tn = fn = 0

        for qid, text, expected_label, expected_match in GROUND_TRUTH_DATASET:
            if qid == "Q1":
                # First question defines the baseline original
                pred_label = ClassificationCategory.ORIGINAL
                matched_id = None
                sim_score = 1.0
                reason = "First question in corpus establishes original baseline."
                stage = "baseline_corpus"
            else:
                verdict = self.detector.check_candidate(text, accumulated_corpus, candidate_id=qid)
                pred_label = verdict.predicted_classification
                matched_id = verdict.matched_question_id
                sim_score = verdict.similarity_score
                reason = verdict.reasoning
                stage = verdict.stage_resolved

            # Check correctness
            # Accept if pred_label matches expected_label
            # For Q10, matched_id can be Q2 (exact match)
            is_match = (pred_label == expected_label)
            if is_match:
                correct_count += 1

            # Binary duplicate metrics
            is_expected_dup = expected_label in (ClassificationCategory.EXACT_DUPLICATE, ClassificationCategory.SEMANTIC_DUPLICATE)
            is_pred_dup = pred_label in (ClassificationCategory.EXACT_DUPLICATE, ClassificationCategory.SEMANTIC_DUPLICATE)

            if is_expected_dup and is_pred_dup:
                tp += 1
            elif not is_expected_dup and not is_pred_dup:
                tn += 1
            elif not is_expected_dup and is_pred_dup:
                fp += 1
            elif is_expected_dup and not is_pred_dup:
                fn += 1

            eval_rows.append({
                "ID": qid,
                "Question": text,
                "Expected": expected_label.value,
                "Predicted": pred_label.value,
                "Match Target": matched_id or "-",
                "Sim Score": f"{sim_score:.3f}" if isinstance(sim_score, float) else str(sim_score),
                "Status": "PASS" if is_match else "FAIL",
                "Stage": stage,
                "Reasoning": reason
            })

            # Add question to corpus
            accumulated_corpus.append(Question(id=qid, text=text, concept="reflection of light"))

        accuracy = correct_count / total_count
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 1.0

        return {
            "rows": eval_rows,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            "total": total_count,
            "passed": correct_count
        }

    def evaluate_calibration_pairs(self, calibration_path: str) -> Dict[str, Any]:
        """
        Evaluates the detector against the comprehensive calibration pair dataset.
        Computes 4-class confusion matrix, duplicate precision, recall, and review-queue rate.
        """
        import os
        import json
        import time

        if not os.path.exists(calibration_path):
            raise FileNotFoundError(f"Calibration dataset not found: {calibration_path}")

        with open(calibration_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)

        correct = 0
        total = len(pairs)
        tp = fp = tn = fn = 0
        silent_misses = 0  # True duplicate passed as unique/different
        review_count = 0

        eval_details = []
        start_time = time.perf_counter()

        for item in pairs:
            q1_text = item["q1"]
            q2_text = item["q2"]
            expected = item["expected"]

            corpus_q = [Question(id="BASE", text=q2_text, concept=item.get("topic", "General"))]
            verdict = self.detector.check_candidate(q1_text, corpus_q, candidate_id=item["id"])
            pred = verdict.predicted_classification.value

            is_match = (pred == expected)
            if is_match:
                correct += 1

            # Precision/Recall logic
            is_true_dup = expected in ("Exact duplicate", "Semantic duplicate")
            is_pred_dup = pred in ("Exact duplicate", "Semantic duplicate")

            if is_true_dup and is_pred_dup:
                tp += 1
            elif not is_true_dup and not is_pred_dup:
                tn += 1
            elif not is_true_dup and is_pred_dup:
                fp += 1
            elif is_true_dup and not is_pred_dup:
                fn += 1
                if pred == "Different question":
                    silent_misses += 1

            if pred == "Potentially ambiguous; review context":
                review_count += 1

            eval_details.append({
                "id": item["id"],
                "q1": q1_text,
                "q2": q2_text,
                "expected": expected,
                "predicted": pred,
                "match": is_match,
                "stage": verdict.stage_resolved,
                "sim": round(verdict.similarity_score, 3)
            })

        elapsed = time.perf_counter() - start_time
        ms_per_check = (elapsed / total) * 1000

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 1.0

        return {
            "total_pairs": total,
            "correct_pairs": correct,
            "accuracy": round(correct / total, 4),
            "duplicate_precision": round(precision, 4),
            "duplicate_recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "silent_misses": silent_misses,
            "review_queue_rate": round(review_count / total, 4),
            "ms_per_check": round(ms_per_check, 3),
            "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            "eval_details": eval_details
        }

    def benchmark_vector_retrieval_scale(
        self,
        bank_sizes: List[int] = [1000, 10000, 50000, 100000]
    ) -> List[Dict[str, Any]]:
        """
        Benchmarks retrieval latency as the question collection scales from 1,000 to 100,000 vectors.
        """
        import time
        import numpy as np

        dim = self.detector.embedding_dim
        benchmark_results = []

        query_vec = np.random.randn(dim).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        for n in bank_sizes:
            # Generate normalized random vector matrix simulating bank embeddings
            bank_matrix = np.random.randn(n, dim).astype(np.float32)
            norms = np.linalg.norm(bank_matrix, axis=1, keepdims=True)
            bank_matrix /= np.maximum(norms, 1e-9)

            # Benchmark exact cosine dot product
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                scores = np.dot(bank_matrix, query_vec)
                top_idx = np.argmax(scores)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            avg_ms = float(np.mean(times))
            benchmark_results.append({
                "collection_size": n,
                "latency_ms": round(avg_ms, 3),
                "queries_per_sec": int(1000 / max(0.001, avg_ms))
            })

        return benchmark_results

    def demonstrate_generation_avoidance(
        self,
        generator: QuestionGeneratorWithAvoidance,
        corpus: List[Question]
    ) -> Dict[str, Any]:
        """
        Demonstrates the generation avoidance pipeline preventing duplicates of Q1-Q10.
        """
        request = QuestionGenerationRequest(
            subject="Physics",
            concept="reflection of light",
            num_questions=3,
            avoid_duplicates=True,
            context_exemplar_limit=4
        )
        gen_result = generator.generate_questions(request, corpus)
        return gen_result
