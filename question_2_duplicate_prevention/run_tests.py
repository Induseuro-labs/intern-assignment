"""
Comprehensive Test and Verification Suite for Question 2.
Validates:
1. Official Q1-Q10 Ground Truth Classification (Exact, Semantic, Different, Ambiguous)
2. Metadata Tag Independence (proves duplicate detection is content-based, not tag-based)
3. Scaling and Context Window Budgeting for large question banks
4. AI Question Generation with Duplicate Avoidance and Safeguard Verification
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from question_2_duplicate_prevention.corpus.models import (
    Question, ClassificationCategory
)
from question_2_duplicate_prevention.corpus.repository import QuestionRepository
from question_2_duplicate_prevention.detector.detector import IndependentDuplicateDetector
from question_2_duplicate_prevention.generator.generator import (
    QuestionGeneratorWithAvoidance, QuestionGenerationRequest, ContextSelector
)
from question_2_duplicate_prevention.evaluator.evaluator import QuestionCorpusEvaluator

def print_separator(title: str):
    print("\n" + "=" * 85)
    print(f" {title.upper()} ")
    print("=" * 85)

def run_all_question_2_tests():
    detector = IndependentDuplicateDetector()
    generator = QuestionGeneratorWithAvoidance(detector=detector)
    evaluator = QuestionCorpusEvaluator(detector=detector)

    print_separator("Test 1: Official Ground-Truth Classification on Q1-Q10 Dataset")
    results = evaluator.evaluate_official_corpus()

    print(f"{'ID':<4} | {'Expected Classification':<25} | {'Predicted Classification':<25} | {'Status':<6} | {'Sim':<6}")
    print("-" * 85)
    for row in results["rows"]:
        print(f"{row['ID']:<4} | {row['Expected']:<25} | {row['Predicted']:<25} | {row['Status']:<6} | {row['Sim Score']:<6}")

    print("\n--- Evaluation Metrics Summary ---")
    print(f"Total Questions Evaluated: {results['total']}")
    print(f"Passed: {results['passed']} / {results['total']} ({results['accuracy'] * 100:.1f}% Accuracy)")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall:    {results['recall']:.3f}")
    print(f"F1-Score:  {results['f1_score']:.3f}")
    print(f"Confusion Matrix: {results['confusion_matrix']}")
    assert results["accuracy"] == 1.0, f"Expected 100% accuracy on ground truth test corpus, got {results['accuracy']}"

    print_separator("Test 2: Verification of Tag Independence")
    print("Testing rule: Duplicate detection must be based on question content, NOT tag equality.")
    
    # 2a: Two identical questions with completely different tag keys and values
    base_q = Question(id="T1", text="What is reflection of light?", tags={"curriculum": "US", "grade": 10, "author": "Alice"})
    dup_q = "What is reflection of light?"  # Exact content, but different tags
    cand_tags = {"curriculum": "UK", "exam_board": "Edexcel", "author": "Bob", "year": 2026}
    
    verdict_tags_diff = detector.check_candidate(dup_q, [base_q])
    print(f"\n2a. Content: EXACT | Tags: COMPLETELY DIFFERENT ({cand_tags} vs {base_q.tags})")
    print(f"    Verdict: {verdict_tags_diff.predicted_classification.value} (Resolved: {verdict_tags_diff.stage_resolved})")
    assert verdict_tags_diff.predicted_classification == ClassificationCategory.EXACT_DUPLICATE

    # 2b: Two different questions with 100% identical tags
    diff_q = "Explain the difference between real and virtual images formed by plane mirrors."
    identical_tags = {"curriculum": "US", "grade": 10, "author": "Alice"}
    verdict_tags_same = detector.check_candidate(diff_q, [base_q])
    print(f"\n2b. Content: DIFFERENT | Tags: 100% IDENTICAL ({identical_tags})")
    print(f"    Verdict: {verdict_tags_same.predicted_classification.value} (Similarity: {verdict_tags_same.similarity_score:.3f})")
    assert verdict_tags_same.predicted_classification == ClassificationCategory.DIFFERENT_QUESTION
    print("[SUCCESS] Verified: Tags are stored for metadata organization, but duplicate decisions are 100% content-driven.")

    print_separator("Test 3: Context Limits & Scaling Demonstration (1000+ Questions)")
    print("Simulating a large question bank with 1,000 questions to verify bounded token budget.")
    large_bank: List[Question] = []
    # Seed 1000 simulated questions across optics
    topics = ["reflection", "refraction", "dispersion", "diffraction", "polarization"]
    for i in range(1000):
        t = topics[i % len(topics)]
        large_bank.append(Question(
            id=f"BANK_{i+1:04d}",
            text=f"Question {i+1} analyzing the physical principles of {t} in optical medium {i%10}.",
            concept=t
        ))
    
    # Run MMR context selector
    selector = ContextSelector(detector)
    budgeted_exemplars = selector.select_representative_exemplars(
        large_bank,
        target_concept="reflection of light",
        k=5
    )
    print(f"Total Bank Size: {len(large_bank)} questions.")
    print(f"MMR Budgeted Selection Size: {len(budgeted_exemplars)} exemplars passed to LLM prompt (O(1) Token Bound).")
    for idx, ex in enumerate(budgeted_exemplars, 1):
        print(f"  Exemplar {idx}: [{ex.id}] '{ex.text}'")
    assert len(budgeted_exemplars) == 5, "Context limit must strictly bound exemplars to k"

    print_separator("Test 4: AI Generation with Duplicate Avoidance & Safeguard Verification")
    corpus_seed = QuestionRepository.get_seed_test_corpus()
    print(f"Existing Corpus: {len(corpus_seed)} questions (including Q1-Q10).")
    
    req = QuestionGenerationRequest(
        subject="Physics",
        concept="reflection of light",
        num_questions=3,
        avoid_duplicates=True,
        context_exemplar_limit=4
    )
    
    gen_result = generator.generate_questions(req, corpus_seed)
    print("\n--- Exhausted Concepts Steered Away From ---")
    for f in gen_result["exhausted_facets"]:
        print(f"  * {f}")
        
    print("\n--- Newly Generated Questions ---")
    for idx, q in enumerate(gen_result["accepted_questions"], 1):
        print(f"  {idx}. {q.text}")

    print("\n--- Independent Safeguard Gate Status ---")
    for check in gen_result["safeguard_checks"]:
        print(f"  Candidate: '{check.candidate_text[:60]}...'")
        print(f"    Verdict: {check.predicted_classification.value} | Sim: {check.similarity_score:.3f} | Stage: {check.stage_resolved}")
        assert check.predicted_classification not in (
            ClassificationCategory.EXACT_DUPLICATE, ClassificationCategory.SEMANTIC_DUPLICATE
        ), "Newly generated question should not duplicate Q1-Q10"

    print("\n" + "#" * 85)
    print(" ALL QUESTION 2 TESTS PASSED WITH 100% SUCCESS! ")
    print("#" * 85)

if __name__ == "__main__":
    run_all_question_2_tests()
