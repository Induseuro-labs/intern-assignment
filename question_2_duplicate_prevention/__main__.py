"""
Question 2 CLI Entry Point: Question Generation & Duplicate Prevention Service.
Usage:
    python -m question_2_duplicate_prevention check "Define the reflection of light." "What is reflection?"
    python -m question_2_duplicate_prevention generate "Reflection of light" -n 5
    python -m question_2_duplicate_prevention list
    python -m question_2_duplicate_prevention eval
    python -m question_2_duplicate_prevention benchmark
    python -m question_2_duplicate_prevention report
    python -m question_2_duplicate_prevention serve
"""

from __future__ import annotations
import os
import sys
import argparse
import subprocess

from .corpus.models import Question, QuestionGenerationRequest, ClassificationCategory
from .corpus.repository import QuestionRepository
from .detector.detector import IndependentDuplicateDetector
from .generator.generator import QuestionGeneratorWithAvoidance
from .evaluator.evaluator import QuestionCorpusEvaluator

def get_repo():
    return QuestionRepository()

def get_detector():
    return IndependentDuplicateDetector()

def cmd_check(args):
    detector = get_detector()
    if args.q2:
        # Pair comparison
        cand = args.q1
        base = [Question(id="BASE", text=args.q2, concept="General")]
        res = detector.check_candidate(cand, base)
        print(f"\n==================== PAIRWISE DUPLICATE CHECK ====================")
        print(f"Candidate: \"{args.q1}\"")
        print(f"Baseline:  \"{args.q2}\"")
        print(f"Verdict:   {res.predicted_classification.value}")
        print(f"Score:     {res.similarity_score:.3f}")
        print(f"Stage:     {res.stage_resolved}")
        print(f"Reasoning: {res.reasoning}")
        print("=" * 66)
    else:
        # Check against entire repository
        repo = get_repo()
        res = detector.check_candidate(args.q1, repo.get_all())
        print(f"\n==================== CORPUS DUPLICATE CHECK ====================")
        print(f"Candidate:   \"{args.q1}\"")
        print(f"Verdict:     {res.predicted_classification.value}")
        print(f"Matched Q:   {res.matched_question_id or 'None'}")
        print(f"Similarity:  {res.similarity_score:.3f}")
        print(f"Stage:       {res.stage_resolved}")
        print(f"Reasoning:   {res.reasoning}")
        print("=" * 66)

def cmd_generate(args):
    repo = get_repo()
    detector = get_detector()
    gen = QuestionGeneratorWithAvoidance(detector)
    req = QuestionGenerationRequest(
        subject=args.subject or "Physics",
        concept=args.concept,
        num_questions=args.n,
        avoid_duplicates=True
    )
    print(f"\nGenerating {args.n} questions on '{args.concept}' with Duplicate Avoidance...")
    res = gen.generate_questions(req, repo.get_all())
    print(f"Exemplars budgeted in prompt: {len(res['exemplars_provided'])}")
    print(f"Accepted questions ({len(res['accepted_questions'])}):")
    for q in res["accepted_questions"]:
        print(f"  + [{q.id}] {q.text}")
    if res["rejected_duplicates"]:
        print(f"\nRejected by In-Loop Safeguard ({len(res['rejected_duplicates'])}):")
        for r in res["rejected_duplicates"]:
            print(f"  - [{r.candidate_id}] {r.candidate_text} (Score: {r.similarity_score:.3f})")

def cmd_list(args):
    repo = get_repo()
    qs = repo.get_all()
    print(f"\n==================== QUESTION BANK ({len(qs)} questions) ====================")
    print(f"{'ID':<6} | {'Concept':<22} | {'Question Text'}")
    print("-" * 75)
    for q in qs:
        print(f"{q.id:<6} | {q.concept[:20]:<22} | {q.text}")
    print("=" * 75)

def cmd_eval(args):
    detector = get_detector()
    evaluator = QuestionCorpusEvaluator(detector)
    res = evaluator.evaluate_official_corpus()
    print(f"\n==================== Q1-Q10 GROUND TRUTH EVALUATION ====================")
    print(f"{'ID':<5} | {'Expected':<22} | {'Predicted':<22} | {'Status':<6} | {'Sim'}")
    print("-" * 72)
    for r in res["rows"]:
        print(f"{r['ID']:<5} | {r['Expected'][:20]:<22} | {r['Predicted'][:20]:<22} | {r['Status']:<6} | {r['Sim Score']}")
    print("-" * 72)
    print(f"Accuracy: {res['accuracy']*100:.1f}% | Precision: {res['precision']:.3f} | Recall: {res['recall']:.3f} | F1: {res['f1_score']:.3f}")
    print("=" * 72)

def cmd_benchmark(args):
    detector = get_detector()
    evaluator = QuestionCorpusEvaluator(detector)
    print("\nRunning vector bank retrieval scale benchmark (1,000 to 100,000 vectors)...")
    res = evaluator.benchmark_vector_retrieval_scale()
    print(f"{'Bank Size (Vectors)':<22} | {'Latency (ms)':<15} | {'Throughput (QPS)'}")
    print("-" * 58)
    for b in res:
        print(f"{b['collection_size']:<22} | {b['latency_ms']:<15.3f} | {b['queries_per_sec']} queries/sec")
    print("=" * 58)

def cmd_report(args):
    print("Running evaluations and generating comprehensive REPORT.md...")
    script_path = os.path.join(os.path.dirname(__file__), "generate_report.py")
    subprocess.run([sys.executable, script_path], check=True)

def cmd_serve(args):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(root_dir, "app", "streamlit_app.py")
    print(f"Launching Streamlit UI from {app_path}...")
    subprocess.run(["streamlit", "run", app_path])

def main():
    parser = argparse.ArgumentParser(description="AI Question Generation & Duplicate Prevention Service")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check
    p_chk = subparsers.add_parser("check", help="Check candidate question for duplicates")
    p_chk.add_argument("q1", help="Candidate question text")
    p_chk.add_argument("q2", nargs="?", help="Optional second question for pairwise comparison")

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate questions with avoidance")
    p_gen.add_argument("concept", help="Subject concept (e.g. 'reflection of light')")
    p_gen.add_argument("-n", type=int, default=3, help="Number of questions to generate")
    p_gen.add_argument("--subject", default="Physics", help="Academic subject")

    # list
    subparsers.add_parser("list", help="List all questions in corpus")

    # eval
    subparsers.add_parser("eval", help="Evaluate detector on Q1-Q10 dataset")

    # benchmark
    subparsers.add_parser("benchmark", help="Benchmark vector retrieval speed at scale")

    # report
    subparsers.add_parser("report", help="Generate full markdown evaluation report")

    # serve
    subparsers.add_parser("serve", help="Launch interactive web browser UI")

    args = parser.parse_args()
    cmds = {
        "check": cmd_check,
        "generate": cmd_generate,
        "list": cmd_list,
        "eval": cmd_eval,
        "benchmark": cmd_benchmark,
        "report": cmd_report,
        "serve": cmd_serve,
    }
    cmds[args.command](args)

if __name__ == "__main__":
    main()
