"""
Master Demo and Verification Runner.
Executes complete end-to-end test suites for both:
- Question 1: Image catalog, object detection, metadata, composite generation
- Question 2: AI question generation and duplicate prevention
"""

import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 80)
    print(" INDUSEURO-LABS INTERN ASSIGNMENT - MASTER TEST & DEMO RUNNER ")
    print("=" * 80)

    # 1. Run Question 1
    print("\n>>> LAUNCHING QUESTION 1 SUITE (Image Catalog & Composite Engine)...")
    q1_script = os.path.join(PROJECT_ROOT, "question_1_image_catalog", "run_tests.py")
    res1 = subprocess.run([sys.executable, q1_script], cwd=PROJECT_ROOT)
    if res1.returncode != 0:
        print("[ERROR] Question 1 test suite failed!")
        sys.exit(res1.returncode)

    # 2. Run Question 2
    print("\n>>> LAUNCHING QUESTION 2 SUITE (Question Generation & Duplicate Prevention)...")
    q2_script = os.path.join(PROJECT_ROOT, "question_2_duplicate_prevention", "run_tests.py")
    res2 = subprocess.run([sys.executable, q2_script], cwd=PROJECT_ROOT)
    if res2.returncode != 0:
        print("[ERROR] Question 2 test suite failed!")
        sys.exit(res2.returncode)

    print("\n" + "#" * 80)
    print(" [COMPLETE SUCCESS] ALL TESTS FOR BOTH QUESTION 1 & QUESTION 2 PASSED! ")
    print("#" * 80)

if __name__ == "__main__":
    main()
