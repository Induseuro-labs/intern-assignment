"""
Unified Command-Line Interface for Induseuro-Labs Intern Assignment.
Provides a single entry point for Question 1 (Image Catalog) and Question 2 (Duplicate Prevention).

Usage:
    python cli.py q1 list
    python cli.py q1 add <image_path> [--name <name>]
    python cli.py q1 compose "<prompt>" [--verify]
    python cli.py q1 detect <entry_id> [--force]
    python cli.py q1 report
    
    python cli.py q2 check "<q1>" ["<q2>"]
    python cli.py q2 generate "<concept>" [-n N]
    python cli.py q2 list
    python cli.py q2 eval
    python cli.py q2 benchmark
    python cli.py q2 report
    
    python cli.py test    (Runs all automated tests for both questions)
    python cli.py report  (Generates REPORT.md for both questions)
    python cli.py serve   (Launches the Streamlit Web Application)
"""

import sys
import subprocess

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return

    sub = sys.argv[1].lower()

    if sub == "q1":
        args = [sys.executable, "-m", "question_1_image_catalog"] + sys.argv[2:]
        subprocess.run(args)
    elif sub == "q2":
        args = [sys.executable, "-m", "question_2_duplicate_prevention"] + sys.argv[2:]
        subprocess.run(args)
    elif sub == "test":
        print(">>> Running Question 1 Test Scenarios...")
        subprocess.run([sys.executable, "question_1_image_catalog/run_tests.py"])
        print("\n>>> Running Question 2 Evaluation & Safeguards...")
        subprocess.run([sys.executable, "question_2_duplicate_prevention/run_tests.py"])
    elif sub == "report":
        print(">>> Compiling Question 1 Visual Report...")
        subprocess.run([sys.executable, "-m", "question_1_image_catalog", "report"])
        print("\n>>> Compiling Question 2 Benchmark Report...")
        subprocess.run([sys.executable, "-m", "question_2_duplicate_prevention", "report"])
    elif sub == "serve":
        if "--streamlit" in sys.argv:
            print(">>> Launching Streamlit Dashboard on port 8501...")
            subprocess.run(["streamlit", "run", "app/streamlit_app.py"])
        else:
            print(">>> Launching Tactile Skeuomorphic & Claymorphic Console on http://127.0.0.1:8000 ...")
            subprocess.run([sys.executable, "-m", "uvicorn", "app.server:app", "--host", "127.0.0.1", "--port", "8000"])
    else:
        print(f"Unknown command '{sub}'. Run 'python cli.py --help' for usage.")

if __name__ == "__main__":
    main()
