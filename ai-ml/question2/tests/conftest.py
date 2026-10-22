import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["QGEN_OFFLINE"] = "1"  # tests never call a hosted LLM
