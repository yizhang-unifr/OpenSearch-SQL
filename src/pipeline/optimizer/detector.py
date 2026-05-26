# Shim — canonical source is src/optimizer/detector.py
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.optimizer.detector import *  # noqa: F401, F403
from src.optimizer.detector import _ROUND_IN_RE, _PAIR_RE  # private symbols used by __init__
