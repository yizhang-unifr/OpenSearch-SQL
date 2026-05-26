# Shim — canonical source is src/optimizer/elevation_rewriter.py
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.optimizer.elevation_rewriter import *  # noqa: F401, F403
