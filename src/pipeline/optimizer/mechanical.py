# Shim — canonical source is src/optimizer/mechanical.py
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.optimizer.mechanical import *  # noqa: F401, F403
