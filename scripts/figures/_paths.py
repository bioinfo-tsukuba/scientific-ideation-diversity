"""Shared output paths for the paper_story_v2 figure scripts.

Scripts in this directory write:
- PNG/PDF figures         → outputs/figures/
- CSV / JSON tables       → outputs/tables/
- Intermediate .npy cache → /tmp/paper_story_v2 (not tracked; regenerable)
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "tables"
CACHE_DIR = Path("/tmp/paper_story_v2")

FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
