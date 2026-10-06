"""Post-hoc: compute Vendi Score per (model, embedding, keyword, effort) group.

Loads ``samples.jsonl`` + ``embeddings.npy`` from each phase-2 run directory
and each embedding subdirectory under ``results/effort_diversity/``, groups
records by (category, keyword, prompt_style, effort), and computes the
Vendi Score~\\citep{friedmanVendiScoreDiversity2023} from the matching
embedding slice using the cosine kernel.

Reads only existing artefacts; does not regenerate or re-embed.

Output: ``results/effort_diversity/vendi_score/vendi_score_by_keyword_effort.csv``
columns: model, embedding, category, keyword, prompt_style, effort, n, vendi_score

Usage:
    uv run python scripts/experiment/diversity/compute_vendi_score.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.metrics import vendi_score  # noqa: E402
from src.model_registry import IDEA_MODEL_PHASE2_DIR  # noqa: E402
from src.schemas.sample import SampleRecord  # noqa: E402

EFFORT_DIR = REPO_ROOT / "results" / "effort_diversity"
EMBEDDING_SUBDIRS = (
    "allenaispecter2-adhoc-query",
    "amazontitan-embed-text-v20",
    "text-embedding-3-large",
)
OUTPUT_DIR = EFFORT_DIR / "vendi_score"
OUTPUT_PATH = OUTPUT_DIR / "vendi_score_by_keyword_effort.csv"


def load_samples(samples_path: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with samples_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(SampleRecord.model_validate_json(line))
    return records


def compute_for_embedding_dir(embedding_dir: Path) -> list[dict[str, object]]:
    samples_path = embedding_dir / "samples.jsonl"
    embeddings_path = embedding_dir / "embeddings.npy"
    if not samples_path.exists() or not embeddings_path.exists():
        return []

    records = load_samples(samples_path)
    embeddings = np.load(embeddings_path)
    if len(records) != embeddings.shape[0]:
        raise ValueError(f"sample count {len(records)} != embedding rows {embeddings.shape[0]} in {embedding_dir}")

    groups: dict[tuple[str, str, str, str], list[int]] = {}
    for i, record in enumerate(records):
        key = (record.category or "", record.keyword, record.prompt_style, record.effort)
        groups.setdefault(key, []).append(i)

    rows: list[dict[str, object]] = []
    for (category, keyword, prompt_style, effort), indices in groups.items():
        rows.append(
            {
                "category": category,
                "keyword": keyword,
                "prompt_style": prompt_style,
                "effort": effort,
                "n": len(indices),
                "vendi_score": vendi_score(embeddings[indices]),
            }
        )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    for model, run_subdir in IDEA_MODEL_PHASE2_DIR.items():
        run_dir = EFFORT_DIR / run_subdir
        if not run_dir.exists():
            print(f"[skip] missing run dir: {run_dir}")
            continue
        for embedding_subdir in EMBEDDING_SUBDIRS:
            embedding_dir = run_dir / embedding_subdir
            if not embedding_dir.exists():
                print(f"[skip] missing embedding dir: {embedding_dir}")
                continue
            print(f"[run] {model.value} / {embedding_subdir}")
            rows = compute_for_embedding_dir(embedding_dir)
            for row in rows:
                row["model"] = model.value
                row["embedding"] = embedding_subdir
            all_rows.extend(rows)
            print(f"       {len(rows)} (keyword, effort) groups")

    fieldnames = [
        "model",
        "embedding",
        "category",
        "keyword",
        "prompt_style",
        "effort",
        "n",
        "vendi_score",
    ]
    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"\nWrote {len(all_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
