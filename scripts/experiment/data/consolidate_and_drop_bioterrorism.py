"""Consolidate prompt-sensitivity cell artifacts and drop bioterrorism (#181).

For all 12 cells (3 models × 2 prompts × 2 efforts):
- Apply ``load_run_sample_records()`` canonical dedup → overwrite ``samples.jsonl``
- Delete top-level ``backfill_successes.jsonl`` (now redundant)
- Delete per-embedder ``backfill_successes.jsonl`` (per-embedder ``samples.jsonl``
  is already canonical-3000 from the embedding pipeline)

For 4 GPT-5.4 cells, additionally drop ``bioterrorism`` keyword from:
- ``samples.jsonl`` (top-level + per-embedder)
- ``embeddings.npy`` and facet ``embeddings_*.npy`` (3 embedders × 4 npy each)
- ``embedding_responses*.jsonl``
- ``llm_judge_*/{pairwise,quality}.jsonl`` (vs cells only — ssot judge already
  ran on the 2970 non-bioterrorism subset)
- ``llm_judge_*/manifest.json``: update completed_judgments / total_judgments_expected

For ``q1_q4_gpt54.csv`` and ``q1_q4_all_models.csv``: delete bioterrorism row.

For ``errors.jsonl`` in 4 GPT-5.4 cells: rename to ``errors.bioterrorism_dropped.jsonl``
so verification doesn't flag the dropped-keyword failures as residual.

Usage:
    uv run python scripts/experiment/data/consolidate_and_drop_bioterrorism.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.artifacts import load_run_sample_records  # noqa: E402

PROMPT_SENS_DIR = REPO_ROOT / "results" / "effort_diversity" / "prompt_sensitivity"

MODELS = ["claude", "gemini31pro", "gpt54"]
PROMPTS = ["vs", "ssot"]
EFFORTS = ["low", "high"]
GPT54_PREFIX = "gpt54"
TARGET_KEYWORD = "bioterrorism"

EMBEDDERS = ["text-embedding-3-large", "allenaispecter2", "amazontitan-embed-text-v20"]
FACET_NPY_NAMES = [
    "embeddings.npy",
    "embeddings_evaluation.npy",
    "embeddings_mechanism.npy",
    "embeddings_purpose.npy",
]
EMBEDDING_RESPONSES = [
    "embedding_responses.jsonl",
    "embedding_responses_evaluation.jsonl",
    "embedding_responses_mechanism.jsonl",
    "embedding_responses_purpose.jsonl",
]
JUDGE_DIRS_PAIRWISE = [
    "llm_judge_gpt_4_1",
    "llm_judge_claude_haiku_4_5_20251001",
]
JUDGE_DIRS_QUALITY = [
    "llm_judge_quality_gpt_4_1",
    "llm_judge_quality_claude_haiku_4_5_20251001",
]


def all_cells() -> list[str]:
    return [f"{m}_{p}_{e}" for m in MODELS for p in PROMPTS for e in EFFORTS]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def consolidate_top_level(cell_dir: Path) -> tuple[int, int]:
    """Overwrite top-level samples.jsonl with canonical dedup; delete backfill.

    Returns (records_before_dedup, records_after_dedup).
    """
    samples_path = cell_dir / "samples.jsonl"
    backfill_path = cell_dir / "backfill_successes.jsonl"

    raw_samples = sum(1 for _ in open(samples_path))
    raw_backfill = sum(1 for _ in open(backfill_path)) if backfill_path.exists() else 0
    raw_total = raw_samples + raw_backfill

    canonical = load_run_sample_records(cell_dir)
    write_jsonl(
        samples_path,
        [r.model_dump(mode="json") for r in canonical],
    )
    if backfill_path.exists():
        backfill_path.unlink()
    return raw_total, len(canonical)


def filter_bioterrorism_from_jsonl(path: Path, key_path: tuple[str, ...]) -> tuple[int, int]:
    """Filter records where the keyword (at key_path) equals bioterrorism.

    key_path is a tuple of keys to traverse, e.g. ("keyword",) or
    ("request", "pair", "keyword") or ("request", "target", "keyword").

    Returns (records_before, records_after).
    """
    records = read_jsonl(path)
    before = len(records)
    filtered = []
    for r in records:
        cur = r
        try:
            for key in key_path:
                cur = cur[key]
        except (KeyError, TypeError):
            cur = None
        if cur != TARGET_KEYWORD:
            filtered.append(r)
    write_jsonl(path, filtered)
    return before, len(filtered)


def filter_bioterrorism_from_samples_jsonl(path: Path) -> tuple[int, int]:
    """Drop records with keyword=bioterrorism. Returns (before, after)."""
    records = read_jsonl(path)
    before = len(records)
    filtered = [r for r in records if r.get("keyword") != TARGET_KEYWORD]
    write_jsonl(path, filtered)
    return before, len(filtered)


def filter_npy_by_indices(npy_path: Path, drop_indices: list[int]) -> tuple[int, int]:
    """Remove rows at drop_indices from a .npy file. Returns (before, after)."""
    arr = np.load(npy_path)
    before = arr.shape[0]
    if drop_indices:
        keep_mask = np.ones(before, dtype=bool)
        keep_mask[drop_indices] = False
        arr = arr[keep_mask]
    np.save(npy_path, arr)
    return before, arr.shape[0]


def update_manifest(manifest_path: Path, expected: int, completed: int) -> None:
    """Update manifest.json's total_judgments_expected and completed_judgments."""
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    m["total_judgments_expected"] = expected
    m["completed_judgments"] = completed
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)


def update_q1_q4_csvs(prompt_sens_dir: Path) -> dict[str, tuple[int, int]]:
    """Remove bioterrorism row from q1_q4_gpt54.csv and q1_q4_all_models.csv.

    Returns dict {csv_filename: (rows_before, rows_after)}.
    """
    out: dict[str, tuple[int, int]] = {}
    for fn in ["q1_q4_gpt54.csv", "q1_q4_all_models.csv"]:
        p = prompt_sens_dir / fn
        with open(p, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        before = len(rows)
        if fn == "q1_q4_all_models.csv":
            filtered = [
                r for r in rows if not (r["keyword"] == TARGET_KEYWORD and r.get("generation_model") == "GPT-5.4")
            ]
        else:
            filtered = [r for r in rows if r["keyword"] != TARGET_KEYWORD]
        with open(p, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(filtered)
        out[fn] = (before, len(filtered))
    return out


def process_cell(cell_dir: Path, cell_name: str) -> dict:
    """Process a single cell. Returns a summary dict for reporting."""
    summary: dict = {"cell": cell_name}
    is_gpt54 = cell_name.startswith(GPT54_PREFIX)
    is_vs = "_vs_" in cell_name

    # Step A: consolidate top-level samples.jsonl
    raw_total, canonical_n = consolidate_top_level(cell_dir)
    summary["consolidated_raw_total"] = raw_total
    summary["canonical_n"] = canonical_n

    # Step A': delete per-embedder backfill_successes.jsonl (already redundant)
    for emb in EMBEDDERS:
        bp = cell_dir / emb / "backfill_successes.jsonl"
        if bp.exists():
            bp.unlink()

    # GPT-5.4 only: drop bioterrorism from everything
    if is_gpt54:
        # Find bioterrorism indices in canonical record list (BEFORE filtering top-level)
        # Re-load canonical to get records (with their order)
        canonical_records = load_run_sample_records(cell_dir)
        # samples.jsonl was just overwritten, canonical_records gives current state
        bio_indices = [i for i, r in enumerate(canonical_records) if r.keyword == TARGET_KEYWORD]
        summary["bio_indices_count"] = len(bio_indices)

        # Top-level samples.jsonl: drop bioterrorism
        before, after = filter_bioterrorism_from_samples_jsonl(cell_dir / "samples.jsonl")
        summary["top_samples_before_after"] = (before, after)

        # Per-embedder samples.jsonl + embeddings.npy + facet embeddings + embedding_responses
        for emb in EMBEDDERS:
            emb_dir = cell_dir / emb
            # samples.jsonl
            sp = emb_dir / "samples.jsonl"
            if sp.exists():
                filter_bioterrorism_from_samples_jsonl(sp)
            # All npy files
            for npy_name in FACET_NPY_NAMES:
                np_path = emb_dir / npy_name
                if np_path.exists():
                    filter_npy_by_indices(np_path, bio_indices)
            # embedding_responses jsonl (keyword is at top level)
            for er_name in EMBEDDING_RESPONSES:
                er_path = emb_dir / er_name
                if er_path.exists():
                    filter_bioterrorism_from_jsonl(er_path, ("keyword",))

        # Judge: vs cells only (ssot already ran on 2970 non-bioterrorism subset)
        if is_vs:
            for jdir in JUDGE_DIRS_PAIRWISE:
                base = cell_dir / jdir
                if not base.exists():
                    continue
                pj = base / "pairwise.jsonl"
                if pj.exists():
                    before, after = filter_bioterrorism_from_jsonl(pj, ("request", "pair", "keyword"))
                    summary[f"{jdir}_pairwise"] = (before, after)
                    mj = base / "manifest.json"
                    if mj.exists():
                        update_manifest(mj, expected=after, completed=after)
            for jdir in JUDGE_DIRS_QUALITY:
                base = cell_dir / jdir
                if not base.exists():
                    continue
                qj = base / "quality.jsonl"
                if qj.exists():
                    before, after = filter_bioterrorism_from_jsonl(qj, ("request", "target", "keyword"))
                    summary[f"{jdir}_quality"] = (before, after)
                    mj = base / "manifest.json"
                    if mj.exists():
                        update_manifest(mj, expected=after, completed=after)

        # errors.jsonl rename
        ep = cell_dir / "errors.jsonl"
        if ep.exists():
            ep.rename(cell_dir / "errors.bioterrorism_dropped.jsonl")

    return summary


def main() -> int:
    if not PROMPT_SENS_DIR.exists():
        print(f"FATAL: {PROMPT_SENS_DIR} does not exist", file=sys.stderr)
        return 1

    print("# Consolidate + drop bioterrorism (#181)")
    print(f"# Working dir: {PROMPT_SENS_DIR}")
    print()

    summaries = []
    for cell in all_cells():
        cell_dir = PROMPT_SENS_DIR / cell
        if not cell_dir.exists():
            print(f"SKIP {cell}: directory not found")
            continue
        print(f"Processing {cell}...")
        s = process_cell(cell_dir, cell)
        summaries.append(s)
        # Compact summary
        cn = s["canonical_n"]
        bio = s.get("bio_indices_count", 0)
        if bio:
            print(f"  canonical={cn}, bioterrorism removed={bio}, final samples={cn - bio}")
        else:
            print(f"  canonical={cn} (no bioterrorism)")

    # CSV updates
    print()
    print("# Updating q1_q4 CSVs")
    csv_summary = update_q1_q4_csvs(PROMPT_SENS_DIR)
    for fn, (before, after) in csv_summary.items():
        print(f"  {fn}: {before} → {after} rows")

    print()
    print("# Done. Run verify_prompt_sensitivity_completeness.py to confirm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
