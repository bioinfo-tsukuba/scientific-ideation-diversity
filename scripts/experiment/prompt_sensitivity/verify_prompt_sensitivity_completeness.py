"""Verification acceptance test for prompt-sensitivity data (#181 Task E).

For each of 12 cells (3 models × 2 prompts × 2 efforts), checks:

1. ``samples.jsonl`` has no duplicate (keyword, sample_index) pairs.
2. Sample count matches expected ``K × 30`` where K is the number of unique
   keywords in the per-model q1_q4 CSV (Claude/Gemini: K=100 → 3000;
   GPT-5.4: K=99 → 2970 after bioterrorism drop).
3. Every keyword in the CSV has exactly 30 samples in the cell.
4. Each of 3 embedders has ``embeddings.npy`` length matching N.
5. Each of 4 judge dirs (2 judges × pairwise/quality) has manifest where
   ``completed_judgments == total_judgments_expected``, and the JSONL row
   count matches.
6. ``q1_q4_gpt54.csv`` does NOT contain bioterrorism.
7. Top-level ``errors.jsonl`` has no ``BadRequestError`` records (renamed
   bioterrorism-only files are excluded).
8. ``backfill_successes.jsonl`` does NOT exist (consolidate completed).

Prints a summary table. Exit 0 if all checks PASS, 1 otherwise.

Usage:
    uv run python scripts/experiment/prompt_sensitivity/verify_prompt_sensitivity_completeness.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

PROMPT_SENS_DIR = REPO_ROOT / "results" / "effort_diversity" / "prompt_sensitivity"
TARGET_DROPPED = "bioterrorism"

MODELS = ["claude", "gemini31pro", "gpt54"]
PROMPTS = ["vs", "ssot"]
EFFORTS = ["low", "high"]
EMBEDDERS = ["text-embedding-3-large", "allenaispecter2-adhoc-query", "amazontitan-embed-text-v20"]
JUDGE_DIRS_PAIRWISE = [
    "llm_judge_gpt_4_1",
    "llm_judge_claude_haiku_4_5_20251001",
]
JUDGE_DIRS_QUALITY = [
    "llm_judge_quality_gpt_4_1",
    "llm_judge_quality_claude_haiku_4_5_20251001",
]
KW_SUBSET_BY_MODEL = {
    "claude": "q1_q4_claude.csv",
    "gemini31pro": "q1_q4_gemini31pro.csv",
    "gpt54": "q1_q4_gpt54.csv",
}


def load_keywords(csv_path: Path) -> list[str]:
    with open(csv_path, encoding="utf-8") as f:
        return [row["keyword"] for row in csv.DictReader(f)]


def check_cell(cell: str) -> dict:
    """Return a dict with PASS/FAIL per check + supporting numbers."""
    parts = cell.split("_")
    model = parts[0]
    cell_dir = PROMPT_SENS_DIR / cell
    out: dict = {"cell": cell, "checks": {}, "values": {}}

    # CHECK 8: backfill_successes.jsonl should not exist
    bp = cell_dir / "backfill_successes.jsonl"
    out["checks"]["8_no_backfill"] = "PASS" if not bp.exists() else "FAIL"

    # Load samples.jsonl
    sp = cell_dir / "samples.jsonl"
    if not sp.exists():
        for c in ("1_no_dup", "2_count", "3_per_kw_30", "4_emb", "5_judge"):
            out["checks"][c] = "FAIL"
        out["values"]["samples_n"] = 0
        return out
    records = [json.loads(line) for line in open(sp)]
    out["values"]["samples_n"] = len(records)

    # CHECK 1: no duplicates
    keys = [(r["keyword"], r.get("sample_index")) for r in records]
    dup_count = sum(1 for k, v in Counter(keys).items() if v > 1)
    out["checks"]["1_no_dup"] = "PASS" if dup_count == 0 else f"FAIL ({dup_count} dups)"

    # Load expected K
    csv_path = PROMPT_SENS_DIR / KW_SUBSET_BY_MODEL[model]
    expected_keywords = load_keywords(csv_path)
    K = len(expected_keywords)
    expected_n = K * 30
    out["values"]["K"] = K
    out["values"]["expected_n"] = expected_n

    # CHECK 2: total count
    out["checks"]["2_count"] = "PASS" if len(records) == expected_n else f"FAIL ({len(records)} vs {expected_n})"

    # CHECK 3: per-keyword == 30, and keyword set matches CSV
    kw_counts = Counter(r["keyword"] for r in records)
    actual_kws = set(kw_counts.keys())
    expected_kws = set(expected_keywords)
    missing = expected_kws - actual_kws
    extra = actual_kws - expected_kws
    bad_count = [k for k, v in kw_counts.items() if v != 30]
    if missing or extra or bad_count:
        msg_parts = []
        if missing:
            msg_parts.append(f"{len(missing)} kw missing")
        if extra:
            msg_parts.append(f"{len(extra)} extra kw")
        if bad_count:
            msg_parts.append(f"{len(bad_count)} kw != 30 ideas")
        out["checks"]["3_per_kw_30"] = f"FAIL ({', '.join(msg_parts)})"
    else:
        out["checks"]["3_per_kw_30"] = "PASS"

    # CHECK 4: embeddings
    emb_results = []
    for emb in EMBEDDERS:
        ep = cell_dir / emb / "embeddings.npy"
        if ep.exists():
            n = np.load(ep).shape[0]
            emb_results.append((emb, n, n == len(records)))
        else:
            emb_results.append((emb, 0, False))
    all_ok = all(ok for _, _, ok in emb_results)
    out["checks"]["4_emb"] = "PASS" if all_ok else "FAIL (" + ", ".join(f"{e}={n}" for e, n, _ in emb_results) + ")"

    # CHECK 5: judges
    judge_problems = []
    expected_pairs_per_kw = 30  # top_m=10 + middle_m=10 + bottom_m=10
    expected_pairs_total = K * expected_pairs_per_kw * 2  # both_orders
    expected_quality_total = K * 30  # 1 quality score per sample

    for jdir in JUDGE_DIRS_PAIRWISE:
        base = cell_dir / jdir
        if not base.exists():
            judge_problems.append(f"{jdir} dir missing")
            continue
        mj = base / "manifest.json"
        if mj.exists():
            with open(mj) as f:
                m = json.load(f)
            exp = m.get("total_judgments_expected", 0)
            comp = m.get("completed_judgments", 0)
            if exp != expected_pairs_total:
                judge_problems.append(f"{jdir}: exp={exp}, want={expected_pairs_total}")
            if comp != exp:
                judge_problems.append(f"{jdir}: completed={comp} != expected={exp}")
        # Verify JSONL line count matches
        pj = base / "pairwise.jsonl"
        if pj.exists():
            n_pj = sum(1 for _ in open(pj))
            if n_pj != expected_pairs_total:
                judge_problems.append(f"{jdir}/pairwise.jsonl: {n_pj} lines vs expected {expected_pairs_total}")

    for jdir in JUDGE_DIRS_QUALITY:
        base = cell_dir / jdir
        if not base.exists():
            judge_problems.append(f"{jdir} dir missing")
            continue
        mj = base / "manifest.json"
        if mj.exists():
            with open(mj) as f:
                m = json.load(f)
            exp = m.get("total_judgments_expected", 0)
            comp = m.get("completed_judgments", 0)
            if exp != expected_quality_total:
                judge_problems.append(f"{jdir}: exp={exp}, want={expected_quality_total}")
            if comp != exp:
                judge_problems.append(f"{jdir}: completed={comp} != expected={exp}")
        qj = base / "quality.jsonl"
        if qj.exists():
            n_qj = sum(1 for _ in open(qj))
            if n_qj != expected_quality_total:
                judge_problems.append(f"{jdir}/quality.jsonl: {n_qj} lines vs expected {expected_quality_total}")

    out["checks"]["5_judge"] = "PASS" if not judge_problems else f"FAIL ({'; '.join(judge_problems[:2])})"

    # CHECK 7 (per-cell scope): no BadRequestError in active errors.jsonl
    ep = cell_dir / "errors.jsonl"
    if ep.exists() and ep.stat().st_size > 0:
        bad = sum(1 for line in open(ep) if json.loads(line).get("error_type") == "BadRequestError")
        out["checks"]["7_no_bad_request"] = "PASS" if bad == 0 else f"FAIL ({bad} BadRequestError)"
    else:
        out["checks"]["7_no_bad_request"] = "PASS"

    return out


def check_global() -> dict:
    """Global checks: q1_q4_gpt54.csv has no bioterrorism."""
    out = {"checks": {}}
    csv_path = PROMPT_SENS_DIR / "q1_q4_gpt54.csv"
    if not csv_path.exists():
        out["checks"]["6_csv_no_bio"] = "FAIL (csv not found)"
        return out
    keywords = load_keywords(csv_path)
    if TARGET_DROPPED in keywords:
        out["checks"]["6_csv_no_bio"] = "FAIL (bioterrorism still in csv)"
    else:
        out["checks"]["6_csv_no_bio"] = "PASS"
    out["values"] = {"q1_q4_gpt54_n": len(keywords)}
    return out


def main() -> int:
    print("# Verification: prompt-sensitivity data completeness (#181 Task E)")
    print(f"# Target: {PROMPT_SENS_DIR}")
    print()

    cells = [f"{m}_{p}_{e}" for m in MODELS for p in PROMPTS for e in EFFORTS]
    all_results = [check_cell(c) for c in cells]
    global_result = check_global()

    # Print summary table
    check_keys = ["1_no_dup", "2_count", "3_per_kw_30", "4_emb", "5_judge", "7_no_bad_request", "8_no_backfill"]
    header = ["cell", "K", "N"] + check_keys
    print(" | ".join(f"{h:<22}" if h == "cell" else f"{h:<14}" for h in header))
    print("-" * (22 + 3 + 14 * (len(header) - 1) + 3 * (len(header) - 1)))
    any_fail = False
    for r in all_results:
        row = [r["cell"], str(r["values"].get("K", "?")), str(r["values"].get("samples_n", "?"))]
        for ck in check_keys:
            status = r["checks"].get(ck, "?")
            row.append(status)
            if not status.startswith("PASS"):
                any_fail = True
        print(" | ".join(f"{v:<22}" if i == 0 else f"{v:<14}" for i, v in enumerate(row)))

    print()
    print(
        f"# Global check: {global_result['checks']['6_csv_no_bio']} (q1_q4_gpt54.csv kw count: {global_result.get('values', {}).get('q1_q4_gpt54_n', '?')})"
    )
    if not global_result["checks"]["6_csv_no_bio"].startswith("PASS"):
        any_fail = True

    print()
    if any_fail:
        # Print details of failures
        print("# Failure details:")
        for r in all_results:
            failed = [(k, v) for k, v in r["checks"].items() if not v.startswith("PASS")]
            if failed:
                print(f"  {r['cell']}:")
                for k, v in failed:
                    print(f"    {k}: {v}")
        print()
        print("RESULT: FAIL")
        return 1
    print("RESULT: ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
