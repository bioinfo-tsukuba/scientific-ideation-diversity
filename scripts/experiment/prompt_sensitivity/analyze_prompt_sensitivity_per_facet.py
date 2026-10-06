"""Task 2 (issue #185): Per-facet pair distance across prompt × effort cells.

Computes within-keyword mean pairwise cosine distance for each (model, prompt,
effort, facet, stratum=Q1/Q4) combination using SPECTER2 per-facet embeddings.

Output: results/effort_diversity/prompt_sensitivity/<model>_per_facet_pair_distance.csv
        (~108 rows per model: 3 prompt × 2 effort × 3 facet × 2 stratum + phase2 ref rows)

Usage:
    uv run python scripts/experiment/prompt_sensitivity/analyze_prompt_sensitivity_per_facet.py [--model claude|gpt54|gemini31pro|all]
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model_registry import (  # noqa: E402
    IDEA_MODEL_LABEL,
    IDEA_MODEL_PHASE2_DIR,
    IDEA_MODEL_SHORT_KEY,
    IdeaModelName,
)

EFFORT_DIR = REPO_ROOT / "results" / "effort_diversity"
PS_DIR = EFFORT_DIR / "prompt_sensitivity"
EMB_SUBDIR = "allenaispecter2-adhoc-query"
FACETS = ("purpose", "mechanism", "evaluation")
N_BOOT = 10_000

# Ordered for CLI --model argument
MODELS: dict[str, IdeaModelName] = {IDEA_MODEL_SHORT_KEY[m]: m for m in IdeaModelName}


def load_kw_effort(samples_path: Path) -> list[tuple[str, str]]:
    rows = []
    with samples_path.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["keyword"], r["effort"]))
    return rows


def per_kw_mean_pd(emb: np.ndarray, kw_eff: list[tuple[str, str]], kw_set: set[str], effort: str) -> dict[str, float]:
    indices_by_kw: dict[str, list[int]] = {}
    for i, (kw, eff) in enumerate(kw_eff):
        if kw in kw_set and eff == effort:
            indices_by_kw.setdefault(kw, []).append(i)

    result: dict[str, float] = {}
    for kw, idxs in indices_by_kw.items():
        if len(idxs) < 2:
            continue
        sub = emb[idxs]
        norms = np.linalg.norm(sub, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = sub / norms
        sim = normed @ normed.T
        n = len(idxs)
        pairs = [1.0 - float(sim[i, j]) for i, j in combinations(range(n), 2)]
        result[kw] = float(np.mean(pairs))
    return result


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(values, size=(N_BOOT, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_model(model: str, rng: np.random.Generator) -> pd.DataFrame:
    idea_model = MODELS[model]
    phase2_dir = EFFORT_DIR / IDEA_MODEL_PHASE2_DIR[idea_model]
    kw_df = pd.read_csv(PS_DIR / f"q1_q4_{model}.csv")
    q1_kws = set(kw_df.loc[kw_df["stratum"] == "Q1", "keyword"])
    q4_kws = set(kw_df.loc[kw_df["stratum"] == "Q4", "keyword"])
    all_kws = q1_kws | q4_kws

    rows: list[dict] = []

    def _add_rows(prompt_label: str, emb_dir: Path, samples_jsonl: Path, effort: str) -> None:
        kw_eff = load_kw_effort(samples_jsonl)
        for facet in FACETS:
            emb = np.load(emb_dir / f"embeddings_{facet}.npy")
            pd_by_kw = per_kw_mean_pd(emb, kw_eff, all_kws, effort)
            for stratum_label, stratum_kws in [("Q1", q1_kws), ("Q4", q4_kws)]:
                vals = np.array([pd_by_kw[kw] for kw in stratum_kws if kw in pd_by_kw])
                mean, ci_lo, ci_hi = bootstrap_ci(vals, rng)
                rows.append(
                    {
                        "model": model,
                        "model_label": IDEA_MODEL_LABEL[idea_model],
                        "prompt": prompt_label,
                        "effort": effort,
                        "facet": facet,
                        "stratum": stratum_label,
                        "n_keywords": len(vals),
                        "mean_pd": mean,
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                    }
                )

    # Default (phase2) — low and high effort
    phase2_emb_dir = phase2_dir / EMB_SUBDIR
    phase2_samples = phase2_emb_dir / "samples.jsonl"
    for effort in ("low", "high"):
        _add_rows("default", phase2_emb_dir, phase2_samples, effort)

    # VS and SSoT prompt-sensitivity cells
    for prompt_slug, prompt_label in [("vs", "VS"), ("ssot", "SSoT")]:
        for effort in ("low", "high"):
            cell_dir = PS_DIR / f"{model}_{prompt_slug}_{effort}"
            cell_emb_dir = cell_dir / EMB_SUBDIR
            cell_samples = cell_emb_dir / "samples.jsonl"
            _add_rows(prompt_label, cell_emb_dir, cell_samples, effort)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all", choices=["all", *MODELS.keys()])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    targets = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model in targets:
        print(f"\n=== {IDEA_MODEL_LABEL[MODELS[model]]} ===")
        df = run_model(model, rng)
        out_path = PS_DIR / f"{model}_per_facet_pair_distance.csv"
        df.to_csv(out_path, index=False, float_format="%.6f")
        print(df.to_string(index=False))
        print(f"\nSaved → {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
