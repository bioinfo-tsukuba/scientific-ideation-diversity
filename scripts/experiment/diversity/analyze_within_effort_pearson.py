"""Task 4 (issue #185): Within-effort Pearson r between pair distance and generation tokens.

For each (model, effort) tier, computes Pearson r between per-keyword:
  - mean pairwise cosine distance (from allenaispecter2-adhoc-query/summary_by_keyword_effort.csv)
  - mean output tokens per idea  (mean_output_tokens from same summary CSV;
                                   includes reasoning tokens for thinking models)

Uses bootstrap 95% CI (resample keywords with replacement).

Output: results/effort_diversity/prompt_sensitivity/within_effort_pearson_r.csv
        (12 rows: 3 model × 4 effort tiers {none,low,medium,high})

Usage:
    uv run python scripts/experiment/diversity/analyze_within_effort_pearson.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

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
PAIR_COL = "mean_pairwise_cosine_distance"
N_BOOT = 10_000
EFFORT_ORDER = ["none", "low", "medium", "high"]


def bootstrap_pearson_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    point_r, _ = pearsonr(x, y)
    n = len(x)
    boot_rs = []
    for _ in range(N_BOOT):
        idx = rng.choice(n, size=n, replace=True)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        r, _ = pearsonr(xb, yb)
        boot_rs.append(r)
    if not boot_rs:
        return float(point_r), float("nan"), float("nan")
    return float(point_r), float(np.percentile(boot_rs, 2.5)), float(np.percentile(boot_rs, 97.5))


def main() -> None:
    rng = np.random.default_rng(42)
    rows: list[dict] = []

    for idea_model in IdeaModelName:
        model_key = IDEA_MODEL_SHORT_KEY[idea_model]
        phase2_dir = EFFORT_DIR / IDEA_MODEL_PHASE2_DIR[idea_model]

        # Per-keyword pair distances and output tokens from same summary CSV
        summary = pd.read_csv(phase2_dir / EMB_SUBDIR / "summary_by_keyword_effort.csv")
        merged = summary[["keyword", "effort", PAIR_COL, "mean_output_tokens"]].rename(
            columns={PAIR_COL: "pair_dist", "mean_output_tokens": "median_gen_tokens"}
        )

        for effort in EFFORT_ORDER:
            sub = merged[merged["effort"] == effort]
            if len(sub) < 3:
                continue
            x = sub["median_gen_tokens"].to_numpy()
            y = sub["pair_dist"].to_numpy()
            r, ci_lo, ci_hi = bootstrap_pearson_ci(x, y, rng)
            rows.append(
                {
                    "model": model_key,
                    "model_label": IDEA_MODEL_LABEL[idea_model],
                    "effort": effort,
                    "n_keywords": len(sub),
                    "pearson_r": r,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "mean_gen_tokens": float(x.mean()),
                    "mean_pair_dist": float(y.mean()),
                }
            )

    out = pd.DataFrame(rows)
    out_path = PS_DIR / "within_effort_pearson_r.csv"
    out.to_csv(out_path, index=False, float_format="%.4f")
    print(out.to_string(index=False))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
