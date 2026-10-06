"""k-NN purity sweep for the 12 prompt × effort cell pairs (#150 sensitivity check).

Companion to ``analyze_prompt_sensitivity_embedding_purity.py``: rather than
fixing k = 1, sweep k ∈ {1, 3, 5, 10} and report **soft purity** (mean fraction
of a point's k nearest neighbors that share its cell label). At k = 1 this
reduces to the 1-NN purity used in the main bar chart; for k > 1 it captures
the local neighbourhood composition rather than the single closest point's
identity, which is more robust to the boundary-flip noise of 1-NN.

Output:
    results/effort_diversity/prompt_sensitivity/<model>_embedding_purity_ksweep[__<embedder>].csv

Columns: label, stratum, k, n, purity_mean, purity_ci_low, purity_ci_high

Usage:
    uv run python scripts/experiment/prompt_sensitivity/analyze_prompt_sensitivity_knn_sweep.py \\
        --model claude --embedding-model text-embedding-3-large
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment.prompt_sensitivity.analyze_prompt_sensitivity_embedding_purity import (  # noqa: E402
    _SUPPORTED_EMBEDDING_MODELS,
    CELL_PAIRS,
    MODELS,
    bootstrap_mean_ci,
    load_cell_vectors,
    resolve_cell_dir,
)
from src.model_registry import EmbeddingModelName  # noqa: E402
from src.text import slugify  # noqa: E402

RESULTS = REPO_ROOT / "results" / "effort_diversity"


def soft_knn_purity(vec_a: np.ndarray, vec_b: np.ndarray, k_values: list[int]) -> dict[int, float]:
    """For each k, mean fraction of point's k nearest neighbours sharing its cell label.

    At k = 1 this matches argmax-based 1-NN purity. For k > 1 it averages over
    the top-k same-label fractions (continuous in [0, 1]) rather than majority-
    voting, which exposes how cleanly the neighbourhood is dominated by one
    cell rather than only the binary "is the majority same-cell" question.
    """
    n_a, n_b = len(vec_a), len(vec_b)
    if n_a < 2 or n_b < 2:
        return {k: float("nan") for k in k_values}

    union = np.vstack([vec_a, vec_b])
    labels = np.concatenate([np.zeros(n_a, dtype=int), np.ones(n_b, dtype=int)])
    norm = union / np.linalg.norm(union, axis=1, keepdims=True)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -np.inf)

    max_k = max(k_values)
    # argpartition gives unsorted top-max_k columns; sort within those by similarity.
    top_k = np.argpartition(-sim, max_k - 1, axis=1)[:, :max_k]
    top_k_sims = np.take_along_axis(sim, top_k, axis=1)
    sort_idx = np.argsort(-top_k_sims, axis=1)
    top_k_sorted = np.take_along_axis(top_k, sort_idx, axis=1)
    nn_labels = labels[top_k_sorted]
    same = (nn_labels == labels[:, None]).astype(float)

    return {k: float(same[:, :k].mean()) for k in k_values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude", choices=list(MODELS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument(
        "--k-neighbors",
        type=str,
        default="1,3,5,10",
        help="Comma-separated list of k values to sweep (default: 1,3,5,10).",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--embedding-model",
        type=EmbeddingModelName,
        default=EmbeddingModelName.TEXT_EMBEDDING_3_LARGE,
        choices=list(_SUPPORTED_EMBEDDING_MODELS),
        metavar=f"{{{','.join(m.value for m in _SUPPORTED_EMBEDDING_MODELS)}}}",
    )
    args = parser.parse_args()

    k_values = sorted(int(x) for x in args.k_neighbors.split(",") if x.strip())
    model_cfg = MODELS[args.model]
    rng = np.random.default_rng(args.seed)
    emb_subdir = slugify(args.embedding_model.value)

    q1q4 = pd.read_csv(RESULTS / "prompt_sensitivity" / f"q1_q4_{args.model}.csv")
    rows: list[dict] = []

    for label, (kind_a, eff_a), (kind_b, eff_b) in CELL_PAIRS:
        dir_a = resolve_cell_dir(model_cfg=model_cfg, kind=kind_a, effort=eff_a)
        dir_b = resolve_cell_dir(model_cfg=model_cfg, kind=kind_b, effort=eff_b)
        if not (dir_a / emb_subdir / "embeddings.npy").exists() or not (dir_b / emb_subdir / "embeddings.npy").exists():
            for k in k_values:
                rows.append({"label": label, "stratum": "—", "k": k, "skipped": "missing cell"})
            continue

        per_kw: list[dict] = []
        for _, kw_row in q1q4.iterrows():
            kw = kw_row["keyword"]
            vec_a = load_cell_vectors(run_dir=dir_a, keyword=kw, effort=eff_a, emb_subdir=emb_subdir)
            vec_b = load_cell_vectors(run_dir=dir_b, keyword=kw, effort=eff_b, emb_subdir=emb_subdir)
            ks = soft_knn_purity(vec_a, vec_b, k_values)
            per_kw.append({"keyword": kw, "stratum": kw_row["stratum"], **{f"k{k}": ks[k] for k in k_values}})

        per_kw_df = pd.DataFrame(per_kw)
        for stratum in ("Q1", "Q4"):
            sub = per_kw_df[per_kw_df["stratum"] == stratum]
            for k in k_values:
                m, lo, hi = bootstrap_mean_ci(sub[f"k{k}"].to_numpy(), rng=rng, n_boot=args.n_boot)
                rows.append(
                    {
                        "label": label,
                        "stratum": stratum,
                        "k": k,
                        "n": len(sub),
                        "purity_mean": m,
                        "purity_ci_low": lo,
                        "purity_ci_high": hi,
                    }
                )

    df = pd.DataFrame(rows)

    print()
    print(f"=== k-NN purity sweep on {args.model} / {args.embedding_model.value} ===")
    pivot = (
        df.dropna(subset=["purity_mean"])
        .pivot_table(index=["label", "stratum"], columns="k", values="purity_mean")
        .round(3)
    )
    print(pivot.to_string())

    if args.output_csv is not None:
        out = args.output_csv
    else:
        if args.embedding_model is EmbeddingModelName.TEXT_EMBEDDING_3_LARGE:
            out_name = f"{args.model}_embedding_purity_ksweep.csv"
        else:
            out_name = f"{args.model}_embedding_purity_ksweep__{emb_subdir}.csv"
        out = RESULTS / "prompt_sensitivity" / out_name
    df.to_csv(out, index=False)
    print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
