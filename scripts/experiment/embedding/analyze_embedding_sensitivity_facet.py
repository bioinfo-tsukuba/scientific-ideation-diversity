#!/usr/bin/env python3
"""Per-facet embedding sensitivity analysis.

Reads facet_embeddings_*.npz (one row per (sample, facet)) and computes:
  1. Intra-keyword: pairs within same (keyword, facet, effort)
  2. Inter-keyword: pairs across different keywords (same facet, effort)
  3. Random: uniform random pairs (same facet, effort)

SPECTER2 here uses the adhoc_query adapter (appropriate for short text),
not the base adapter.

Usage:
    uv run python scripts/analyze_embedding_sensitivity_facet.py \
        --run-dir results/effort_diversity/phase1_claude_22kw_50x4_facet_seed42
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import normalize_rows
from src.model_registry import EffortName

EFFORT_ORDER = [e.value for e in EffortName]
FACET_ORDER = ["purpose", "mechanism", "evaluation"]

EMBEDDING_FILES: list[dict[str, str]] = [
    {"name": "Titan v2", "file": "facet_embeddings_titan.npz"},
    {"name": "SPECTER2-adhoc", "file": "facet_embeddings_specter2-adhoc.npz"},
    {"name": "OpenAI", "file": "facet_embeddings_openai.npz"},
]

N_RANDOM_PAIRS = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def intra_keyword_distances(embeddings, keywords, mask):
    distances: list[float] = []
    for kw in np.unique(keywords[mask]):
        idx = np.where(mask & (keywords == kw))[0]
        if len(idx) < 2:
            continue
        mat = normalize_rows(embeddings[idx])
        sim = mat @ mat.T
        tri = np.triu_indices(len(idx), k=1)
        distances.extend((1.0 - sim[tri]).tolist())
    return np.asarray(distances)


def inter_keyword_distances(embeddings, keywords, mask, rng, n_pairs):
    idx = np.where(mask)[0]
    kws = keywords[idx]
    normed = normalize_rows(embeddings[idx])
    distances: list[float] = []
    found = 0
    while found < n_pairs:
        ii = rng.integers(0, len(idx), size=n_pairs * 2)
        jj = rng.integers(0, len(idx), size=n_pairs * 2)
        valid = kws[ii] != kws[jj]
        for a, b in zip(ii[valid], jj[valid]):
            distances.append(1.0 - float(normed[a] @ normed[b]))
            found += 1
            if found >= n_pairs:
                break
    return np.asarray(distances)


def random_pair_distances(embeddings, mask, rng, n_pairs):
    idx = np.where(mask)[0]
    normed = normalize_rows(embeddings[idx])
    i_samples = rng.integers(0, len(idx), size=n_pairs)
    j_samples = rng.integers(0, len(idx), size=n_pairs)
    return np.asarray([1.0 - float(normed[i] @ normed[j]) for i, j in zip(i_samples, j_samples)])


def fmt(arr):
    return f"{arr.mean():.4f}±{arr.std():.4f} (med {np.median(arr):.4f})"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    csv_rows: list[dict] = []

    for cfg in EMBEDDING_FILES:
        npz_path = run_dir / cfg["file"]
        if not npz_path.exists():
            raise FileNotFoundError(
                f"[{cfg['name']}] facet embeddings not found: {npz_path}. "
                f"Run the embedding pipeline for this model before analysis."
            )

        npz = np.load(npz_path, allow_pickle=True)
        vectors = npz["vectors"]
        keywords = np.asarray([str(k) for k in npz["keywords"]])
        efforts = np.asarray([str(e) for e in npz["efforts"]])
        facets = np.asarray([str(f) for f in npz["facets"]])

        effort_levels = sorted(set(efforts), key=lambda e: EFFORT_ORDER.index(e))
        facet_levels = [f for f in FACET_ORDER if f in set(facets)]

        print(f"\n{'=' * 72}")
        print(f"  {cfg['name']}  ({vectors.shape[1]}d)  — {npz_path.name}")
        print(f"{'=' * 72}")

        # Grid: rows=facets, cols=efforts
        fig, axes = plt.subplots(
            len(facet_levels),
            len(effort_levels),
            figsize=(4 * len(effort_levels), 3 * len(facet_levels)),
            sharex="row",
            sharey="row",
        )
        if len(facet_levels) == 1:
            axes = np.array([axes])
        if len(effort_levels) == 1:
            axes = axes.reshape(-1, 1)

        for fi, facet in enumerate(facet_levels):
            facet_mask = facets == facet
            print(f"\n  [facet={facet}]")
            for ei, effort in enumerate(effort_levels):
                mask = facet_mask & (efforts == effort)

                intra = intra_keyword_distances(vectors, keywords, mask)
                inter = inter_keyword_distances(vectors, keywords, mask, rng, N_RANDOM_PAIRS)
                rand = random_pair_distances(vectors, mask, rng, N_RANDOM_PAIRS)
                sep = inter.mean() / intra.mean() if intra.mean() > 0 else float("inf")

                print(f"    {effort:7s} intra={fmt(intra):<35s}  inter={fmt(inter):<35s}  sep={sep:.3f}")

                csv_rows.append(
                    {
                        "embedding": cfg["name"],
                        "facet": facet,
                        "effort": effort,
                        "intra_mean": intra.mean(),
                        "intra_std": intra.std(),
                        "inter_mean": inter.mean(),
                        "inter_std": inter.std(),
                        "random_mean": rand.mean(),
                        "random_std": rand.std(),
                        "separability": sep,
                    }
                )

                ax = axes[fi, ei]
                all_max = max(intra.max(), inter.max(), rand.max())
                bins = np.linspace(0, all_max * 1.05, 80)
                ax.hist(intra, bins=bins, alpha=0.6, label="Intra-kw", color="#2196F3", density=True)
                ax.hist(inter, bins=bins, alpha=0.6, label="Inter-kw", color="#F44336", density=True)
                ax.hist(rand, bins=bins, alpha=0.3, label="Random", color="#9E9E9E", density=True)
                ax.axvline(intra.mean(), color="#1565C0", linestyle="--", linewidth=1)
                ax.axvline(inter.mean(), color="#C62828", linestyle="--", linewidth=1)
                ax.set_title(f"{facet} / {effort}  (sep={sep:.2f})", fontsize=10)
                if fi == len(facet_levels) - 1:
                    ax.set_xlabel("Cosine distance")
                if ei == 0:
                    ax.set_ylabel("Density")
                if fi == 0 and ei == 0:
                    ax.legend(fontsize=8)

        fig.suptitle(f"Per-facet Embedding Sensitivity — {cfg['name']}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig_path = (
            output_dir / f"embedding_sensitivity_facet_{cfg['name'].lower().replace(' ', '_').replace('-', '_')}.png"
        )
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"\n  -> {fig_path}")

    # Summary table
    print(f"\n\n{'=' * 96}")
    print("  SUMMARY (per facet)")
    print(f"{'=' * 96}")
    header = f"{'Embedding':<16} {'Facet':<11} {'Effort':<7} {'Intra':>9} {'Inter':>9} {'Random':>9} {'Sep':>5}"
    print(header)
    print("-" * len(header))
    for r in csv_rows:
        print(
            f"{r['embedding']:<16} {r['facet']:<11} {r['effort']:<7} "
            f"{r['intra_mean']:>9.4f} {r['inter_mean']:>9.4f} "
            f"{r['random_mean']:>9.4f} {r['separability']:>5.2f}"
        )

    # Write CSV
    csv_path = output_dir / "embedding_sensitivity_facet_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n-> {csv_path}")


if __name__ == "__main__":
    main()
