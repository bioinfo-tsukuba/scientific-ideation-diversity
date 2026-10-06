#!/usr/bin/env python3
"""Analyze embedding model sensitivity to keyword differences.

Computes cosine distance distributions for three pair types:
  1. Intra-keyword: pairs from the same keyword (same effort)
  2. Inter-keyword: pairs from different keywords (same effort)
  3. Random: uniformly sampled pairs (same effort)

Outputs a summary table and per-effort histograms for each embedding model.

Usage:
    uv run python scripts/analyze_embedding_sensitivity.py \
        --run-dir results/effort_diversity/phase1_claude_22kw_50x4_facet_seed42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from src.artifacts import load_sample_records
from src.metrics import normalize_rows
from src.model_registry import EffortName

EFFORT_ORDER = [e.value for e in EffortName]

EMBEDDING_CONFIGS: list[dict[str, str]] = [
    {"name": "Titan v2", "subdir": "", "file": "embeddings.npy"},
    {"name": "SPECTER2", "subdir": "_allenaispecter2", "file": "embeddings.npy"},
    {"name": "OpenAI", "subdir": "_text-embedding-3-large", "file": "embeddings.npy"},
]

N_RANDOM_PAIRS = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding sensitivity analysis")
    parser.add_argument("--run-dir", type=Path, required=True, help="Base run directory (Titan embedding dir)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to run-dir.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def cosine_distances_flat(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """Cosine distances between all pairs (a_i, b_j), returned flat."""
    return cdist(matrix_a, matrix_b, metric="cosine").ravel()


def intra_keyword_distances(
    embeddings: np.ndarray,
    keywords: np.ndarray,
    effort_mask: np.ndarray,
) -> np.ndarray:
    """Pairwise cosine distances within same keyword, same effort."""
    distances: list[float] = []
    for kw in np.unique(keywords[effort_mask]):
        kw_mask = effort_mask & (keywords == kw)
        idx = np.where(kw_mask)[0]
        if len(idx) < 2:
            continue
        mat = normalize_rows(embeddings[idx])
        sim = mat @ mat.T
        tri = np.triu_indices(len(idx), k=1)
        distances.extend((1.0 - sim[tri]).tolist())
    return np.asarray(distances)


def inter_keyword_distances(
    embeddings: np.ndarray,
    keywords: np.ndarray,
    effort_mask: np.ndarray,
    rng: np.random.Generator,
    n_pairs: int = N_RANDOM_PAIRS,
) -> np.ndarray:
    """Cosine distances between random pairs from *different* keywords, same effort."""
    idx = np.where(effort_mask)[0]
    kws = keywords[idx]
    distances: list[float] = []
    normed = normalize_rows(embeddings[idx])

    pairs_found = 0
    batch = n_pairs * 2
    while pairs_found < n_pairs:
        i_samples = rng.integers(0, len(idx), size=batch)
        j_samples = rng.integers(0, len(idx), size=batch)
        valid = kws[i_samples] != kws[j_samples]
        for ii, jj in zip(i_samples[valid], j_samples[valid]):
            distances.append(1.0 - float(normed[ii] @ normed[jj]))
            pairs_found += 1
            if pairs_found >= n_pairs:
                break
    return np.asarray(distances)


def random_pair_distances(
    embeddings: np.ndarray,
    effort_mask: np.ndarray,
    rng: np.random.Generator,
    n_pairs: int = N_RANDOM_PAIRS,
) -> np.ndarray:
    """Cosine distances between uniformly random pairs, same effort."""
    idx = np.where(effort_mask)[0]
    normed = normalize_rows(embeddings[idx])
    i_samples = rng.integers(0, len(idx), size=n_pairs)
    j_samples = rng.integers(0, len(idx), size=n_pairs)
    dists = [1.0 - float(normed[i] @ normed[j]) for i, j in zip(i_samples, j_samples)]
    return np.asarray(dists)


def format_stats(arr: np.ndarray) -> str:
    return f"{arr.mean():.4f} ± {arr.std():.4f}  (median {np.median(arr):.4f}, n={len(arr)})"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    records = load_sample_records(run_dir / "samples.jsonl")
    keywords = np.array([r.keyword for r in records])
    efforts = np.array([r.effort.value for r in records])
    effort_levels = sorted(set(efforts), key=lambda e: EFFORT_ORDER.index(e))

    print(f"Loaded {len(records)} records, {len(set(keywords))} keywords, efforts={effort_levels}")

    # Collect all results for summary table
    all_results: list[dict] = []

    for cfg in EMBEDDING_CONFIGS:
        emb_dir = Path(str(run_dir) + cfg["subdir"])
        emb_path = emb_dir / cfg["file"]
        if not emb_path.exists():
            raise FileNotFoundError(
                f"[{cfg['name']}] embeddings not found: {emb_path}. "
                f"Run the embedding pipeline for this model before analysis."
            )

        embeddings = np.load(emb_path)
        print(f"\n{'=' * 70}")
        print(f"  {cfg['name']}  ({embeddings.shape[1]}d)")
        print(f"{'=' * 70}")

        fig, axes = plt.subplots(1, len(effort_levels), figsize=(5 * len(effort_levels), 4), sharey=True)
        if len(effort_levels) == 1:
            axes = [axes]

        for ei, effort in enumerate(effort_levels):
            effort_mask = efforts == effort
            ax = axes[ei]

            intra = intra_keyword_distances(embeddings, keywords, effort_mask)
            inter = inter_keyword_distances(embeddings, keywords, effort_mask, rng)
            rand = random_pair_distances(embeddings, effort_mask, rng)

            separability = inter.mean() / intra.mean() if intra.mean() > 0 else float("inf")

            print(f"\n  [{effort}]")
            print(f"    Intra-keyword : {format_stats(intra)}")
            print(f"    Inter-keyword : {format_stats(inter)}")
            print(f"    Random pairs  : {format_stats(rand)}")
            print(f"    Separability  : {separability:.3f}  (inter/intra)")

            all_results.append(
                {
                    "embedding": cfg["name"],
                    "effort": effort,
                    "intra_mean": intra.mean(),
                    "intra_std": intra.std(),
                    "intra_median": float(np.median(intra)),
                    "inter_mean": inter.mean(),
                    "inter_std": inter.std(),
                    "inter_median": float(np.median(inter)),
                    "random_mean": rand.mean(),
                    "random_std": rand.std(),
                    "separability": separability,
                }
            )

            # Histogram
            bins = np.linspace(0, max(intra.max(), inter.max(), rand.max()) * 1.05, 80)
            ax.hist(intra, bins=bins, alpha=0.6, label="Intra-kw", color="#2196F3", density=True)
            ax.hist(inter, bins=bins, alpha=0.6, label="Inter-kw", color="#F44336", density=True)
            ax.hist(rand, bins=bins, alpha=0.4, label="Random", color="#9E9E9E", density=True, linestyle="--")

            ax.axvline(intra.mean(), color="#1565C0", linestyle="--", linewidth=1.5)
            ax.axvline(inter.mean(), color="#C62828", linestyle="--", linewidth=1.5)

            ax.set_title(f"{effort}  (sep={separability:.2f})")
            ax.set_xlabel("Cosine distance")
            if ei == 0:
                ax.set_ylabel("Density")
            ax.legend(fontsize=8)

        fig.suptitle(f"Embedding Sensitivity: {cfg['name']}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig_path = output_dir / f"embedding_sensitivity_{cfg['name'].lower().replace(' ', '_')}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"\n  -> {fig_path}")

    # Summary table
    print(f"\n\n{'=' * 90}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 90}")
    header = f"{'Embedding':<12} {'Effort':<8} {'Intra':>10} {'Inter':>10} {'Random':>10} {'Sep.':>6}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['embedding']:<12} {r['effort']:<8} "
            f"{r['intra_mean']:>10.4f} {r['inter_mean']:>10.4f} "
            f"{r['random_mean']:>10.4f} {r['separability']:>6.2f}"
        )

    # Write CSV
    import csv

    csv_path = output_dir / "embedding_sensitivity_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n-> {csv_path}")


if __name__ == "__main__":
    main()
