#!/usr/bin/env python3
"""Per-keyword breakdown of embedding sensitivity.

Two analyses:
  1. Per-keyword intra-distance bar chart (sorted), showing which keywords
     have tight vs spread clusters in each effort level.
  2. Leave-one-keyword-out (LOO) stability: remove each keyword and recompute
     aggregate intra/inter means. If one keyword dominates, its removal shifts
     the aggregate significantly.

Usage:
    uv run python scripts/analyze_embedding_sensitivity_per_keyword.py \
        --run-dir results/effort_diversity/phase1_claude_22kw_50x4_facet_seed42
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.artifacts import load_sample_records
from src.metrics import normalize_rows
from src.model_registry import EffortName

EFFORT_ORDER = [e.value for e in EffortName]

EMBEDDING_CONFIGS: list[dict[str, str]] = [
    {"name": "Titan v2", "subdir": "", "file": "embeddings.npy"},
    {"name": "SPECTER2", "subdir": "_allenaispecter2", "file": "embeddings.npy"},
    {"name": "OpenAI", "subdir": "_text-embedding-3-large", "file": "embeddings.npy"},
]

N_RANDOM_INTER = 5_000  # per LOO iteration, lighter than full analysis

# Tolerance for the cosine-distance range check.  Normalized float64 vectors
# drift by a few ULPs from their true cosine similarity; anything beyond 1e-9
# is a bug (un-normalized input or NaN embedding).
_DISTANCE_EPS: float = 1e-9


def _cosine_distances_from_normalized(sim_triu: np.ndarray) -> np.ndarray:
    """Return cosine distances from an upper-triangular similarity slice.

    Raises ``RuntimeError`` if any value falls outside
    ``[-_DISTANCE_EPS, 2.0 + _DISTANCE_EPS]`` so that a genuine bug (e.g.
    non-normalized inputs) surfaces instead of being silently clipped.
    """
    raw = 1.0 - sim_triu
    if np.any((raw < -_DISTANCE_EPS) | (raw > 2.0 + _DISTANCE_EPS)):
        raise RuntimeError(
            f"Cosine distance out of expected [0, 2] range: "
            f"min={float(raw.min())!r}, max={float(raw.max())!r}. "
            "Embeddings are likely not row-normalized or contain NaN."
        )
    return np.clip(raw, 0.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def per_keyword_intra_distances(
    embeddings: np.ndarray,
    keywords: np.ndarray,
    effort_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return {keyword: array_of_pairwise_distances} for one effort."""
    result: dict[str, np.ndarray] = {}
    for kw in np.unique(keywords[effort_mask]):
        idx = np.where(effort_mask & (keywords == kw))[0]
        if len(idx) < 2:
            continue
        mat = normalize_rows(embeddings[idx])
        sim = mat @ mat.T
        tri = np.triu_indices(len(idx), k=1)
        result[kw] = _cosine_distances_from_normalized(sim[tri])
    return result


def loo_aggregate(
    embeddings: np.ndarray,
    keywords: np.ndarray,
    effort_mask: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """Leave-one-keyword-out: for each removed keyword, compute aggregate intra & inter means."""
    unique_kws = np.unique(keywords[effort_mask])
    normed_all = normalize_rows(embeddings)

    # Pre-compute per-keyword intra means
    kw_intra: dict[str, float] = {}
    for kw in unique_kws:
        idx = np.where(effort_mask & (keywords == kw))[0]
        mat = normed_all[idx]
        sim = mat @ mat.T
        tri = np.triu_indices(len(idx), k=1)
        kw_intra[kw] = float(_cosine_distances_from_normalized(sim[tri]).mean())

    results: dict[str, dict[str, float]] = {}
    for drop_kw in unique_kws:
        remaining = [k for k in unique_kws if k != drop_kw]
        # Aggregate intra = mean of per-keyword intra means (excluding dropped)
        agg_intra = float(np.mean([kw_intra[k] for k in remaining]))

        # Inter: sample pairs from different remaining keywords
        remaining_mask = effort_mask & np.isin(keywords, remaining)
        idx = np.where(remaining_mask)[0]
        kws_sub = keywords[idx]
        normed_sub = normed_all[idx]
        dists: list[float] = []
        found = 0
        while found < N_RANDOM_INTER:
            ii = rng.integers(0, len(idx), size=N_RANDOM_INTER * 2)
            jj = rng.integers(0, len(idx), size=N_RANDOM_INTER * 2)
            valid = kws_sub[ii] != kws_sub[jj]
            for a, b in zip(ii[valid], jj[valid]):
                dists.append(1.0 - float(normed_sub[a] @ normed_sub[b]))
                found += 1
                if found >= N_RANDOM_INTER:
                    break
        agg_inter = float(np.mean(dists))

        results[drop_kw] = {
            "intra_mean": agg_intra,
            "inter_mean": agg_inter,
            "separability": agg_inter / agg_intra if agg_intra > 0 else float("inf"),
        }
    return results


def plot_per_keyword_bars(
    all_kw_intra: dict[str, dict[str, float]],
    effort_levels: list[str],
    model_name: str,
    output_path: Path,
) -> None:
    """Bar chart: per-keyword mean intra-distance, grouped by effort, sorted by high-effort distance."""
    # Sort keywords by high-effort intra mean (or last effort)
    sort_effort = "high" if "high" in effort_levels else effort_levels[-1]
    sorted_kws = sorted(
        all_kw_intra.keys(),
        key=lambda kw: all_kw_intra[kw].get(sort_effort, 0),
    )

    n_kw = len(sorted_kws)
    n_eff = len(effort_levels)
    fig, ax = plt.subplots(figsize=(max(12, n_kw * 0.5), 5))

    bar_width = 0.8 / n_eff
    colors = {"none": "#90CAF9", "low": "#42A5F5", "medium": "#1E88E5", "high": "#0D47A1"}
    x = np.arange(n_kw)

    for ei, effort in enumerate(effort_levels):
        vals = [all_kw_intra[kw].get(effort, 0) for kw in sorted_kws]
        ax.bar(
            x + ei * bar_width,
            vals,
            bar_width,
            label=effort,
            color=colors.get(effort, "#999"),
            alpha=0.85,
        )

    ax.set_xticks(x + bar_width * (n_eff - 1) / 2)
    ax.set_xticklabels(sorted_kws, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean intra-keyword cosine distance")
    ax.set_title(f"Per-keyword intra-distance by effort — {model_name}")
    ax.legend(title="effort")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_loo_stability(
    loo_results: dict[str, dict[str, dict[str, float]]],
    effort_levels: list[str],
    model_name: str,
    full_intra: dict[str, float],
    full_inter: dict[str, float],
    output_path: Path,
) -> None:
    """Strip plot: LOO aggregate intra/inter/separability per effort, with full baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("intra_mean", "Intra mean (LOO)"),
        ("inter_mean", "Inter mean (LOO)"),
        ("separability", "Separability (LOO)"),
    ]
    full_vals = [full_intra, full_inter, {e: full_inter[e] / full_intra[e] for e in effort_levels}]

    for mi, (metric, title) in enumerate(metrics):
        ax = axes[mi]
        for ei, effort in enumerate(effort_levels):
            vals = [loo_results[effort][kw][metric] for kw in loo_results[effort]]
            kws = list(loo_results[effort].keys())
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
            ax.scatter(
                np.full(len(vals), ei) + jitter,
                vals,
                alpha=0.6,
                s=25,
                zorder=2,
            )
            # Annotate the outliers (most extreme 2)
            sorted_idx = np.argsort(vals)
            for oi in [sorted_idx[0], sorted_idx[-1]]:
                ax.annotate(
                    kws[oi],
                    (ei + jitter[oi], vals[oi]),
                    fontsize=6,
                    alpha=0.7,
                    textcoords="offset points",
                    xytext=(5, 3),
                )
            # Baseline
            ax.hlines(full_vals[mi][effort], ei - 0.3, ei + 0.3, color="red", linewidth=1.5, zorder=3)

        ax.set_xticks(range(len(effort_levels)))
        ax.set_xticklabels(effort_levels)
        ax.set_title(title)
        ax.set_xlabel("Effort")

    fig.suptitle(f"Leave-one-keyword-out stability — {model_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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

    print(f"Loaded {len(records)} records, {len(set(keywords))} keywords")

    csv_rows: list[dict] = []

    for cfg in EMBEDDING_CONFIGS:
        emb_dir = Path(str(run_dir) + cfg["subdir"])
        emb_path = emb_dir / cfg["file"]
        if not emb_path.exists():
            raise FileNotFoundError(
                f"[{cfg['name']}] embeddings not found: {emb_path}. "
                f"Run the embedding pipeline for this model before analysis."
            )

        embeddings = np.load(emb_path)
        model_name = cfg["name"]
        print(f"\n{'=' * 60}")
        print(f"  {model_name}")
        print(f"{'=' * 60}")

        # --- 1. Per-keyword intra distances ---
        all_kw_intra: dict[str, dict[str, float]] = {}  # kw -> {effort: mean_dist}
        full_intra: dict[str, float] = {}
        full_inter: dict[str, float] = {}

        for effort in effort_levels:
            effort_mask = efforts == effort
            kw_dists = per_keyword_intra_distances(embeddings, keywords, effort_mask)

            for kw, dists in kw_dists.items():
                all_kw_intra.setdefault(kw, {})[effort] = float(dists.mean())
                csv_rows.append(
                    {
                        "embedding": model_name,
                        "keyword": kw,
                        "effort": effort,
                        "intra_mean": float(dists.mean()),
                        "intra_std": float(dists.std()),
                        "intra_median": float(np.median(dists)),
                        "n_pairs": len(dists),
                    }
                )

            full_intra[effort] = float(np.mean([v[effort] for v in all_kw_intra.values()]))

        # Print per-keyword table for high effort
        print("\n  Per-keyword intra-distance (sorted, high effort):")
        sort_eff = "high" if "high" in effort_levels else effort_levels[-1]
        for kw in sorted(all_kw_intra, key=lambda k: all_kw_intra[k].get(sort_eff, 0)):
            vals = "  ".join(f"{e}={all_kw_intra[kw].get(e, 0):.4f}" for e in effort_levels)
            print(f"    {kw:<30s} {vals}")

        # Check CV (coefficient of variation) across keywords
        for effort in effort_levels:
            vals = [all_kw_intra[kw][effort] for kw in all_kw_intra if effort in all_kw_intra[kw]]
            cv = np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0
            print(f"\n  CV across keywords [{effort}]: {cv:.3f}  (mean={np.mean(vals):.4f}, std={np.std(vals):.4f})")

        # Bar chart
        bar_path = output_dir / f"sensitivity_per_keyword_{model_name.lower().replace(' ', '_')}.png"
        plot_per_keyword_bars(all_kw_intra, effort_levels, model_name, bar_path)
        print(f"  -> {bar_path}")

        # --- 2. LOO stability ---
        loo_results: dict[str, dict[str, dict[str, float]]] = {}
        for effort in effort_levels:
            effort_mask = efforts == effort

            # Full inter for baseline
            normed = normalize_rows(embeddings)
            idx = np.where(effort_mask)[0]
            kws_e = keywords[idx]
            normed_e = normed[idx]
            inter_dists: list[float] = []
            found = 0
            while found < N_RANDOM_INTER:
                ii = rng.integers(0, len(idx), size=N_RANDOM_INTER * 2)
                jj = rng.integers(0, len(idx), size=N_RANDOM_INTER * 2)
                valid = kws_e[ii] != kws_e[jj]
                for a, b in zip(ii[valid], jj[valid]):
                    inter_dists.append(1.0 - float(normed_e[a] @ normed_e[b]))
                    found += 1
                    if found >= N_RANDOM_INTER:
                        break
            full_inter[effort] = float(np.mean(inter_dists))

            loo_results[effort] = loo_aggregate(embeddings, keywords, effort_mask, rng)

        # LOO summary
        for effort in effort_levels:
            seps = [v["separability"] for v in loo_results[effort].values()]
            intras = [v["intra_mean"] for v in loo_results[effort].values()]
            print(
                f"\n  LOO [{effort}]: sep range [{min(seps):.3f}, {max(seps):.3f}]  "
                f"intra range [{min(intras):.4f}, {max(intras):.4f}]"
            )
            # Most influential keyword
            full_sep = full_inter[effort] / full_intra[effort]
            kw_impact = {kw: abs(v["separability"] - full_sep) for kw, v in loo_results[effort].items()}
            top_kw = max(kw_impact, key=kw_impact.get)
            print(f"    Most influential: {top_kw} (sep shift={kw_impact[top_kw]:.3f})")

        loo_path = output_dir / f"sensitivity_loo_{model_name.lower().replace(' ', '_')}.png"
        plot_loo_stability(loo_results, effort_levels, model_name, full_intra, full_inter, loo_path)
        print(f"  -> {loo_path}")

    # Write CSV
    csv_path = output_dir / "sensitivity_per_keyword.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n-> {csv_path}")


if __name__ == "__main__":
    main()
