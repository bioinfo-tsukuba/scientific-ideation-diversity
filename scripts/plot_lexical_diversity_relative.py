#!/usr/bin/env python3
"""Plot relative improvement in lexical diversity metrics (low effort as baseline)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    run_dir = Path("results/effort_diversity/phase2_gemini31pro_1180kw_30x3_facet_seed42")
    agg_path = run_dir / "lexical_diversity_aggregate.csv"

    df = pd.read_csv(agg_path)

    # Compute relative change: (value - low) / low * 100
    facets = ["purpose", "mechanism", "evaluation"]
    efforts = ["low", "medium", "high"]

    relative_data = []

    for facet in facets:
        subset = df[df["facet"] == facet].set_index("effort")
        low = subset.loc["low"]

        for effort in efforts:
            row = subset.loc[effort]
            rel_distinct_2 = (row["distinct_2_mean"] - low["distinct_2_mean"]) / low["distinct_2_mean"] * 100
            rel_distinct_4 = (row["distinct_4_mean"] - low["distinct_4_mean"]) / low["distinct_4_mean"] * 100
            rel_self_bleu_2 = (row["self_bleu_2_mean"] - low["self_bleu_2_mean"]) / low["self_bleu_2_mean"] * 100
            rel_self_bleu_4 = (row["self_bleu_4_mean"] - low["self_bleu_4_mean"]) / low["self_bleu_4_mean"] * 100

            relative_data.append({
                "facet": facet,
                "effort": effort,
                "distinct_2_rel": rel_distinct_2,
                "distinct_4_rel": rel_distinct_4,
                "self_bleu_2_rel": rel_self_bleu_2,
                "self_bleu_4_rel": rel_self_bleu_4,
            })

    rel_df = pd.DataFrame(relative_data)

    # Plot 1: Relative Distinct-N improvement
    fig, ax = plt.subplots(figsize=(8, 5))

    x_pos = {"low": 0, "medium": 1, "high": 2}
    colors = {"purpose": "#1f77b4", "mechanism": "#ff7f0e", "evaluation": "#2ca02c"}
    markers = {"distinct_2_rel": "o", "distinct_4_rel": "s"}

    for facet in facets:
        subset = rel_df[rel_df["facet"] == facet]
        for metric, marker in markers.items():
            x = [x_pos[e] for e in subset["effort"]]
            y = subset[metric].values
            label = f"{facet} (n={'2' if '2' in metric else '4'})"
            ax.plot(x, y, marker=marker, label=label, color=colors[facet],
                   linestyle="-" if "2" in metric else "--", linewidth=2, markersize=8)

    ax.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_xlabel("Effort", fontsize=12)
    ax.set_ylabel("Distinct-N improvement vs. Low (%)", fontsize=12)
    ax.set_title("Relative Distinct-N improvement (low effort = baseline)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(run_dir / "lexical_diversity_distinct_relative.png", dpi=150)
    plt.close()

    # Plot 2: Relative Self-BLEU change (negative = better)
    fig, ax = plt.subplots(figsize=(8, 5))

    self_bleu_markers = {"self_bleu_2_rel": "o", "self_bleu_4_rel": "s"}

    for facet in facets:
        subset = rel_df[rel_df["facet"] == facet]
        for metric, marker in self_bleu_markers.items():
            x = [x_pos[e] for e in subset["effort"]]
            y = subset[metric].values
            label = f"{facet} (n={'2' if '2' in metric else '4'})"
            ax.plot(x, y, marker=marker, label=label, color=colors[facet],
                   linestyle="-" if "2" in metric else "--", linewidth=2, markersize=8)

    ax.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_xlabel("Effort", fontsize=12)
    ax.set_ylabel("Self-BLEU change vs. Low (%)", fontsize=12)
    ax.set_title("Relative Self-BLEU change (low effort = baseline, negative = more diverse)",
                fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(run_dir / "lexical_diversity_self_bleu_relative.png", dpi=150)
    plt.close()

    print("✓ Generated relative improvement plots:")
    print(f"  - {run_dir}/lexical_diversity_distinct_relative.png")
    print(f"  - {run_dir}/lexical_diversity_self_bleu_relative.png")

    # Print summary statistics
    print("\nRelative improvement summary (low → high):")
    for facet in facets:
        subset = rel_df[(rel_df["facet"] == facet) & (rel_df["effort"] == "high")]
        print(f"\n{facet.capitalize()}:")
        print(f"  Distinct-2: +{subset['distinct_2_rel'].values[0]:.1f}%")
        print(f"  Distinct-4: +{subset['distinct_4_rel'].values[0]:.1f}%")
        print(f"  Self-BLEU-2: {subset['self_bleu_2_rel'].values[0]:.1f}%")
        print(f"  Self-BLEU-4: {subset['self_bleu_4_rel'].values[0]:.1f}%")

if __name__ == "__main__":
    main()
