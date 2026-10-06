#!/usr/bin/env python3
"""Compute LLM-separability metrics in the original embedding space.

For each (keyword, embedding model, field) panel, compute:

* kNN purity (default k=10, excluding self): fraction of nearest neighbors
  that share the same generation-model (LLM) label. Reported overall and
  per LLM.
* Silhouette score with LLM as the cluster label (cosine metric).
* Mean within-LLM and between-LLM pairwise distances, and their ratio.

Emits two CSVs plus summary heatmaps. All metrics are computed in the
original high-dimensional embedding space (L2-normalized), so they are
independent of UMAP/PCA projection artifacts.

Usage:
    uv run python scripts/experiment/embedding/compute_knn_purity.py \\
        --input-dirs \\
            results/.../phase1_claude_22kw_50x4_facet_seed42 \\
            results/.../phase1_gpt54_22kw_50x4_facet_seed42 \\
            results/.../phase1_gemini31pro_22kw_50x3_facet_seed42 \\
        --embed-subdirs amazontitan-embed-text-v20 text-embedding-3-large allenaispecter2 \\
        --embed-labels "Titan v2" "OpenAI" "SPECTER2" \\
        --output-dir results/.../knn_purity
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.display import IDEA_MODEL_DISPLAY_NAMES  # noqa: E402
from src.analysis.embedding_loader import load_facet_embeddings  # noqa: E402
from src.model_registry import IdeaModelName  # noqa: E402
from src.schemas.facet_embedding import FacetEmbeddingData  # noqa: E402
from src.schemas.knn_purity import KnnPurityGroupRow, KnnPurityOverallRow  # noqa: E402


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def compute_panel_metrics(
    vectors: np.ndarray,
    labels: list[IdeaModelName],
    *,
    k: int,
    idea_models: list[IdeaModelName],
) -> tuple[dict[IdeaModelName, dict[str, float]], dict[str, float]]:
    """Return (per_model_metrics, overall_metrics) for one panel."""
    n = len(vectors)
    labels_arr = np.array([label.value for label in labels])
    vectors_n = l2_normalize(vectors)

    # kNN purity via cosine (== euclidean on L2-normalized vectors for ranking).
    actual_k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=actual_k + 1, metric="cosine").fit(vectors_n)
    _, indices = nn.kneighbors(vectors_n)
    # Drop self (column 0).
    neighbor_labels = labels_arr[indices[:, 1:]]
    same_label = neighbor_labels == labels_arr[:, None]
    purity_per_point = same_label.mean(axis=1)

    distances = pairwise_distances(vectors_n, metric="cosine")
    per_model: dict[IdeaModelName, dict[str, float]] = {}
    for idea_model in idea_models:
        mask = labels_arr == idea_model.value
        count = int(mask.sum())
        if count == 0:
            continue
        purity_mean = float(purity_per_point[mask].mean())
        if count >= 2:
            within_block = distances[np.ix_(mask, mask)]
            iu = np.triu_indices(count, k=1)
            within_mean = float(within_block[iu].mean())
        else:
            within_mean = float("nan")
        other_mask = ~mask
        if other_mask.any():
            between_mean = float(distances[np.ix_(mask, other_mask)].mean())
        else:
            between_mean = float("nan")
        ratio = within_mean / between_mean if between_mean and not np.isnan(between_mean) else float("nan")
        per_model[idea_model] = {
            "n": count,
            "knn_purity": purity_mean,
            "within_dist": within_mean,
            "between_dist": between_mean,
            "within_between_ratio": ratio,
        }

    overall: dict[str, float] = {
        "n_total": n,
        "knn_purity_overall": float(purity_per_point.mean()),
        "actual_k": actual_k,
    }
    unique_labels = np.unique(labels_arr)
    if len(unique_labels) >= 2 and n > len(unique_labels):
        overall["silhouette"] = float(silhouette_score(vectors_n, labels_arr, metric="cosine"))
    else:
        overall["silhouette"] = float("nan")
    return per_model, overall


def write_csv(
    path: Path,
    rows: list[KnnPurityGroupRow] | list[KnnPurityOverallRow],
) -> None:
    if not rows:
        return
    dumped = [row.model_dump(mode="json") for row in rows]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dumped[0].keys()))
        writer.writeheader()
        writer.writerows(dumped)


def plot_heatmap(
    matrix: np.ndarray,
    *,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    fmt: str = ".2f",
) -> None:
    fig, ax = plt.subplots(figsize=(1.6 * len(col_labels) + 2, 1.1 * len(row_labels) + 2))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=11)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=11)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                text = "n/a"
            else:
                text = f"{value:{fmt}}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value < (vmax or matrix.max()) * 0.6 else "black",
                fontsize=10,
            )
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_overall_matrix(
    overall_rows: list[KnnPurityOverallRow],
    *,
    embed_labels: list[str],
    fields: list[str],
    value_attr: str,
) -> np.ndarray:
    matrix = np.full((len(embed_labels), len(fields)), np.nan)
    accum: dict[tuple[int, int], list[float]] = {}
    for row in overall_rows:
        try:
            i = embed_labels.index(row.embed_label)
            j = fields.index(row.field)
        except ValueError:
            continue
        value = getattr(row, value_attr)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        accum.setdefault((i, j), []).append(float(value))
    for (i, j), values in accum.items():
        matrix[i, j] = float(np.mean(values))
    return matrix


def build_group_matrix(
    group_rows: list[KnnPurityGroupRow],
    *,
    embed_labels: list[str],
    fields: list[str],
    value_attr: str,
    idea_model: IdeaModelName,
) -> np.ndarray:
    matrix = np.full((len(embed_labels), len(fields)), np.nan)
    accum: dict[tuple[int, int], list[float]] = {}
    for row in group_rows:
        if row.idea_model != idea_model:
            continue
        try:
            i = embed_labels.index(row.embed_label)
            j = fields.index(row.field)
        except ValueError:
            continue
        value = getattr(row, value_attr)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        accum.setdefault((i, j), []).append(float(value))
    for (i, j), values in accum.items():
        matrix[i, j] = float(np.mean(values))
    return matrix


def run(
    data: FacetEmbeddingData,
    *,
    k: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    per_group_rows: list[KnnPurityGroupRow] = []
    overall_rows: list[KnnPurityOverallRow] = []
    panel_index = data.index_by_panel()

    for kw in data.keywords:
        for embed_subdir, embed_label in zip(data.embed_subdirs, data.embed_labels):
            for field_name in data.fields:
                entries = panel_index.get((kw, embed_subdir, field_name), [])
                if len(entries) < 3:
                    continue
                vectors = np.array([e.vector for e in entries], dtype=np.float64)
                labels = [e.idea_model for e in entries]
                per_model, overall = compute_panel_metrics(
                    vectors,
                    labels,
                    k=k,
                    idea_models=data.idea_models,
                )
                overall_rows.append(
                    KnnPurityOverallRow(
                        keyword=kw,
                        embed_subdir=embed_subdir,
                        embed_label=embed_label,
                        field=field_name,
                        n_total=int(overall["n_total"]),
                        actual_k=int(overall["actual_k"]),
                        knn_purity_overall=float(overall["knn_purity_overall"]),
                        silhouette=float(overall["silhouette"]),
                    )
                )
                for idea_model, metrics in per_model.items():
                    per_group_rows.append(
                        KnnPurityGroupRow(
                            keyword=kw,
                            embed_subdir=embed_subdir,
                            embed_label=embed_label,
                            field=field_name,
                            idea_model=idea_model,
                            n=int(metrics["n"]),
                            knn_purity=float(metrics["knn_purity"]),
                            within_dist=float(metrics["within_dist"]),
                            between_dist=float(metrics["between_dist"]),
                            within_between_ratio=float(metrics["within_between_ratio"]),
                        )
                    )

    per_group_path = output_dir / "per_group_metrics.csv"
    overall_path = output_dir / "overall_metrics.csv"
    write_csv(per_group_path, per_group_rows)
    write_csv(overall_path, overall_rows)
    print(f"wrote {per_group_path} ({len(per_group_rows)} rows)")
    print(f"wrote {overall_path} ({len(overall_rows)} rows)")

    silhouette_mat = build_overall_matrix(
        overall_rows,
        embed_labels=data.embed_labels,
        fields=data.fields,
        value_attr="silhouette",
    )
    plot_heatmap(
        silhouette_mat,
        row_labels=data.embed_labels,
        col_labels=data.fields,
        title=f"Mean silhouette (LLM label, cosine) — avg over {len(data.keywords)} keywords",
        path=output_dir / "summary_silhouette.png",
        vmin=-0.1,
        vmax=0.6,
    )

    purity_overall_mat = build_overall_matrix(
        overall_rows,
        embed_labels=data.embed_labels,
        fields=data.fields,
        value_attr="knn_purity_overall",
    )
    plot_heatmap(
        purity_overall_mat,
        row_labels=data.embed_labels,
        col_labels=data.fields,
        title=f"Mean kNN purity (k={k}) — avg over {len(data.keywords)} keywords",
        path=output_dir / "summary_knn_purity.png",
        vmin=0.0,
        vmax=1.0,
    )

    n_models = len(data.idea_models)
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(1.6 * len(data.fields) * n_models + 2, 1.1 * len(data.embed_labels) + 2),
        squeeze=False,
    )
    for idx, idea_model in enumerate(data.idea_models):
        ax = axes[0][idx]
        mat = build_group_matrix(
            per_group_rows,
            embed_labels=data.embed_labels,
            fields=data.fields,
            value_attr="knn_purity",
            idea_model=idea_model,
        )
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(data.fields)), data.fields, fontsize=10)
        ax.set_yticks(range(len(data.embed_labels)), data.embed_labels, fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = mat[i, j]
                text = "n/a" if np.isnan(value) else f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", color="white" if value < 0.6 else "black", fontsize=9)
        ax.set_title(IDEA_MODEL_DISPLAY_NAMES[idea_model], fontsize=12, fontweight="bold")
    fig.suptitle(f"kNN purity per LLM (k={k})", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(output_dir / "summary_knn_purity_per_model.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote summary plots to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute LLM-separability metrics in embedding space (kNN purity, silhouette, within/between).",
    )
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--embed-subdirs", type=Path, nargs="+", required=True)
    parser.add_argument("--embed-labels", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10, help="kNN neighborhood size (default: 10).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_facet_embeddings(
        input_dirs=args.input_dirs,
        embed_subdirs=args.embed_subdirs,
        embed_labels=args.embed_labels,
    )
    run(data, k=args.k, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
