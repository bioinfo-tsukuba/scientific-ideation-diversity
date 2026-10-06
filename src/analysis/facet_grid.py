"""Per-keyword 3x3 grid plotter used by UMAP and PCA scripts.

Layout: rows = embedding models, cols = schema fields.
Points colored by generation model, shaped by effort level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.analysis.display import IDEA_MODEL_COLORS, IDEA_MODEL_DISPLAY_NAMES
from src.model_registry import EffortName
from src.schemas.facet_embedding import FacetEmbeddingData

EFFORT_MARKERS: dict[EffortName, str] = {
    EffortName.NONE: "o",
    EffortName.LOW: "s",
    EffortName.MEDIUM: "^",
    EffortName.HIGH: "D",
}


def plot_facet_grid(
    *,
    data: FacetEmbeddingData,
    reduce_2d: Callable[[np.ndarray], np.ndarray],
    output_dir: Path,
    filename_prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_index = data.index_by_panel()

    n_rows = len(data.embed_subdirs)
    n_cols = len(data.fields)

    for kw in data.keywords:
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 5.2 * n_rows),
            squeeze=False,
        )
        fig.suptitle(kw, fontsize=18, fontweight="bold", y=1.00)

        for row_idx, (embed_subdir, embed_label) in enumerate(
            zip(data.embed_subdirs, data.embed_labels)
        ):
            for col_idx, field_name in enumerate(data.fields):
                ax = axes[row_idx][col_idx]
                entries = panel_index.get((kw, embed_subdir, field_name), [])

                if row_idx == 0:
                    ax.set_title(field_name, fontsize=13, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(embed_label, fontsize=13, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])

                if len(entries) < 3:
                    ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                            transform=ax.transAxes)
                    continue

                vectors = np.array([e.vector for e in entries])
                efforts_list = [e.effort for e in entries]
                models_list = [e.idea_model for e in entries]

                coords = reduce_2d(vectors)

                for effort in data.efforts:
                    for idea_model in data.idea_models:
                        mask = [
                            i for i, (e, m) in enumerate(zip(efforts_list, models_list))
                            if e == effort and m == idea_model
                        ]
                        if mask:
                            ax.scatter(
                                coords[mask, 0], coords[mask, 1],
                                c=IDEA_MODEL_COLORS[idea_model],
                                marker=EFFORT_MARKERS[effort],
                                s=35, alpha=0.6,
                                edgecolors="white", linewidths=0.3,
                            )

        legend_elements: list[Line2D] = []
        for idea_model in data.idea_models:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=IDEA_MODEL_COLORS[idea_model], markersize=9,
                       label=IDEA_MODEL_DISPLAY_NAMES[idea_model])
            )
        legend_elements.append(Line2D([0], [0], marker="None", color="w", label=""))
        for effort in data.efforts:
            legend_elements.append(
                Line2D([0], [0], marker=EFFORT_MARKERS[effort], color="w",
                       markerfacecolor="#666666", markersize=9, label=effort.value)
            )

        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=len(data.idea_models) + 1 + len(data.efforts),
            fontsize=10,
            bbox_to_anchor=(0.5, -0.01),
        )
        plt.tight_layout(rect=(0, 0.02, 1, 0.98))

        slug = kw.replace(" ", "_").replace("/", "_")
        path = output_dir / f"{filename_prefix}_{slug}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}", flush=True)

    print(f"\nall plots saved to {output_dir}")
