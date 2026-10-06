#!/usr/bin/env python3
"""Plot per-keyword PCA grids from runner-generated per-field embeddings.

Same layout as ``plot_facet_umap.py`` (rows=embedding model, cols=schema
field), but projects each (keyword, field, embedding model) panel with
PCA(n_components=2) instead of UMAP. Useful as a linear sanity check:
if the LLM clusters seen in UMAP survive linear projection, they are
not a UMAP-specific artifact.

Usage:
    uv run python scripts/experiment/embedding/plot_facet_pca.py \
        --input-dirs \
            results/.../phase1_claude_22kw_50x4_facet_seed42 \
            results/.../phase1_gpt54_22kw_50x4_facet_seed42 \
            results/.../phase1_gemini31pro_22kw_50x3_facet_seed42 \
        --embed-subdirs amazontitan-embed-text-v20 text-embedding-3-large allenaispecter2 \
        --embed-labels "Titan v2" "OpenAI" "SPECTER2" \
        --output-dir results/.../pca_3x3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.embedding_loader import load_facet_embeddings  # noqa: E402
from src.analysis.facet_grid import plot_facet_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a per-keyword PCA(2D) grid (rows=embedding model, cols=field).",
    )
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--embed-subdirs", type=Path, nargs="+", required=True)
    parser.add_argument("--embed-labels", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = load_facet_embeddings(
        input_dirs=args.input_dirs,
        embed_subdirs=args.embed_subdirs,
        embed_labels=args.embed_labels,
    )

    def reduce_2d(vectors: np.ndarray) -> np.ndarray:
        return PCA(n_components=2, random_state=42).fit_transform(vectors)

    plot_facet_grid(
        data=data,
        reduce_2d=reduce_2d,
        output_dir=args.output_dir,
        filename_prefix="pca",
    )


if __name__ == "__main__":
    main()
