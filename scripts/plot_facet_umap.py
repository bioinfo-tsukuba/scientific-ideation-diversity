#!/usr/bin/env python3
"""Plot per-keyword UMAP grids from runner-generated per-field embeddings.

Layout: rows = embedding models, cols = schema fields.
Points colored by generation model, shaped by effort level.

Each `--input-dir` is a generation-model output directory containing one
subdirectory per embedding model (e.g. ``amazontitan-embed-text-v20/``).
Each embedding subdirectory must contain ``samples.jsonl`` and
``embeddings_{field}.npy``.

Usage:
    uv run python scripts/plot_facet_umap.py \
        --input-dirs \
            results/.../phase1_claude_22kw_50x4_facet_seed42 \
            results/.../phase1_gpt54_22kw_50x4_facet_seed42 \
            results/.../phase1_gemini31pro_22kw_50x3_facet_seed42 \
        --embed-subdirs amazontitan-embed-text-v20 text-embedding-3-large allenaispecter2 \
        --embed-labels "Titan v2" "OpenAI" "SPECTER2" \
        --output-dir results/.../umap_3x3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import umap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.embedding_loader import load_facet_embeddings  # noqa: E402
from src.analysis.facet_grid import plot_facet_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a per-keyword UMAP grid (rows=embedding model, cols=field).",
    )
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True,
                        help="Generation-model output directories. Each contains embedding-model subdirs.")
    parser.add_argument("--embed-subdirs", type=Path, nargs="+", required=True,
                        help="Embedding-model subdirectory names present under each --input-dir (rows of the grid).")
    parser.add_argument("--embed-labels", nargs="+", required=True,
                        help="Display labels for --embed-subdirs (1-1).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = load_facet_embeddings(
        input_dirs=args.input_dirs,
        embed_subdirs=args.embed_subdirs,
        embed_labels=args.embed_labels,
    )

    def reduce_2d(vectors: np.ndarray) -> np.ndarray:
        actual_neighbors = min(args.umap_n_neighbors, len(vectors) - 1)
        reducer = umap.UMAP(
            n_neighbors=actual_neighbors,
            min_dist=args.umap_min_dist,
            random_state=42,
        )
        return reducer.fit_transform(vectors)

    plot_facet_grid(
        data=data,
        reduce_2d=reduce_2d,
        output_dir=args.output_dir,
        filename_prefix="umap",
    )


if __name__ == "__main__":
    main()
