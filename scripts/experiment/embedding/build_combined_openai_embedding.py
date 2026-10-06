#!/usr/bin/env python3
"""Build a combined ``embeddings.npy`` for a run that only has per-facet data.

Some earlier runs (phase1_gpt54_*, phase1_gemini31pro_*) persisted only
``facet_embeddings_*.npz`` and per-facet ``.npy`` files, not the combined
``embeddings.npy`` of the full ``record.idea`` text used by Claude's Phase A.
This script re-embeds ``record.idea`` via OpenAI ``text-embedding-3-large``
so cross-model analyses can use the same combined-text embedding basis.

Output:
    {run_dir}_text-embedding-3-large/embeddings.npy  (N, 3072) float64

Usage:
    uv run python scripts/experiment/embedding/build_combined_openai_embedding.py \\
        --run-dir results/effort_diversity/phase1_gpt54_22kw_50x4_facet_seed42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import openai

from src.artifacts import load_sample_records
from src.providers.retry import call_with_retry
from src.text import normalize_embedding_text

MODEL = "text-embedding-3-large"
BATCH_SIZE = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("build_combined_embedding")

    run_dir: Path = args.run_dir
    out_dir = Path(str(run_dir) + "_" + MODEL)
    out_path = out_dir / "embeddings.npy"

    if out_path.exists():
        log.info("Output already exists, skipping: %s", out_path)
        return

    records = load_sample_records(run_dir / "samples.jsonl")
    texts = [normalize_embedding_text(r.idea) for r in records]
    log.info("Embedding %d texts with %s (batch=%d)", len(texts), MODEL, args.batch_size)

    client = openai.OpenAI()
    vectors: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        response = call_with_retry(lambda batch=batch: client.embeddings.create(model=MODEL, input=batch))
        for offset, item in enumerate(response.data):
            vectors[start + offset] = item.embedding
        log.info("  %d/%d done", start + len(batch), len(texts))

    matrix = np.asarray(vectors, dtype=np.float64)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_path, matrix)
    log.info("Saved %s  shape=%s", out_path, matrix.shape)


if __name__ == "__main__":
    main()
