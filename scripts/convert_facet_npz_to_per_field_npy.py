#!/usr/bin/env python3
"""Convert legacy facet_embeddings_{embed}.npz caches into the current per-field
`embeddings_{facet}.npy` layout used by the runner (PR #45).

For each embed model, writes:
    {output-dir}/{embed_slug}/samples.jsonl
    {output-dir}/{embed_slug}/embeddings_{facet}.npy

where embed_slug follows scripts/run_diversity_experiment.py slugify semantics:
    titan          -> amazontitan-embed-text-v20
    openai         -> text-embedding-3-large
    specter2-adhoc -> allenaispecter2

Usage:
    uv run python scripts/convert_facet_npz_to_per_field_npy.py \
        --input-dir results/effort_diversity/phase1_claude_22kw_50x4_facet_seed42
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifacts import load_sample_records  # noqa: E402
from src.model_registry import EmbeddingModelName  # noqa: E402
from src.schemas.sample import SampleRecord  # noqa: E402
from src.text import slugify  # noqa: E402

EMBED_NPZ_KEY_TO_MODEL: dict[str, EmbeddingModelName] = {
    "titan": EmbeddingModelName.AMAZON_TITAN_EMBED_TEXT_V2_0,
    "openai": EmbeddingModelName.TEXT_EMBEDDING_3_LARGE,
    "specter2-adhoc": EmbeddingModelName.SPECTER2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing samples.jsonl and facet_embeddings_{embed}.npz")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to write {embed_slug}/ subdirs (defaults to --input-dir)")
    return parser.parse_args()


def convert_one_embed_model(
    *,
    npz_path: Path,
    records: list[SampleRecord],
    out_subdir: Path,
    samples_jsonl_src: Path,
) -> None:
    out_subdir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    vectors = data["vectors"]
    efforts = data["efforts"]
    keywords = data["keywords"]
    facets = data["facets"]

    assert vectors.shape[0] == efforts.shape[0] == keywords.shape[0] == facets.shape[0], (
        f".npz arrays length mismatch in {npz_path}"
    )

    dim = vectors.shape[1]

    samples_by_ek: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        samples_by_ek[(record.effort.value, record.keyword)].append(i)

    npz_by_ekf: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i in range(vectors.shape[0]):
        npz_by_ekf[(str(efforts[i]), str(keywords[i]), str(facets[i]))].append(i)

    unique_facets = sorted(set(str(f) for f in facets.tolist()))

    for facet in unique_facets:
        out = np.empty((len(records), dim), dtype=vectors.dtype)
        filled = np.zeros(len(records), dtype=bool)
        for (e, k), sample_rows in samples_by_ek.items():
            npz_indices = npz_by_ekf.get((e, k, facet), [])
            if len(npz_indices) != len(sample_rows):
                raise RuntimeError(
                    f"sample count mismatch for effort={e!r} kw={k!r} facet={facet!r} "
                    f"in {npz_path}: samples.jsonl has {len(sample_rows)}, .npz has {len(npz_indices)}"
                )
            for rank, sample_row in enumerate(sample_rows):
                out[sample_row] = vectors[npz_indices[rank]]
                filled[sample_row] = True

        if not filled.all():
            missing = int((~filled).sum())
            raise RuntimeError(
                f"facet={facet!r}: {missing} rows of samples.jsonl not covered by {npz_path}"
            )

        out_path = out_subdir / f"embeddings_{facet}.npy"
        np.save(out_path, out)
        print(f"  wrote {out_path} shape={out.shape}")

    shutil.copy2(samples_jsonl_src, out_subdir / "samples.jsonl")
    print(f"  copied samples.jsonl -> {out_subdir / 'samples.jsonl'}")


def main() -> None:
    args = parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir or input_dir

    samples_jsonl = input_dir / "samples.jsonl"
    records = load_sample_records(samples_jsonl)
    print(f"[{input_dir.name}] loaded {len(records)} records")

    for embed_key, embedding_model in EMBED_NPZ_KEY_TO_MODEL.items():
        npz_path = input_dir / f"facet_embeddings_{embed_key}.npz"
        if not npz_path.exists():
            print(f"  [skip] {npz_path.name} not found")
            continue
        slug = slugify(embedding_model.value)
        out_subdir = output_dir / slug
        print(f"[{embed_key}] {npz_path.name} -> {out_subdir}/ (slug={slug})")
        convert_one_embed_model(
            npz_path=npz_path,
            records=records,
            out_subdir=out_subdir,
            samples_jsonl_src=samples_jsonl,
        )


if __name__ == "__main__":
    main()
