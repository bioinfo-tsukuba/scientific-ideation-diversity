"""Shared loader for per-keyword facet embedding grids.

Reads the artifact layout produced by the effort-diversity runner:

    <input_dir>/<embed_subdir>/samples.jsonl
    <input_dir>/<embed_subdir>/embeddings_{field}.npy

and returns a ``FacetEmbeddingData`` with one entry per (keyword, embed,
field, sample). The generation (LLM) model for each input directory is
inferred from the samples themselves — each directory must contain
records for exactly one ``IdeaModelName``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.artifacts import load_sample_records
from src.model_registry import EFFORT_ORDER, EffortName, IdeaModelName
from src.schemas.facet_embedding import FacetEmbeddingData, FacetEmbeddingEntry
from src.schemas.sample import SampleRecord


def _infer_idea_model(records: list[SampleRecord], *, source: Path) -> IdeaModelName:
    models = {record.idea_model for record in records}
    if len(models) != 1:
        raise ValueError(
            f"Expected exactly one IdeaModelName in {source}, got {sorted(m.value for m in models)}"
        )
    return next(iter(models))


def load_facet_embeddings(
    *,
    input_dirs: list[Path],
    embed_subdirs: list[Path],
    embed_labels: list[str],
    verbose: bool = True,
) -> FacetEmbeddingData:
    if len(embed_subdirs) != len(embed_labels):
        raise ValueError("embed_subdirs and embed_labels must have the same length")

    entries: list[FacetEmbeddingEntry] = []
    all_keywords: set[str] = set()
    all_efforts: set[EffortName] = set()
    all_fields: list[str] = []
    idea_models: list[IdeaModelName] = []

    for input_dir in input_dirs:
        dir_idea_model: IdeaModelName | None = None
        for embed_subdir in embed_subdirs:
            embed_dir = input_dir / embed_subdir
            samples_path = embed_dir / "samples.jsonl"
            if not samples_path.exists():
                if verbose:
                    print(f"WARNING: {samples_path} not found, skipping")
                continue
            records: list[SampleRecord] = load_sample_records(samples_path)
            field_names = list(type(records[0].structured_output).model_fields.keys())
            if not all_fields:
                all_fields = field_names
            if dir_idea_model is None:
                dir_idea_model = _infer_idea_model(records, source=samples_path)

            for field_name in field_names:
                embeddings_path = embed_dir / f"embeddings_{field_name}.npy"
                if not embeddings_path.exists():
                    if verbose:
                        print(f"WARNING: {embeddings_path} not found, skipping")
                    continue
                embeddings = np.load(embeddings_path, mmap_mode="r")
                for i, record in enumerate(records):
                    entries.append(FacetEmbeddingEntry(
                        keyword=record.keyword,
                        embed_subdir=embed_subdir,
                        field=field_name,
                        vector=np.asarray(embeddings[i]),
                        effort=record.effort,
                        idea_model=record.idea_model,
                    ))
                    all_keywords.add(record.keyword)
                    all_efforts.add(record.effort)

            if verbose:
                print(f"loaded {embed_dir} ({len(records)} records, {len(field_names)} fields)")
        if dir_idea_model is None:
            raise ValueError(f"No loadable samples under {input_dir}")
        idea_models.append(dir_idea_model)

    data = FacetEmbeddingData(
        entries=entries,
        keywords=sorted(all_keywords),
        efforts=[e for e in EFFORT_ORDER if e in all_efforts],
        fields=all_fields,
        embed_subdirs=list(embed_subdirs),
        embed_labels=list(embed_labels),
        idea_models=idea_models,
    )

    if verbose:
        print(
            f"keywords={len(data.keywords)} efforts={[e.value for e in data.efforts]} "
            f"fields={data.fields} embed_rows={[str(s) for s in data.embed_subdirs]} "
            f"idea_models={[m.value for m in data.idea_models]}"
        )
    return data
