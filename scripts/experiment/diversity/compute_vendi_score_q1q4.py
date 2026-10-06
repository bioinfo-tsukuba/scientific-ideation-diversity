"""Post-hoc: compute Vendi Score per (model, embedding, prompt, effort, Q1/Q4 keyword).

For each model and embedding, computes per-keyword Vendi Scores across the
joint (prompt x effort) axis on the same Q1/Q4 keyword sets used by the
existing pair-distance analysis. This mirrors the cross-axis structure of
``*_q1q4_pair_distance__*.csv`` but at the raw per-keyword level so that
downstream analysis can apply its own aggregation.

Data sources, per model and embedding subdirectory:
  - default prompt x {low, high} effort: phase-2 run dir, filtered to
    Q1/Q4 keywords selected for that model.
  - VS prompt x {low, high} effort: ``{short_key}_vs_{low,high}/`` under
    results/effort_diversity/prompt_sensitivity/.
  - SSoT prompt x {low, high} effort: ``{short_key}_ssot_{low,high}/``.

Reads only existing artefacts; does not regenerate or re-embed.

Output:
  results/effort_diversity/vendi_score/vendi_score_q1q4_by_keyword.csv
  columns: model, embedding, prompt_style, effort, stratum, category,
           keyword, n, vendi_score

Usage:
    uv run python scripts/experiment/diversity/compute_vendi_score_q1q4.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.metrics import vendi_score  # noqa: E402
from src.model_registry import (  # noqa: E402
    IDEA_MODEL_LABEL,
    IDEA_MODEL_PHASE2_DIR,
    IDEA_MODEL_SHORT_KEY,
    IdeaModelName,
)
from src.schemas.sample import SampleRecord  # noqa: E402

EFFORT_DIR = REPO_ROOT / "results" / "effort_diversity"
PS_DIR = EFFORT_DIR / "prompt_sensitivity"
EMBEDDING_SUBDIRS = (
    "allenaispecter2-adhoc-query",
    "amazontitan-embed-text-v20",
    "text-embedding-3-large",
)
PROMPT_LABELS = ("default", "vs", "ssot")
PROMPT_STYLE_BY_LABEL = {
    "default": "structured_facet",
    "vs": "structured_vs",
    "ssot": "structured_ssot",
}
EFFORTS = ("low", "high")
OUTPUT_DIR = EFFORT_DIR / "vendi_score"
OUTPUT_PATH = OUTPUT_DIR / "vendi_score_q1q4_by_keyword.csv"


def load_samples(samples_path: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with samples_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(SampleRecord.model_validate_json(line))
    return records


def load_q1q4(model: IdeaModelName) -> pd.DataFrame:
    label = IDEA_MODEL_LABEL[model]
    df = pd.read_csv(PS_DIR / "q1_q4_all_models.csv")
    df = df[df["generation_model"] == label].copy()
    df = df[["category", "keyword", "stratum"]].drop_duplicates()
    return df


def compute_vendi_for_keywords(
    samples_path: Path,
    embeddings_path: Path,
    keyword_strata: dict[tuple[str, str], str],
    expected_prompt_style: str,
    expected_effort: str,
) -> list[dict[str, object]]:
    """Compute per-keyword Vendi Score, restricting to keywords present in keyword_strata."""

    if not samples_path.exists() or not embeddings_path.exists():
        return []

    records = load_samples(samples_path)
    embeddings = np.load(embeddings_path)
    if len(records) != embeddings.shape[0]:
        raise ValueError(f"sample count {len(records)} != embedding rows {embeddings.shape[0]} for {samples_path}")

    groups: dict[tuple[str, str], list[int]] = {}
    for i, record in enumerate(records):
        if record.prompt_style != expected_prompt_style:
            continue
        if record.effort != expected_effort:
            continue
        key = (record.category or "", record.keyword)
        if key not in keyword_strata:
            continue
        groups.setdefault(key, []).append(i)

    rows: list[dict[str, object]] = []
    for (category, keyword), indices in groups.items():
        rows.append(
            {
                "category": category,
                "keyword": keyword,
                "stratum": keyword_strata[(category, keyword)],
                "n": len(indices),
                "vendi_score": vendi_score(embeddings[indices]),
            }
        )
    return rows


def collect_default(
    model: IdeaModelName,
    embedding_subdir: str,
    keyword_strata: dict[tuple[str, str], str],
) -> list[dict[str, object]]:
    run_dir = EFFORT_DIR / IDEA_MODEL_PHASE2_DIR[model] / embedding_subdir
    samples_path = run_dir / "samples.jsonl"
    embeddings_path = run_dir / "embeddings.npy"

    out: list[dict[str, object]] = []
    for effort in EFFORTS:
        rows = compute_vendi_for_keywords(
            samples_path,
            embeddings_path,
            keyword_strata=keyword_strata,
            expected_prompt_style="structured_facet",
            expected_effort=effort,
        )
        for row in rows:
            row.update({"prompt_style": "default", "effort": effort})
        out.extend(rows)
    return out


def collect_prompt_sensitivity(
    model: IdeaModelName,
    embedding_subdir: str,
    prompt_label: str,
    keyword_strata: dict[tuple[str, str], str],
) -> list[dict[str, object]]:
    short_key = IDEA_MODEL_SHORT_KEY[model]
    expected_prompt_style = PROMPT_STYLE_BY_LABEL[prompt_label]

    out: list[dict[str, object]] = []
    for effort in EFFORTS:
        run_dir = PS_DIR / f"{short_key}_{prompt_label}_{effort}" / embedding_subdir
        samples_path = run_dir / "samples.jsonl"
        embeddings_path = run_dir / "embeddings.npy"
        rows = compute_vendi_for_keywords(
            samples_path,
            embeddings_path,
            keyword_strata=keyword_strata,
            expected_prompt_style=expected_prompt_style,
            expected_effort=effort,
        )
        for row in rows:
            row.update({"prompt_style": prompt_label, "effort": effort})
        out.extend(rows)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    for model in IDEA_MODEL_PHASE2_DIR:
        q1q4 = load_q1q4(model)
        keyword_strata = {(row["category"], row["keyword"]): row["stratum"] for _, row in q1q4.iterrows()}
        print(f"[{model.value}] {len(keyword_strata)} Q1/Q4 keywords")

        for embedding_subdir in EMBEDDING_SUBDIRS:
            print(f"  embedding: {embedding_subdir}")
            for prompt_label in PROMPT_LABELS:
                if prompt_label == "default":
                    rows = collect_default(model, embedding_subdir, keyword_strata)
                else:
                    rows = collect_prompt_sensitivity(model, embedding_subdir, prompt_label, keyword_strata)
                for row in rows:
                    row["model"] = model.value
                    row["embedding"] = embedding_subdir
                all_rows.extend(rows)
                print(f"    {prompt_label:8s}  {len(rows)} rows")

    fieldnames = [
        "model",
        "embedding",
        "prompt_style",
        "effort",
        "stratum",
        "category",
        "keyword",
        "n",
        "vendi_score",
    ]
    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"\nWrote {len(all_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
