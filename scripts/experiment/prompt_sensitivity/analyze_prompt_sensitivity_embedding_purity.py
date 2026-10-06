"""Q-B numerical test (#133): are VS samples in the same embedding region as
default-high (substitutability), or in a different region (complementarity)?

For each (cell-A, cell-B) pair and each keyword in the Q1/Q4 subset, compute:

* **1-NN purity** — fraction of samples whose nearest neighbor (excluding
  self) in the union of A∪B (60 vectors) is from the same cell. 0.5 = the
  two cells freely mix (same region); 1.0 = the cells are perfectly
  separable (different regions).
* **Centroid cosine distance** — distance between cell-A's mean vector and
  cell-B's mean vector. Quantifies how far apart the cells' centers sit.
* **Cross / within ratio** — mean cross-pair cosine distance (A↔B) divided
  by the mean of within-cell cosine distance. Ratio ≈ 1: cells overlap;
  ratio > 1: cells are separated by more than their internal spread.

Aggregate per stratum (Q1, Q4): mean ± bootstrap 95% CI across keywords.

Cell pairs:
  ``default-low vs default-high``        — effort-axis reference
  ``default-low vs VS-low``              — prompt at low effort (does VS
                                           shift the region at all?)
  ``default-high vs VS-low``             — **Q-B substitutability test**
  ``default-high vs VS-high``            — prompt at high effort
  ``VS-low vs VS-high``                  — effort within VS
  ``default-low vs SSoT-low``            — SSoT shift at low effort
  ``default-high vs SSoT-high``          — SSoT shift at high effort
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.artifacts import load_sample_records  # noqa: E402
from src.model_registry import EmbeddingModelName  # noqa: E402
from src.text import slugify  # noqa: E402

RESULTS = REPO_ROOT / "results" / "effort_diversity"

# Mirrors `_SUPPORTED_EMBEDDING_MODELS` in scripts/experiment/llm_judge/llm_judge_pairwise.py.
_SUPPORTED_EMBEDDING_MODELS: tuple[EmbeddingModelName, ...] = (
    EmbeddingModelName.AMAZON_TITAN_EMBED_TEXT_V2_0,
    EmbeddingModelName.SPECTER2,
    EmbeddingModelName.SPECTER2_ADHOC_QUERY,
    EmbeddingModelName.TEXT_EMBEDDING_3_LARGE,
)

MODELS: dict[str, tuple[str, str]] = {
    "claude": ("phase2_claude_1180kw_30x4_facet_seed42", "claude"),
    "gpt54": ("phase2_gpt54_1180kw_30x4_facet_seed42", "gpt54"),
    "gemini31pro": ("phase2_gemini31pro_1180kw_30x3_facet_seed42", "gemini31pro"),
}

# (label, (kind_a, effort_a), (kind_b, effort_b))
CELL_PAIRS: list[tuple[str, tuple[str, str], tuple[str, str]]] = [
    ("default-low vs default-high  (effort-axis reference)", ("phase2", "low"), ("phase2", "high")),
    ("default-low vs VS-low        (prompt at low)", ("phase2", "low"), ("vs", "low")),
    ("default-high vs VS-low       (Q-B substitutability)", ("phase2", "high"), ("vs", "low")),
    ("default-high vs VS-high      (prompt at high)", ("phase2", "high"), ("vs", "high")),
    ("VS-low vs VS-high            (effort within VS)", ("vs", "low"), ("vs", "high")),
    ("default-low vs SSoT-low      (SSoT at low)", ("phase2", "low"), ("ssot", "low")),
    ("default-high vs SSoT-high    (SSoT at high)", ("phase2", "high"), ("ssot", "high")),
    ("SSoT-low vs SSoT-high        (effort within SSoT)", ("ssot", "low"), ("ssot", "high")),
    ("default-high vs SSoT-low     (Q-B' default-SSoT substitutability)", ("phase2", "high"), ("ssot", "low")),
    ("VS-low vs SSoT-low           (prompt at low VS-SSoT)", ("vs", "low"), ("ssot", "low")),
    ("VS-high vs SSoT-high         (prompt at high VS-SSoT)", ("vs", "high"), ("ssot", "high")),
    ("VS-high vs SSoT-low          (Q-B VS-SSoT substitutability)", ("vs", "high"), ("ssot", "low")),
]

_SAMPLES_CACHE: dict[Path, list] = {}
_EMB_CACHE: dict[Path, np.ndarray] = {}


def load_cell_vectors(*, run_dir: Path, keyword: str, effort: str, emb_subdir: str) -> np.ndarray:
    samples_path = run_dir / emb_subdir / "samples.jsonl"
    emb_path = run_dir / emb_subdir / "embeddings.npy"
    if not samples_path.exists() or not emb_path.exists():
        return np.empty((0, 0))
    if samples_path not in _SAMPLES_CACHE:
        _SAMPLES_CACHE[samples_path] = load_sample_records(samples_path)
        _EMB_CACHE[emb_path] = np.load(emb_path, mmap_mode="r")
    samples = _SAMPLES_CACHE[samples_path]
    embeddings = _EMB_CACHE[emb_path]
    indices = [i for i, s in enumerate(samples) if s.keyword == keyword and s.effort.value == effort]
    if not indices:
        return np.empty((0, embeddings.shape[1]))
    return np.asarray(embeddings[indices])


def resolve_cell_dir(*, model_cfg: tuple[str, str], kind: str, effort: str) -> Path:
    phase2_subdir, ps_prefix = model_cfg
    if kind == "phase2":
        return RESULTS / phase2_subdir
    return RESULTS / "prompt_sensitivity" / f"{ps_prefix}_{kind}_{effort}"


def cosine_distance_matrix(vecs: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance, shape (n, n). Self-distance set to 0."""
    norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sim = norm @ norm.T
    return 1.0 - sim


def cosine_distance_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    norm_a = a / np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - (norm_a @ norm_b.T)


def compute_per_kw_metrics(vec_a: np.ndarray, vec_b: np.ndarray) -> dict[str, float]:
    """1-NN purity, centroid cosine distance, cross/within ratio."""
    n_a, n_b = len(vec_a), len(vec_b)
    if n_a < 2 or n_b < 2:
        return {"purity": float("nan"), "centroid_dist": float("nan"), "cross_within_ratio": float("nan")}

    # 1-NN purity on union
    union = np.vstack([vec_a, vec_b])
    labels = np.concatenate([np.zeros(n_a, dtype=int), np.ones(n_b, dtype=int)])
    norm_union = union / np.linalg.norm(union, axis=1, keepdims=True)
    sim_union = norm_union @ norm_union.T
    np.fill_diagonal(sim_union, -np.inf)  # exclude self
    nn = np.argmax(sim_union, axis=1)
    purity = float((labels[nn] == labels).mean())

    # Centroid cosine distance
    c_a = vec_a.mean(axis=0)
    c_b = vec_b.mean(axis=0)
    centroid_dist = float(1.0 - (c_a @ c_b) / (np.linalg.norm(c_a) * np.linalg.norm(c_b)))

    # Cross / within ratio (mean pairwise cosine distance)
    iu_a = np.triu_indices(n_a, k=1)
    iu_b = np.triu_indices(n_b, k=1)
    within_a = float(cosine_distance_matrix(vec_a)[iu_a].mean())
    within_b = float(cosine_distance_matrix(vec_b)[iu_b].mean())
    cross = float(cosine_distance_cross(vec_a, vec_b).mean())
    mean_within = 0.5 * (within_a + within_b)
    ratio = cross / mean_within if mean_within > 0 else float("nan")
    return {
        "purity": purity,
        "centroid_dist": centroid_dist,
        "cross_within_ratio": ratio,
    }


def bootstrap_mean_ci(
    values: np.ndarray, *, rng: np.random.Generator, n_boot: int = 10_000
) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude", choices=list(MODELS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--exclude-keyword",
        action="append",
        default=[],
        metavar="KW",
        help=(
            "Drop all rows for KW from the Q1/Q4 set before computing per-pair "
            "purity / centroid Δ / cross-within. Can be repeated. Use for "
            "keywords whose generation yield is below the per-cell target "
            "(e.g., bioterrorism on gpt54 SSoT cells via OpenAI safety filter)."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=EmbeddingModelName,
        default=EmbeddingModelName.TEXT_EMBEDDING_3_LARGE,
        choices=list(_SUPPORTED_EMBEDDING_MODELS),
        metavar=f"{{{','.join(m.value for m in _SUPPORTED_EMBEDDING_MODELS)}}}",
        help=(
            "Embedding model whose samples.jsonl + embeddings.npy to read. "
            f"Default: {EmbeddingModelName.TEXT_EMBEDDING_3_LARGE.value}."
        ),
    )
    args = parser.parse_args()

    model_cfg = MODELS[args.model]
    rng = np.random.default_rng(args.seed)
    excluded_keywords: tuple[str, ...] = tuple(sorted(set(args.exclude_keyword)))
    emb_subdir = slugify(args.embedding_model.value)

    q1q4 = pd.read_csv(RESULTS / "prompt_sensitivity" / f"q1_q4_{args.model}.csv")
    if excluded_keywords:
        n_before = len(q1q4)
        q1q4 = q1q4[~q1q4["keyword"].isin(excluded_keywords)].reset_index(drop=True)
        print(f"Excluded {n_before - len(q1q4)} keywords from Q1/Q4 set ({list(excluded_keywords)}); kept {len(q1q4)}")
    rows: list[dict] = []

    for label, (kind_a, eff_a), (kind_b, eff_b) in CELL_PAIRS:
        per_kw: list[dict] = []
        dir_a = resolve_cell_dir(model_cfg=model_cfg, kind=kind_a, effort=eff_a)
        dir_b = resolve_cell_dir(model_cfg=model_cfg, kind=kind_b, effort=eff_b)
        # Skip pair if either cell is missing (e.g., gemini31pro_ssot_high not yet generated)
        if not (dir_a / emb_subdir / "embeddings.npy").exists() or not (dir_b / emb_subdir / "embeddings.npy").exists():
            rows.append({"label": label, "skipped": "missing cell"})
            continue
        for _, kw_row in q1q4.iterrows():
            kw = kw_row["keyword"]
            stratum = kw_row["stratum"]
            vec_a = load_cell_vectors(run_dir=dir_a, keyword=kw, effort=eff_a, emb_subdir=emb_subdir)
            vec_b = load_cell_vectors(run_dir=dir_b, keyword=kw, effort=eff_b, emb_subdir=emb_subdir)
            metrics = compute_per_kw_metrics(vec_a, vec_b)
            per_kw.append({"keyword": kw, "stratum": stratum, **metrics})

        per_kw_df = pd.DataFrame(per_kw)
        for stratum in ("Q1", "Q4"):
            sub = per_kw_df[per_kw_df["stratum"] == stratum]
            stats = {"label": label, "stratum": stratum, "n": len(sub)}
            for metric in ("purity", "centroid_dist", "cross_within_ratio"):
                m, lo, hi = bootstrap_mean_ci(sub[metric].to_numpy(), rng=rng, n_boot=args.n_boot)
                stats[f"{metric}_mean"] = m
                stats[f"{metric}_ci_low"] = lo
                stats[f"{metric}_ci_high"] = hi
            rows.append(stats)

    df = pd.DataFrame(rows)

    print()
    print(f"=== Q-B kNN purity / centroid Δ / cross-within ratio on {args.model} ===")
    print(f"   Bootstrap 95% CI on per-kw distributions (n_boot={args.n_boot}, seed={args.seed}).")
    print("   purity 0.5 = cells freely mix (same region) | 1.0 = perfectly separable.")
    print("   centroid_dist 0 = same center | 0.5+ = far apart (cosine units).")
    print("   cross_within_ratio 1.0 = cross-pair distance ~ within-cell | >1 = separated.")
    print()

    cur_label = None
    for r in rows:
        if r.get("skipped"):
            print(f"{r['label']}  [skip: {r['skipped']}]")
            cur_label = None
            continue
        if r["label"] != cur_label:
            print(f"\n{r['label']}")
            cur_label = r["label"]
        print(
            f"  {r['stratum']} (n={r['n']:>2}):  "
            f"purity {r['purity_mean']:.2f} [{r['purity_ci_low']:.2f},{r['purity_ci_high']:.2f}] | "
            f"centroid Δ {r['centroid_dist_mean']:.3f} [{r['centroid_dist_ci_low']:.3f},{r['centroid_dist_ci_high']:.3f}] | "
            f"cross/within {r['cross_within_ratio_mean']:.2f} [{r['cross_within_ratio_ci_low']:.2f},{r['cross_within_ratio_ci_high']:.2f}]"
        )

    if args.output_csv is not None:
        out = args.output_csv
    else:
        # Backwards-compatible default for text-embedding-3-large; tag with the
        # embedding slug otherwise so multiple embedding-model runs co-exist.
        if args.embedding_model is EmbeddingModelName.TEXT_EMBEDDING_3_LARGE:
            out_name = f"{args.model}_embedding_purity.csv"
        else:
            out_name = f"{args.model}_embedding_purity__{emb_subdir}.csv"
        out = RESULTS / "prompt_sensitivity" / out_name
    df.to_csv(out, index=False)
    print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
