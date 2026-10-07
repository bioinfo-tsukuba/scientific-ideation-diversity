"""Compute per-(model, facet, effort) within-keyword pair distance.

Loads each (model, embedder) `samples.jsonl` from the embedder directory (one
line per sample, in row order matching that directory's `embeddings.npy` /
`embeddings_<facet>.npy`), groups rows by (keyword, effort), computes the
within-group cosine pair-distance mean, and writes:

- `outputs/tables/cache/pair_distances_<model>_<embedding>_<facet>_<effort>.npy` :
    flat array of within-keyword pair distances (one entry per pair), with a
    `.meta.json` sidecar holding a fingerprint of the inputs (path, size and
    mtime of `samples.jsonl` and the array, plus the row order). A cache file
    is reused only when the fingerprint matches, so re-materialized or
    re-ordered inputs are never silently served stale distances.
- `outputs/tables/per_facet_pair_distance_summary.csv` :
    aggregate (mean over all pooled pairs) per (model, facet, effort) cell.
- `outputs/tables/table2_per_facet_distance_low_high.csv` :
    `low → high` deltas for the new Table 2 (combined + per-facet).
- `outputs/tables/app_per_facet_per_embedding.csv` :
    per-facet x per-embedding cross-check table for the appendix.
"""
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
CACHE_DIR = OUTPUT_DIR / "tables" / "cache"
DATA_DIR = OUTPUT_DIR / "tables"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "claude": REPO_ROOT / "results/effort_diversity/phase2_claude_1180kw_30x4_facet_seed42",
    "gpt54": REPO_ROOT / "results/effort_diversity/phase2_gpt54_1180kw_30x4_facet_seed42",
    "gemini31pro": REPO_ROOT / "results/effort_diversity/phase2_gemini31pro_1180kw_30x3_facet_seed42",
}
MODEL_LABEL = {
    "claude": "Claude Sonnet 4.6",
    "gpt54": "GPT-5.4",
    "gemini31pro": "Gemini 3.1 Pro",
}
FACETS = ("purpose", "mechanism", "evaluation", "combined")
EFFORTS_PER_MODEL = {
    "claude": ("none", "low", "medium", "high"),
    "gpt54": ("none", "low", "medium", "high"),
    "gemini31pro": ("low", "medium", "high"),
}
PRIMARY_EMBEDDING = "text-embedding-3-large"
EMBEDDING_DIRS = {
    "text-embedding-3-large": "text-embedding-3-large",
    "titan-embed-text-v2:0": "amazontitan-embed-text-v20",
    "SPECTER2 adhoc_query": "allenaispecter2-adhoc-query",
}


def facet_emb_path(run_dir: Path, facet: str, embedding_subdir: str) -> Path:
    sub = run_dir / embedding_subdir
    if facet == "combined":
        return sub / "embeddings.npy"
    return sub / f"embeddings_{facet}.npy"


def samples_path(run_dir: Path, embedding_subdir: str) -> Path:
    return run_dir / embedding_subdir / "samples.jsonl"


def load_keyword_effort(path: Path) -> list[tuple[str, str]]:
    rows = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["keyword"], r["effort"]))
    return rows


def input_fingerprint(samples: Path, emb_path: Path, kw_eff: list[tuple[str, str]]) -> str:
    """Hash of everything a cached pair-distance array depends on."""
    h = hashlib.sha256()
    for path in (samples, emb_path):
        st = path.stat()
        h.update(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    h.update(json.dumps(kw_eff).encode())
    return h.hexdigest()


def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim = norm @ norm.T
    np.fill_diagonal(sim, 1.0)
    return 1.0 - sim


def pair_distances_for_cell(emb: np.ndarray, indices: list[int]) -> np.ndarray:
    if len(indices) < 2:
        return np.empty(0, dtype=np.float32)
    sub = emb[indices]
    norm = sub / np.linalg.norm(sub, axis=1, keepdims=True)
    sim = norm @ norm.T
    pairs = []
    n = len(indices)
    for i, j in combinations(range(n), 2):
        pairs.append(1.0 - float(sim[i, j]))
    return np.asarray(pairs, dtype=np.float32)


def compute_pair_distances(model_key: str, embedding_key: str, facet: str,
                           effort: str, emb: np.ndarray,
                           kw_eff: list[tuple[str, str]],
                           fingerprint: str) -> np.ndarray:
    safe_emb = embedding_key.replace(":", "-").replace(" ", "-").replace("/", "-")
    cache = CACHE_DIR / f"pair_distances_{model_key}_{safe_emb}_{facet}_{effort}.npy"
    meta = cache.with_suffix(".meta.json")
    if cache.exists() and meta.exists():
        try:
            cached_fingerprint = json.loads(meta.read_text()).get("fingerprint")
        except (OSError, ValueError):
            cached_fingerprint = None
        if cached_fingerprint == fingerprint:
            return np.load(cache)
    by_keyword: dict[str, list[int]] = {}
    for idx, (kw, eff) in enumerate(kw_eff):
        if eff != effort:
            continue
        by_keyword.setdefault(kw, []).append(idx)
    chunks = []
    for kw, ids in by_keyword.items():
        chunks.append(pair_distances_for_cell(emb, ids))
    if not chunks:
        arr = np.empty(0, dtype=np.float32)
    else:
        arr = np.concatenate(chunks)
    np.save(cache, arr)
    meta.write_text(json.dumps({"fingerprint": fingerprint}) + "\n")
    return arr


def main() -> None:
    summary_rows = []
    for model_key, run_dir in RUNS.items():
        for embedding_key, embedding_subdir in EMBEDDING_DIRS.items():
            samples = samples_path(run_dir, embedding_subdir)
            kw_eff = load_keyword_effort(samples)
            for facet in FACETS:
                emb_path = facet_emb_path(run_dir, facet, embedding_subdir)
                emb = np.load(emb_path)
                assert emb.shape[0] == len(kw_eff), (
                    f"{model_key}/{embedding_key}/{facet}: "
                    f"rows={emb.shape[0]} but samples={len(kw_eff)}"
                )
                fingerprint = input_fingerprint(samples, emb_path, kw_eff)
                for effort in EFFORTS_PER_MODEL[model_key]:
                    arr = compute_pair_distances(
                        model_key, embedding_key, facet, effort, emb, kw_eff, fingerprint
                    )
                    summary_rows.append({
                        "model": MODEL_LABEL[model_key],
                        "model_key": model_key,
                        "embedding": embedding_key,
                        "facet": facet,
                        "effort": effort,
                        "n_pairs": int(arr.size),
                        "mean": float(arr.mean()) if arr.size else float("nan"),
                        "median": float(np.median(arr)) if arr.size else float("nan"),
                    })

    summary_csv = DATA_DIR / "per_facet_pair_distance_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)
    print(f"saved {summary_csv}")

    by_key = {(r["model_key"], r["embedding"], r["facet"], r["effort"]): r
              for r in summary_rows}

    # Table 2: low -> high for ALL three models, primary embedding only.
    table2_rows = []
    for model_key in RUNS:
        for facet in FACETS:
            base = by_key[(model_key, PRIMARY_EMBEDDING, facet, "low")]
            high = by_key[(model_key, PRIMARY_EMBEDDING, facet, "high")]
            d_abs = high["mean"] - base["mean"]
            d_rel = d_abs / max(base["mean"], 1e-9) * 100
            table2_rows.append({
                "model": MODEL_LABEL[model_key],
                "facet": facet,
                "baseline_effort": "low",
                "mean_baseline": base["mean"],
                "mean_high": high["mean"],
                "delta_absolute": d_abs,
                "delta_relative_pct": d_rel,
            })
    table2_csv = DATA_DIR / "table2_per_facet_distance_low_high.csv"
    with table2_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table2_rows[0].keys())
        w.writeheader()
        w.writerows(table2_rows)
    print(f"saved {table2_csv}")

    # App H cross-embedding x per-facet table (all 3 embeddings).
    app_xemb_rows = []
    for model_key in RUNS:
        for embedding_key in EMBEDDING_DIRS:
            for facet in FACETS:
                row = {"model": MODEL_LABEL[model_key],
                       "embedding": embedding_key,
                       "facet": facet}
                for effort in ("low", "medium", "high"):
                    row[f"mean_{effort}"] = by_key[
                        (model_key, embedding_key, facet, effort)
                    ]["mean"]
                base = row["mean_low"]
                row["delta_relative_pct_low_to_high"] = (
                    (row["mean_high"] - base) / max(base, 1e-9) * 100
                )
                app_xemb_rows.append(row)
    app_xemb_csv = DATA_DIR / "app_per_facet_per_embedding.csv"
    with app_xemb_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=app_xemb_rows[0].keys())
        w.writeheader()
        w.writerows(app_xemb_rows)
    print(f"saved {app_xemb_csv}")

    # Appendix `none` control retired -- main analysis uses low/medium/high only.

    print()
    print("=== Table 2 (low -> high) [primary embedding only] ===")
    for r in table2_rows:
        print(f"  {r['model']:<20} {r['facet']:<10} "
              f"{r['mean_baseline']:.4f} -> {r['mean_high']:.4f} "
              f"({r['delta_relative_pct']:+.2f}%)")

    print()
    print("=== App H per-facet x per-embedding (low -> high) ===")
    for r in app_xemb_rows:
        print(f"  {r['model']:<20} {r['embedding']:<22} {r['facet']:<10} "
              f"{r['mean_low']:.3f} {r['mean_medium']:.3f} {r['mean_high']:.3f} "
              f"({r['delta_relative_pct_low_to_high']:+.1f}%)")


if __name__ == "__main__":
    main()
