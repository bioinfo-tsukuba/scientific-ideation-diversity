"""Aggregate kNN purity across 3 models x 3 embedders for #150 validation.

Q-B substitutability [(default, high) vs (VS, low)] was 0.87 on Claude / text-emb-3-l
in PR #148, suggesting the two cells occupy distinct embedding regions. This
script checks whether that separation holds across SPECTER2 + Titan as a
control against single-embedder artifact.

Reads existing CSVs produced by
``scripts/experiment/analyze_prompt_sensitivity_embedding_purity.py``:

    results/effort_diversity/prompt_sensitivity/<model>_embedding_purity[__<embedder>].csv

Writes:

    results/effort_diversity/prompt_sensitivity/cross_embedder_purity_qb.csv

Usage:
    uv run python paper/scripts/aggregate_cross_embedder_purity.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "results" / "effort_diversity" / "prompt_sensitivity"

MODELS = ("claude", "gpt54", "gemini31pro")
EMBEDDERS = (
    ("text-emb-3-large", ""),
    ("titan-v2", "__amazontitan-embed-text-v20"),
    ("specter2", "__allenaispecter2-adhoc-query"),
)


def _purity_prefix(p1: str, e1: str, p2: str | None = None, e2: str | None = None) -> str:
    """Construct the label-prefix used by the purity CSV (old notation)."""
    lhs = f"{p1}-{e1}"
    if p2 is not None:
        return f"{lhs} vs {p2}-{e2}"
    return lhs


CONTRASTS = (
    ("effort @ default", _purity_prefix("default", "low", "default", "high")),
    ("effort @ VS", _purity_prefix("VS", "low", "VS", "high")),
    ("effort @ SSoT", _purity_prefix("SSoT", "low", "SSoT", "high")),
    ("prompt @ low (default-VS)", _purity_prefix("default", "low", "VS", "low")),
    ("prompt @ low (default-SSoT)", _purity_prefix("default", "low", "SSoT", "low")),
    ("prompt @ low (VS-SSoT)", _purity_prefix("VS", "low", "SSoT", "low")),
    ("prompt @ high (default-VS)", _purity_prefix("default", "high", "VS", "high")),
    ("prompt @ high (default-SSoT)", _purity_prefix("default", "high", "SSoT", "high")),
    ("prompt @ high (VS-SSoT)", _purity_prefix("VS", "high", "SSoT", "high")),
    ("Q-B (default-VS)", _purity_prefix("default", "high", "VS", "low")),
    ("Q-B (default-SSoT)", _purity_prefix("default", "high", "SSoT", "low")),
    ("Q-B (VS-SSoT)", _purity_prefix("VS", "high", "SSoT", "low")),
)


def load_purity(model: str, embedder_suffix: str) -> pd.DataFrame:
    path = RESULTS / f"{model}_embedding_purity{embedder_suffix}.csv"
    if not path.exists():
        raise SystemExit(f"missing: {path}")
    df = pd.read_csv(path)
    df["label"] = df["label"].str.strip()
    return df


def select(df: pd.DataFrame, contrast_prefix: str, stratum: str) -> dict[str, float]:
    sub = df[df["label"].str.startswith(contrast_prefix) & (df["stratum"] == stratum)]
    if len(sub) != 1:
        raise SystemExit(f"expected 1 row for {contrast_prefix} / {stratum}, got {len(sub)}")
    row = sub.iloc[0]
    return {
        "n": int(row["n"]),
        "purity_mean": float(row["purity_mean"]),
        "purity_ci_low": float(row["purity_ci_low"]),
        "purity_ci_high": float(row["purity_ci_high"]),
    }


def main() -> None:
    rows: list[dict] = []
    for model in MODELS:
        for emb_label, emb_suffix in EMBEDDERS:
            df = load_purity(model, emb_suffix)
            for contrast_label, contrast_prefix in CONTRASTS:
                for stratum in ("Q1", "Q4"):
                    r = select(df, contrast_prefix, stratum)
                    rows.append(
                        {
                            "model": model,
                            "embedder": emb_label,
                            "contrast": contrast_label,
                            "stratum": stratum,
                            "n": r["n"],
                            "purity_mean": r["purity_mean"],
                            "purity_ci_low": r["purity_ci_low"],
                            "purity_ci_high": r["purity_ci_high"],
                        }
                    )

    out = pd.DataFrame(rows)
    out_path = RESULTS / "cross_embedder_purity_qb.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    for qb_label in ("Q-B (default-VS)", "Q-B (default-SSoT)", "Q-B (VS-SSoT)"):
        qb = out[out["contrast"] == qb_label]
        pivot = qb.pivot_table(
            index=["model", "stratum"],
            columns="embedder",
            values="purity_mean",
        ).round(3)
        pivot = pivot[["text-emb-3-large", "titan-v2", "specter2"]]
        pivot["min"] = pivot.min(axis=1)
        pivot["max"] = pivot.max(axis=1)
        pivot["range"] = (pivot["max"] - pivot["min"]).round(3)
        print()
        print(f"== {qb_label} — purity_mean by embedder ==")
        print(pivot.to_string())

    print()
    print("== summary across all 3 model x 2 stratum, by contrast ==")
    summary = out.groupby(["contrast", "embedder"])["purity_mean"].agg(["min", "median", "max"]).round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
