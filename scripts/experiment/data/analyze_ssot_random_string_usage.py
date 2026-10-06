"""Issue #161: SSoT random_string usage diagnostic across 3 generation models.

The SSoT prompt (Misaki & Akiba 2025, arxiv 2510.21150) instructs the model to
emit a unique random string and then "leverage the generated seed---making sure
to extract maximum randomness from the string by using all of its content---to
generate ONE response that is unique and diverse". Issue #133 / #145 already
established that on our scientific-ideation domain, SSoT-low/high pair
distance lies *below* default-low/high for all 3 generation models — the
intervention does not help, and on Q4 it actively hurts. The open question is
whether this is because (a) the model emits the seed but does not condition
on it (decorative seed) or (b) the seed is conditioned on but does not
translate into idea-space diversity on this domain.

This script answers (a) vs (b) with three diagnostics per (model, effort):

1. **Mechanical compliance** — uniqueness rate, length distribution, char-level
   Shannon entropy of random_strings. Confirms whether the model produces
   well-formed, diverse-looking seeds.
2. **Template-share** — the most common ``prefix-7`` substring share within
   each kw's 30 random_strings. If the model anchors on a fixed template and
   only swaps a few characters, this share is high (≥ 0.5). A genuinely random
   draw would yield << 0.1.
3. **Functional usage (the primary signal)** — for each kw's 30 samples, the
   Spearman correlation between the pairwise random_string distance
   (character-bigram Jaccard) and the pairwise idea-embedding cosine distance,
   plus a within-kw permutation null. If the model conditions on the seed,
   ρ > 0 (more different seed ⇒ more different idea); ρ ≈ 0 means the seed is
   decorative.

Outputs:
  - results/effort_diversity/prompt_sensitivity/ssot_random_string_diagnostic.csv
    (6 rows: model × effort).
  - results/effort_diversity/prompt_sensitivity/ssot_random_string_per_kw.csv
    (600 rows: 6 cells × 100 kw).
  - results/effort_diversity/prompt_sensitivity/ssot_qualitative_examples.md
    (5 kw × 5 (random_string, idea) pairs per cell, for human eyeballing).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

CELLS: list[tuple[str, str, str]] = [
    ("claude", "low", "claude_ssot_low"),
    ("claude", "high", "claude_ssot_high"),
    ("gpt54", "low", "gpt54_ssot_low"),
    ("gpt54", "high", "gpt54_ssot_high"),
    ("gemini31pro", "low", "gemini31pro_ssot_low"),
    ("gemini31pro", "high", "gemini31pro_ssot_high"),
]

# Per-model keywords to drop from analysis (e.g., partial-coverage / safety blocks).
# bioterrorism on gpt54 SSoT-low landed only 10/30 samples, SSoT-high 0/30 — review
# (PR #180) confirmed exclusion is appropriate for this analysis.
PER_MODEL_EXCLUDE: dict[str, frozenset[str]] = {
    "gpt54": frozenset({"bioterrorism"}),
}

PERM_REPS = 200
RNG_SEED = 42


def char_entropy(s: str) -> float:
    """Shannon entropy in bits over the character distribution of s."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def bigram_set(s: str) -> set[str]:
    """Character bigram set; falls back to {s} for length-1 strings."""
    if len(s) < 2:
        return {s} if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def jaccard_distance(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - inter / union if union else 0.0


def pairwise_jaccard(strings: list[str]) -> np.ndarray:
    """Upper-triangle pairwise Jaccard distance vector for char bigrams."""
    bigrams = [bigram_set(s) for s in strings]
    n = len(strings)
    out = np.empty(n * (n - 1) // 2, dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            out[k] = jaccard_distance(bigrams[i], bigrams[j])
            k += 1
    return out


def pairwise_cosine_distance(emb: np.ndarray) -> np.ndarray:
    """Upper-triangle cosine-distance vector. emb is (n, d), L2-normalized inside."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    unit = emb / norms
    sim = unit @ unit.T
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    return 1.0 - sim[iu]


def load_cell(cell_dir: Path) -> tuple[list[dict], np.ndarray]:
    """Read samples.jsonl + embeddings.npy aligned by row order."""
    samples_path = cell_dir / "text-embedding-3-large" / "samples.jsonl"
    embs_path = cell_dir / "text-embedding-3-large" / "embeddings.npy"
    samples: list[dict] = []
    with samples_path.open() as f:
        for line in f:
            samples.append(json.loads(line))
    embs = np.load(embs_path)
    if embs.shape[0] != len(samples):
        raise RuntimeError(f"Row mismatch in {cell_dir}: samples={len(samples)} embeddings={embs.shape[0]}")
    return samples, embs


def per_kw_metrics(
    samples: list[dict],
    embs: np.ndarray,
    rng: np.random.Generator,
    perm_reps: int,
    excluded_keywords: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Per-keyword random_string diagnostics + Spearman ρ + permutation null.

    Returns a DataFrame with one row per keyword.
    """
    by_kw: dict[str, list[int]] = {}
    for i, rec in enumerate(samples):
        kw = rec["keyword"]
        if kw in excluded_keywords:
            continue
        by_kw.setdefault(kw, []).append(i)

    rows = []
    for kw, idxs in by_kw.items():
        if len(idxs) < 2:
            continue
        rs_list = [samples[i].get("structured_output", {}).get("random_string", "") for i in idxs]
        emb = embs[idxs]
        n = len(rs_list)
        unique_frac = len(set(rs_list)) / n
        lens = np.array([len(s) for s in rs_list])
        ents = np.array([char_entropy(s) for s in rs_list])

        # template-share: most common 7-char prefix among 30 strings
        prefix7 = Counter(s[:7] for s in rs_list)
        top_prefix7_share = prefix7.most_common(1)[0][1] / n if prefix7 else 0.0

        # functional usage: Spearman ρ between (rs Jaccard) and (idea cosine)
        rs_d = pairwise_jaccard(rs_list)
        idea_d = pairwise_cosine_distance(emb)
        if len(rs_d) > 1 and rs_d.std() > 0 and idea_d.std() > 0:
            rho, _ = spearmanr(rs_d, idea_d)
        else:
            rho = float("nan")

        # permutation null: shuffle the random_strings within the kw and
        # recompute ρ; report the fraction of perms with |ρ_perm| ≥ |ρ_obs|.
        if not math.isnan(rho):
            null = np.empty(perm_reps)
            for r in range(perm_reps):
                perm = rng.permutation(n)
                rs_perm = [rs_list[k] for k in perm]
                rs_d_perm = pairwise_jaccard(rs_perm)
                rho_p, _ = spearmanr(rs_d_perm, idea_d)
                null[r] = rho_p
            perm_p_two_sided = float(np.mean(np.abs(null) >= abs(rho)))
            null_mean = float(null.mean())
        else:
            perm_p_two_sided = float("nan")
            null_mean = float("nan")

        rows.append(
            {
                "keyword": kw,
                "n_samples": n,
                "rs_unique_frac": unique_frac,
                "rs_mean_len": float(lens.mean()),
                "rs_std_len": float(lens.std()),
                "rs_mean_char_entropy": float(ents.mean()),
                "rs_top_prefix7_share": top_prefix7_share,
                "spearman_rs_idea": float(rho) if not math.isnan(rho) else float("nan"),
                "spearman_perm_p": perm_p_two_sided,
                "spearman_perm_null_mean": null_mean,
            }
        )
    return pd.DataFrame(rows)


def aggregate_cell(per_kw: pd.DataFrame, model: str, effort: str) -> dict:
    """One-row aggregate across kws for the (model, effort) cell."""
    n_kw = len(per_kw)
    return {
        "model": model,
        "effort": effort,
        "n_keywords": n_kw,
        "n_samples_total": int(per_kw["n_samples"].sum()),
        "rs_unique_frac_mean": float(per_kw["rs_unique_frac"].mean()),
        "rs_kw_with_all_unique": int((per_kw["rs_unique_frac"] == 1.0).sum()),
        "rs_mean_len": float(per_kw["rs_mean_len"].mean()),
        "rs_mean_len_std_across_kw": float(per_kw["rs_mean_len"].std()),
        "rs_mean_char_entropy": float(per_kw["rs_mean_char_entropy"].mean()),
        "rs_top_prefix7_share_mean": float(per_kw["rs_top_prefix7_share"].mean()),
        "rs_top_prefix7_share_median": float(per_kw["rs_top_prefix7_share"].median()),
        "rs_top_prefix7_share_q90": float(per_kw["rs_top_prefix7_share"].quantile(0.9)),
        "spearman_rs_idea_mean": float(per_kw["spearman_rs_idea"].mean()),
        "spearman_rs_idea_median": float(per_kw["spearman_rs_idea"].median()),
        "spearman_perm_p_mean": float(per_kw["spearman_perm_p"].mean()),
        "spearman_perm_p_median": float(per_kw["spearman_perm_p"].median()),
        "spearman_kw_significant_p05": int((per_kw["spearman_perm_p"] < 0.05).sum()),
        "spearman_perm_null_mean": float(per_kw["spearman_perm_null_mean"].mean()),
    }


def write_qualitative_dump(
    output_md: Path,
    per_cell_samples: dict[tuple[str, str], list[dict]],
    per_cell_kws: dict[tuple[str, str], list[str]],
    n_kw_per_cell: int = 5,
    n_examples_per_kw: int = 5,
) -> None:
    """Write a markdown table with random_string + first ~120 idea chars.

    Picks the 5 keywords whose top_prefix7_share is highest (= most templated)
    plus randomness — to surface the failure-mode visually.
    """
    lines: list[str] = []
    lines.append("# SSoT random_string qualitative examples (issue #161)\n")
    lines.append(
        "5 keywords per cell, 5 (random_string, idea) pairs per kw. Keywords are "
        "selected as the top-5 most-templated (highest ``rs_top_prefix7_share``) "
        "to surface the fixed-template failure mode visually. Idea text is "
        "truncated to 200 chars.\n"
    )
    for (model, effort), samples in per_cell_samples.items():
        kws = per_cell_kws[(model, effort)]
        lines.append(f"\n## {model} / SSoT-{effort}\n")
        for kw in kws:
            kw_samples = [s for s in samples if s["keyword"] == kw][:n_examples_per_kw]
            lines.append(f"\n### kw=`{kw}`\n")
            lines.append("| sample_index | random_string | idea (200 chars) |")
            lines.append("|---|---|---|")
            for s in kw_samples:
                rs = s.get("structured_output", {}).get("random_string", "")
                idea = (s.get("idea") or "").replace("\n", " ").replace("|", "\\|")
                idea = idea[:200] + ("…" if len(idea) > 200 else "")
                lines.append(f"| {s.get('sample_index')} | `{rs}` | {idea} |")
    output_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results/effort_diversity",
        help="Path to results/effort_diversity (default: repo-relative)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <results-dir>/prompt_sensitivity)",
    )
    parser.add_argument(
        "--perm-reps",
        type=int,
        default=PERM_REPS,
        help="Permutation reps per kw for the Spearman null (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RNG_SEED,
        help="RNG seed for the permutation null (default: %(default)s)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.output_dir or (results_dir / "prompt_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cell_dfs: list[pd.DataFrame] = []
    cell_aggs: list[dict] = []
    per_cell_samples: dict[tuple[str, str], list[dict]] = {}
    per_cell_top_kws: dict[tuple[str, str], list[str]] = {}

    for model, effort, cell_name in CELLS:
        cell_dir = results_dir / "prompt_sensitivity" / cell_name
        print(f"[{cell_name}] reading samples + embeddings…", flush=True)
        samples, embs = load_cell(cell_dir)
        rng = np.random.default_rng(args.seed)
        excluded = PER_MODEL_EXCLUDE.get(model, frozenset())
        per_kw = per_kw_metrics(samples, embs, rng, args.perm_reps, excluded_keywords=excluded)
        per_kw.insert(0, "effort", effort)
        per_kw.insert(0, "model", model)
        per_cell_dfs.append(per_kw)
        cell_aggs.append(aggregate_cell(per_kw, model, effort))
        per_cell_samples[(model, effort)] = samples
        # pick top-5 most-templated kws for the qualitative dump
        top_kws = per_kw.sort_values("rs_top_prefix7_share", ascending=False).head(5)["keyword"].tolist()
        per_cell_top_kws[(model, effort)] = top_kws
        agg = cell_aggs[-1]
        print(
            f"  -> {model}/{effort}: n_kw={agg['n_keywords']}, "
            f"unique_frac={agg['rs_unique_frac_mean']:.3f}, "
            f"top_prefix7_share_mean={agg['rs_top_prefix7_share_mean']:.3f}, "
            f"spearman_mean={agg['spearman_rs_idea_mean']:.3f}, "
            f"perm_p_mean={agg['spearman_perm_p_mean']:.3f}, "
            f"sig_kw={agg['spearman_kw_significant_p05']}/{agg['n_keywords']}"
        )

    diag = pd.DataFrame(cell_aggs)
    diag_path = out_dir / "ssot_random_string_diagnostic.csv"
    diag.to_csv(diag_path, index=False)
    print(f"\nWrote {diag_path} ({len(diag)} rows)")

    per_kw_all = pd.concat(per_cell_dfs, ignore_index=True)
    per_kw_path = out_dir / "ssot_random_string_per_kw.csv"
    per_kw_all.to_csv(per_kw_path, index=False)
    print(f"Wrote {per_kw_path} ({len(per_kw_all)} rows)")

    qual_path = out_dir / "ssot_qualitative_examples.md"
    write_qualitative_dump(qual_path, per_cell_samples, per_cell_top_kws)
    print(f"Wrote {qual_path}")


if __name__ == "__main__":
    main()
