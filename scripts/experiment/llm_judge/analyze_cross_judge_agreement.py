#!/usr/bin/env python3
"""Cross-judge agreement between two LLM judges on the same run dir.

Two modes, selected by ``--mode``:

  * ``pairwise`` — aligns the two judges on shared
    (keyword, effort, sample_i, sample_j, order) pairs and reports
    quadratic-weighted Cohen's kappa (ordinal ABCD), flat agreement %,
    and a 4x4 confusion matrix.  Breakdown tables by stratum and
    effort expose whether agreement is driven by the easy top/bottom
    strata or holds in the middle where judges genuinely disagree.

  * ``quality`` — aligns on
    (keyword, effort, sample_row_index) and reports per-axis Pearson
    r / Spearman ρ / mean absolute difference for the three 1-10
    axes (originality / feasibility / clarity).  Useful for confirming
    that per-keyword Δquality (the §6 correlation input) moves
    together under a different judge, even if absolute scale shifts.

Usage:
    uv run python scripts/experiment/llm_judge/analyze_cross_judge_agreement.py \\
        --run-dir results/effort_diversity/phase2_claude_1180kw_30x4_facet_seed42 \\
        --judge-a gpt-4.1 --judge-b claude-haiku-4-5-20251001 \\
        --mode pairwise

Output layout (under ``run_dir/``):
    cross_judge_<slug_a>_vs_<slug_b>/
        pairwise_agreement_summary.csv
        pairwise_confusion.csv
        pairwise_by_stratum.csv
        quality_agreement.csv
        quality_score_scatter.png
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from src.model_registry import JudgeModelName

_ABCD_ORDER: tuple[str, ...] = ("A", "B", "C", "D")
_ABCD_INDEX: dict[str, int] = {label: i for i, label in enumerate(_ABCD_ORDER)}
_AXES: tuple[str, ...] = ("originality", "feasibility", "clarity")

# Global colorscale upper bound for quality-score heatmaps (linear % scale).
# Fixed so the same color represents the same % across every cell, model,
# and facet.  30 % is a safe ceiling: the modal agreement cell rarely
# exceeds ~25 % even on high-agreement judge pairs.
_HEATMAP_VMAX_PCT: float = 50.0
_HEATMAP_TICKS_PCT = np.array([0, 10, 20, 30, 40, 50])


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "judge"


def _dir_for_pair(run_dir: Path, judge: JudgeModelName) -> Path:
    return run_dir / f"llm_judge_{_slugify(judge.value)}"


def _dir_for_quality(run_dir: Path, judge: JudgeModelName) -> Path:
    return run_dir / f"llm_judge_quality_{_slugify(judge.value)}"


def _load_pairwise(path: Path) -> pd.DataFrame:
    """Load pairwise.jsonl as a DataFrame keyed by the dedup tuple."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            req = payload["request"]
            pair = req["pair"]
            rows.append(
                {
                    "keyword": pair["keyword"],
                    "effort": pair["effort"],
                    "stratum": pair["stratum"],
                    "sample_i": int(pair["sample_i_index"]),
                    "sample_j": int(pair["sample_j_index"]),
                    "order": req["order"],
                    "answer": payload["answer"],
                }
            )
    return pd.DataFrame(rows)


def _load_quality(path: Path) -> pd.DataFrame:
    """Load quality.jsonl keyed by (keyword, effort, sample_row_index)."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            target = payload["request"]["target"]
            scores = payload["scores"]
            rows.append(
                {
                    "keyword": target["keyword"],
                    "effort": target["effort"],
                    "sample_row_index": int(target["sample_row_index"]),
                    "originality": float(scores["originality"]),
                    "feasibility": float(scores["feasibility"]),
                    "clarity": float(scores["clarity"]),
                }
            )
    return pd.DataFrame(rows)


_EFFORT_COLUMN_SUFFIX: dict[str, str] = {
    "none": "none",
    "low": "low",
    "medium": "med",
    "high": "high",
}
_AXIS_TAG: dict[str, str] = {
    "originality": "orig",
    "feasibility": "feas",
    "clarity": "clar",
}


def _emit_joint_subset_per_keyword(
    merged: pd.DataFrame,
    suffix: str,
    judge: JudgeModelName,
    output_dir: Path,
    log: logging.Logger,
) -> None:
    """Re-aggregate one judge's scores on the inner-merged joint subset.

    Emits the same wide per-keyword CSV and long per-effort CSV that
    ``aggregate_llm_judge_quality_per_keyword.py`` writes under each
    judge's own dir, but restricted to samples both judges scored.
    Column layout matches the single-judge aggregates so either CSV
    is a drop-in replacement for the §6 correlation input.
    """
    # suffix is "_a" or "_b" for the two judges in the inner merge.
    cols = {axis: axis + suffix for axis in _AXIS_TAG}
    long = merged[["keyword", "effort"] + list(cols.values())].rename(
        columns={v: _AXIS_TAG[axis] for axis, v in cols.items()}
    )
    per_kw_mean = long.groupby(["keyword", "effort"])[list(_AXIS_TAG.values())].mean().reset_index()
    # Detect which effort tiers actually appear in the joint subset, in the
    # canonical order defined by _EFFORT_COLUMN_SUFFIX. phase2 Claude/GPT
    # cover all 4 (none/low/medium/high); phase2 Gemini covers 3 (no none);
    # prompt-sensitivity cells cover 1. The wide pivot must adapt or every
    # row gets dropna-ed away when a tier is absent.
    observed = set(per_kw_mean["effort"])
    present_efforts = [e for e in _EFFORT_COLUMN_SUFFIX if e in observed]

    wide = per_kw_mean.pivot(index="keyword", columns="effort", values=list(_AXIS_TAG.values()))
    wide.columns = [f"{axis}_{_EFFORT_COLUMN_SUFFIX[effort]}" for axis, effort in wide.columns]
    wide = wide.reset_index()
    needed = [f"{axis}_{_EFFORT_COLUMN_SUFFIX[e]}" for axis in _AXIS_TAG.values() for e in present_efforts]
    before = len(wide)
    wide = wide.dropna(subset=needed)
    dropped = before - len(wide)
    if dropped:
        log.warning(
            "joint-subset per-keyword for %s: dropped %d kw missing an effort bucket (kept %d)",
            judge.value,
            dropped,
            len(wide),
        )
    # delta_hn = high - none is the §6 correlation input. Only the 4-tier
    # phase2 shape has both endpoints; skip on 3-tier / 1-tier runs.
    has_delta_hn = "high" in present_efforts and "none" in present_efforts
    if has_delta_hn:
        for axis in _AXIS_TAG.values():
            wide[f"{axis}_delta_hn"] = wide[f"{axis}_high"] - wide[f"{axis}_none"]
    ordered_cols = (
        ["keyword"]
        + [f"{axis}_{_EFFORT_COLUMN_SUFFIX[e]}" for axis in _AXIS_TAG.values() for e in present_efforts]
        + ([f"{axis}_delta_hn" for axis in _AXIS_TAG.values()] if has_delta_hn else [])
    )
    slug = _slugify(judge.value)
    per_kw_path = output_dir / f"per_keyword_effort_{slug}.csv"
    wide[ordered_cols].to_csv(per_kw_path, index=False)
    log.info("joint-subset per-keyword -> %s (%d kw)", per_kw_path, len(wide))

    # Per-effort means on the joint subset.  Bootstrap is over
    # keywords (the natural unit of independent variation here).
    rng = np.random.default_rng(seed=42)
    rows: list[dict] = []
    for effort in present_efforts:
        sub = per_kw_mean[per_kw_mean["effort"] == effort]
        row: dict = {"effort": effort, "n_keywords": int(len(sub))}
        for axis in _AXIS_TAG.values():
            vals = sub[axis].to_numpy()
            row[f"{axis}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            if len(vals) >= 2:
                boots = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(1000)])
                row[f"{axis}_ci_lo"] = float(np.percentile(boots, 2.5))
                row[f"{axis}_ci_hi"] = float(np.percentile(boots, 97.5))
            else:
                row[f"{axis}_ci_lo"] = float("nan")
                row[f"{axis}_ci_hi"] = float("nan")
        rows.append(row)
    by_effort_path = output_dir / f"by_effort_{slug}.csv"
    pd.DataFrame(rows).to_csv(by_effort_path, index=False)
    log.info("joint-subset by-effort -> %s", by_effort_path)


def _confusion_and_kappa(a: np.ndarray, b: np.ndarray) -> tuple[pd.DataFrame, float, float]:
    """Return 4x4 confusion (judge-A rows × judge-B cols) + kappas.

    ``a`` / ``b`` are ABCD label arrays of equal length.
    Returns (confusion_df, quadratic_kappa, plain_agreement_rate).
    """
    idx_a = np.array([_ABCD_INDEX[label] for label in a])
    idx_b = np.array([_ABCD_INDEX[label] for label in b])
    conf = np.zeros((4, 4), dtype=int)
    for ia, ib in zip(idx_a, idx_b):
        conf[ia, ib] += 1
    conf_df = pd.DataFrame(conf, index=_ABCD_ORDER, columns=_ABCD_ORDER)
    kappa = cohen_kappa_score(idx_a, idx_b, weights="quadratic")
    agree = float((idx_a == idx_b).mean())
    return conf_df, float(kappa), agree


def run_pairwise(
    run_dir: Path, judge_a: JudgeModelName, judge_b: JudgeModelName, output_dir: Path, log: logging.Logger
) -> None:
    df_a = _load_pairwise(_dir_for_pair(run_dir, judge_a) / "pairwise.jsonl")
    df_b = _load_pairwise(_dir_for_pair(run_dir, judge_b) / "pairwise.jsonl")
    log.info("loaded pair judgments: %s=%d %s=%d", judge_a.value, len(df_a), judge_b.value, len(df_b))
    key = ["keyword", "effort", "sample_i", "sample_j", "order"]
    merged = df_a.merge(
        df_b,
        on=key,
        suffixes=("_a", "_b"),
        how="inner",
        validate="one_to_one",
    )
    log.info(
        "aligned pair judgments: %d (judge_a only %d, judge_b only %d)",
        len(merged),
        len(df_a) - len(merged),
        len(df_b) - len(merged),
    )

    conf, kappa, agree = _confusion_and_kappa(merged["answer_a"].to_numpy(), merged["answer_b"].to_numpy())
    summary = pd.DataFrame(
        [
            {
                "n_aligned": int(len(merged)),
                "quadratic_kappa": kappa,
                "exact_agreement_rate": agree,
            }
        ]
    )
    summary.to_csv(output_dir / "pairwise_agreement_summary.csv", index=False)
    conf.to_csv(output_dir / "pairwise_confusion.csv")
    log.info("pairwise overall: kappa=%.3f exact_agreement=%.2f%%", kappa, agree * 100)

    # Per-stratum breakdown using the stratum recorded on judge A
    # (both judges see the same pair catalogue so this label is shared).
    per_stratum_rows: list[dict] = []
    for stratum, sub in merged.groupby("stratum_a"):
        _, k, a = _confusion_and_kappa(sub["answer_a"].to_numpy(), sub["answer_b"].to_numpy())
        per_stratum_rows.append(
            {"stratum": stratum, "n": int(len(sub)), "quadratic_kappa": k, "exact_agreement_rate": a}
        )
    pd.DataFrame(per_stratum_rows).to_csv(output_dir / "pairwise_by_stratum.csv", index=False)


def run_quality(
    run_dir: Path, judge_a: JudgeModelName, judge_b: JudgeModelName, output_dir: Path, log: logging.Logger
) -> None:
    df_a = _load_quality(_dir_for_quality(run_dir, judge_a) / "quality.jsonl")
    df_b = _load_quality(_dir_for_quality(run_dir, judge_b) / "quality.jsonl")
    log.info("loaded quality judgments: %s=%d %s=%d", judge_a.value, len(df_a), judge_b.value, len(df_b))
    key = ["keyword", "effort", "sample_row_index"]
    merged = df_a.merge(df_b, on=key, suffixes=("_a", "_b"), how="inner", validate="one_to_one")
    log.info(
        "aligned quality judgments: %d (judge_a only %d, judge_b only %d)",
        len(merged),
        len(df_a) - len(merged),
        len(df_b) - len(merged),
    )

    # Emit joint-subset per-keyword aggregates for each judge so §6
    # correlation (and any other cross-judge comparison) can be run
    # over the intersection of scored samples, not each judge's full
    # independent subset.  The Haiku side drops ~0.07% of samples to
    # repetition-loop degeneration; GPT-4.1 scored those samples, so
    # comparing each judge's per-keyword mean on its own subset would
    # mix a full-data estimate against a slightly-thinned one, which
    # is not a fair like-for-like cross-judge check.
    _emit_joint_subset_per_keyword(merged, "_a", judge_a, output_dir, log)
    _emit_joint_subset_per_keyword(merged, "_b", judge_b, output_dir, log)

    rows: list[dict] = []
    for axis in _AXES:
        x = merged[f"{axis}_a"].to_numpy()
        y = merged[f"{axis}_b"].to_numpy()
        r = stats.pearsonr(x, y)
        rho = stats.spearmanr(x, y)
        rows.append(
            {
                "axis": axis,
                "n_aligned": int(len(merged)),
                "pearson_r": float(r.statistic),
                "spearman_rho": float(rho.statistic),
                "mean_abs_diff": float(np.mean(np.abs(x - y))),
                "mean_a": float(x.mean()),
                "mean_b": float(y.mean()),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "quality_agreement.csv", index=False)
    for row in rows:
        log.info(
            "%s: r=%.3f rho=%.3f abs_diff=%.2f (mean_a=%.2f mean_b=%.2f)",
            row["axis"],
            row["pearson_r"],
            row["spearman_rho"],
            row["mean_abs_diff"],
            row["mean_a"],
            row["mean_b"],
        )

    # 10x10 percentage heatmap per axis (standard confusion-matrix
    # style).  Quality scores are integers 1-10, so every cell
    # corresponds to an exact (score_a, score_b) pair and is
    # annotated with "XX.XX" so the grid is a fully-populated matrix
    # with no placeholder strings or blank cells.  Percentage is
    # consistent across the three panels despite their slightly
    # different aligned-sample counts; the colourbar label reads
    # "% of judgments" directly.  Colour uses log(1 + pct) so both
    # the dominant peak (~30%) and the long tail (0.01%-level) stay
    # visible — a linear scale leaves ~95% of the grid white because
    # one cell holds ~30% of the mass.  Raw matrices are also
    # dumped as CSV so any downstream diagonal-bandwidth metric
    # (|a-b| <= k) can be recomputed without re-reading the ~280k-row
    # quality.jsonl.
    grid_n = 10
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, axis_name in zip(axes, _AXES):
        x = merged[f"{axis_name}_a"].to_numpy().astype(int)
        y = merged[f"{axis_name}_b"].to_numpy().astype(int)
        mat = np.zeros((grid_n, grid_n), dtype=int)
        for xi, yi in zip(x, y):
            if 1 <= xi <= grid_n and 1 <= yi <= grid_n:
                # rows = judge B (y axis), cols = judge A (x axis);
                # flip y so row 0 is the top (highest score) per
                # standard matrix reading order.
                mat[grid_n - yi, xi - 1] += 1
        pct = mat.astype(float) / mat.sum() * 100
        im = ax.imshow(
            pct,
            cmap="Blues",
            vmin=0,
            vmax=_HEATMAP_VMAX_PCT,
            aspect="equal",
        )
        midpoint = _HEATMAP_VMAX_PCT / 2
        for r in range(grid_n):
            for c in range(grid_n):
                v = float(pct[r, c])
                color = "white" if v > midpoint else "black"
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
        ax.set_xticks(range(grid_n))
        ax.set_xticklabels(range(1, grid_n + 1))
        ax.set_yticks(range(grid_n))
        ax.set_yticklabels(range(grid_n, 0, -1))
        ax.set_xlabel(f"{judge_a.value}")
        ax.set_ylabel(f"{judge_b.value}")
        ax.set_title(f"{axis_name} (n={len(merged):,})")
        # Overlay the y = x diagonal.  In the flipped-y display a
        # score pair (s, s) lands at (row = grid_n - s, col = s - 1),
        # so the diagonal runs from (col=0, row=grid_n-1) to
        # (col=grid_n-1, row=0) in display coordinates.
        ax.plot([-0.5, grid_n - 0.5], [grid_n - 0.5, -0.5], "k--", linewidth=0.7)
        tick_pct = _HEATMAP_TICKS_PCT[_HEATMAP_TICKS_PCT <= _HEATMAP_VMAX_PCT]
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=tick_pct)
        cbar.set_ticklabels([f"{v:g}%" for v in tick_pct])
        cbar.set_label("% of judgments")
        pd.DataFrame(
            mat,
            index=[f"b={v}" for v in range(grid_n, 0, -1)],
            columns=[f"a={v}" for v in range(1, grid_n + 1)],
        ).to_csv(output_dir / f"quality_score_matrix_{axis_name}.csv")
    fig.savefig(output_dir / "quality_score_heatmap.pdf")
    fig.savefig(output_dir / "quality_score_heatmap.png", dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--judge-a",
        type=JudgeModelName,
        required=True,
        choices=list(JudgeModelName),
        metavar=f"{{{','.join(m.value for m in JudgeModelName)}}}",
    )
    parser.add_argument(
        "--judge-b",
        type=JudgeModelName,
        required=True,
        choices=list(JudgeModelName),
        metavar=f"{{{','.join(m.value for m in JudgeModelName)}}}",
    )
    parser.add_argument("--mode", choices=["pairwise", "quality"], required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override output directory. Defaults to "
            "<run-dir>/cross_judge_<judge_a>_vs_<judge_b>/. Useful for "
            "regression tests (write to /tmp and diff against the canonical "
            "location) or to keep multiple judge-pair runs side by side."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("analyze_cross_judge_agreement")

    if args.judge_a == args.judge_b:
        raise SystemExit("--judge-a and --judge-b must differ")

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        out_name = f"cross_judge_{_slugify(args.judge_a.value)}_vs_{_slugify(args.judge_b.value)}"
        output_dir = args.run_dir / out_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "pairwise":
        run_pairwise(args.run_dir, args.judge_a, args.judge_b, output_dir, log)
    else:
        run_quality(args.run_dir, args.judge_a, args.judge_b, output_dir, log)


if __name__ == "__main__":
    main()
