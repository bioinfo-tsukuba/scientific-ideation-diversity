"""Path A/B/C decision for the 5-judge ensemble (W3 rebuttal).

Computes the two pre-specified criteria from
``rebuttal/strong_judge_design.md``:

1.  Trend criterion: for each (generator, judge, axis) combination,
    |mean(default-high) - mean(default-low)| < 0.5 on the 1-10 scale.
    9 (gen × axis) combinations × 5 judges = 45 trend deltas; the
    criterion must hold for all 3 strong judges.
2.  Calibration criterion: for each (strong, weak) judge pair × (gen,
    axis, effort) combination, per-idea Pearson r.  Mean r across
    (gen × axis) combinations ≥ 0.30, minimum ≥ 0.21, evaluated
    separately for default-low and default-high.

Also computes the §3.6 VS-low flatness check:
   |mean(VS-low) - mean(default-low)| < 0.5 across all 5 judges.

Outputs (all under ``results/effort_diversity/rebuttal_strong_judge_path_decision/``):

* ``condition_mean.csv``      — per (gen, judge, axis, condition) mean / sd / n
* ``trend_delta.csv``         — Δ_high-low matrix + criterion-1 pass/fail
* ``vs_low_delta.csv``        — Δ_VS-default-low matrix + §3.6 flatness check
* ``calibration_r.csv``       — per-idea r between strong-weak pairs
* ``path_decision.csv``       — criterion-by-criterion roll-up + final path
* ``per_idea_long.csv.gz``    — flat long-form per-idea scores (5 judges
  × 9 (gen × condition) combinations × 3 axes joined on dedup key,
  gzipped, ~1.8 MB); **gitignored** because it is fully regenerable
  from the upstream ``quality.jsonl`` files which are themselves
  pinned via S3 manifest.  Downstream scripts
  (``plot_strong_judge_path_decision.py``) consume this file directly,
  so re-run analyze first if it is missing.

Run: uv run python scripts/experiment/rebuttal/analyze_strong_judge_path_decision.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "results/effort_diversity/rebuttal_strong_judge_path_decision"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENERATORS = ["claude", "gpt54", "gemini"]
AXES = ["originality", "feasibility", "clarity"]

PHASE2_DIR = {
    "claude": REPO / "results/effort_diversity/phase2_claude_1180kw_30x4_facet_seed42",
    "gpt54":  REPO / "results/effort_diversity/phase2_gpt54_1180kw_30x4_facet_seed42",
    "gemini": REPO / "results/effort_diversity/phase2_gemini31pro_1180kw_30x3_facet_seed42",
}
VS_LOW_DIR = {
    "claude": REPO / "results/effort_diversity/prompt_sensitivity/claude_vs_low",
    "gpt54":  REPO / "results/effort_diversity/prompt_sensitivity/gpt54_vs_low",
    "gemini": REPO / "results/effort_diversity/prompt_sensitivity/gemini31pro_vs_low",
}

WEAK_SLUGS = {
    "gpt_4_1":   "llm_judge_quality_gpt_4_1",
    "haiku_4_5": "llm_judge_quality_claude_haiku_4_5_20251001",
}
# Strong judges on phase2 (full sweep dirs) live in two sibling dirs split by
# generation effort (gen_low / gen_high).  On VS-low dirs, they live in a
# single dir (no gen_* suffix).
STRONG_SLUGS_PHASE2 = {
    "gpt_5_4_medium":  "llm_judge_quality_gpt_5_4__medium__{cond}",
    "sonnet_4_6_high": "llm_judge_quality_jp_anthropic_claude_sonnet_4_6__high__{cond}",
    "gemini_pro_high": "llm_judge_quality_gemini_3_1_pro_preview__high__{cond}",
}
STRONG_SLUGS_VS_LOW = {
    "gpt_5_4_medium":  "llm_judge_quality_gpt_5_4__medium",
    "sonnet_4_6_high": "llm_judge_quality_jp_anthropic_claude_sonnet_4_6__high",
    "gemini_pro_high": "llm_judge_quality_gemini_3_1_pro_preview__high",
}

WEAK_NAMES = list(WEAK_SLUGS.keys())
STRONG_NAMES = list(STRONG_SLUGS_PHASE2.keys())

KW_FILTER = {
    "claude":  REPO / "results/effort_diversity/prompt_sensitivity/judge_keyword_filters/q1q4_claude.csv",
    "gpt54":   REPO / "results/effort_diversity/prompt_sensitivity/judge_keyword_filters/q1q4_gpt54.csv",
    "gemini":  REPO / "results/effort_diversity/prompt_sensitivity/judge_keyword_filters/q1q4_gemini31pro.csv",
}

TREND_TOL = 0.5
CALIB_MEAN_TARGET = 0.30
CALIB_MIN_THRESHOLD = 0.21


def _idea_hash(text: str) -> str:
    """Stable short hash of an idea text -- used as a per-idea join key
    across judge runs (``sample_row_index`` is re-indexed per filter so it
    cannot be used; ``idea_text`` itself is the real identifier)."""
    import hashlib
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def load_quality_jsonl(path: Path) -> pd.DataFrame:
    """Return a DataFrame with columns
    ``[keyword, effort, idea_hash, originality, feasibility, clarity]``.
    """
    cols = ["keyword", "effort", "idea_hash", *AXES]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    rows = []
    with path.open() as fh:
        for line in fh:
            d = json.loads(line)
            t = d["request"]["target"]
            s = d["scores"]
            idea_text = d["request"].get("idea_text", "")
            rows.append({
                "keyword": t["keyword"],
                "effort": t["effort"],
                "idea_hash": _idea_hash(idea_text),
                "originality": s.get("originality"),
                "feasibility": s.get("feasibility"),
                "clarity": s.get("clarity"),
            })
    return pd.DataFrame(rows)


def load_q1q4_keywords(gen: str) -> set[str]:
    df = pd.read_csv(KW_FILTER[gen])
    return set(df["keyword"].astype(str).tolist())



def build_per_idea_long() -> pd.DataFrame:
    """Build a long-form DataFrame of every (judge × gen × condition × idea
    × axis) score.

    Columns: gen, judge, judge_tier {weak,strong}, condition {default-low,
    default-high, vs-low}, keyword, sample_row_index, axis, score.

    Weak judges are loaded from their full-sweep dirs and then filtered to
    the per-generator Q1∪Q4 keyword set + the relevant effort.
    """
    pieces: list[pd.DataFrame] = []

    for gen in GENERATORS:
        kw_keep = load_q1q4_keywords(gen)
        # ---------- WEAK judges (full sweep, then filter to subset) ----------
        for j, slug in WEAK_SLUGS.items():
            path = PHASE2_DIR[gen] / slug / "quality.jsonl"
            df = load_quality_jsonl(path)
            df = df[df["keyword"].isin(kw_keep)]
            # default-low + default-high
            for cond, eff in [("default-low", "low"), ("default-high", "high")]:
                sub = df[df["effort"] == eff].copy()
                sub["gen"] = gen
                sub["judge"] = j
                sub["judge_tier"] = "weak"
                sub["condition"] = cond
                pieces.append(sub)
            # VS-low: load separate dir
            vs_path = VS_LOW_DIR[gen] / slug / "quality.jsonl"
            vs = load_quality_jsonl(vs_path)
            vs = vs[vs["keyword"].isin(kw_keep)]
            vs["gen"] = gen
            vs["judge"] = j
            vs["judge_tier"] = "weak"
            vs["condition"] = "vs-low"
            pieces.append(vs)

        # ---------- STRONG judges (Plan L+ scoped runs) ----------
        for j in STRONG_NAMES:
            # phase2 sweep: 2 sibling dirs split by gen_low / gen_high
            for cond, gencond in [("default-low", "gen_low"), ("default-high", "gen_high")]:
                slug = STRONG_SLUGS_PHASE2[j].format(cond=gencond)
                path = PHASE2_DIR[gen] / slug / "quality.jsonl"
                df = load_quality_jsonl(path)
                df["gen"] = gen
                df["judge"] = j
                df["judge_tier"] = "strong"
                df["condition"] = cond
                pieces.append(df)
            # VS-low
            slug = STRONG_SLUGS_VS_LOW[j]
            path = VS_LOW_DIR[gen] / slug / "quality.jsonl"
            df = load_quality_jsonl(path)
            df["gen"] = gen
            df["judge"] = j
            df["judge_tier"] = "strong"
            df["condition"] = "vs-low"
            pieces.append(df)

    return pd.concat(pieces, ignore_index=True)



def condition_means(long: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / n per (gen, judge, condition, axis)."""
    rows = []
    for (gen, judge, cond), grp in long.groupby(["gen", "judge", "condition"]):
        for ax in AXES:
            ser = grp[ax].dropna()
            rows.append({
                "gen": gen, "judge": judge, "condition": cond, "axis": ax,
                "n": int(len(ser)),
                "mean": float(ser.mean()) if len(ser) else np.nan,
                "std": float(ser.std(ddof=1)) if len(ser) > 1 else np.nan,
            })
    return pd.DataFrame(rows)



def trend_delta(means: pd.DataFrame) -> pd.DataFrame:
    """Δ_high-low per (gen, judge, axis) + criterion-1 pass."""
    rows = []
    for (gen, judge, ax), grp in means.groupby(["gen", "judge", "axis"]):
        gh = grp[grp["condition"] == "default-high"]["mean"]
        gl = grp[grp["condition"] == "default-low"]["mean"]
        if len(gh) and len(gl):
            delta = float(gh.iloc[0]) - float(gl.iloc[0])
            rows.append({
                "gen": gen, "judge": judge, "axis": ax,
                "mean_default_low": float(gl.iloc[0]),
                "mean_default_high": float(gh.iloc[0]),
                "delta_high_low": delta,
                "abs_delta": abs(delta),
                "passes_trend_lt_0_5": abs(delta) < TREND_TOL,
            })
    return pd.DataFrame(rows)



def vs_low_delta(means: pd.DataFrame) -> pd.DataFrame:
    """Δ_VS-low - default-low per (gen, judge, axis)."""
    rows = []
    for (gen, judge, ax), grp in means.groupby(["gen", "judge", "axis"]):
        vs = grp[grp["condition"] == "vs-low"]["mean"]
        dl = grp[grp["condition"] == "default-low"]["mean"]
        if len(vs) and len(dl):
            delta = float(vs.iloc[0]) - float(dl.iloc[0])
            rows.append({
                "gen": gen, "judge": judge, "axis": ax,
                "mean_default_low": float(dl.iloc[0]),
                "mean_vs_low": float(vs.iloc[0]),
                "delta_vs_minus_default": delta,
                "abs_delta": abs(delta),
                "passes_flatness_lt_0_5": abs(delta) < TREND_TOL,
            })
    return pd.DataFrame(rows)



def calibration_r(long: pd.DataFrame) -> pd.DataFrame:
    """Pearson r per (gen, axis, condition, strong, weak)."""
    rows = []
    join_keys = ["gen", "condition", "keyword", "effort", "idea_hash"]
    for gen in GENERATORS:
        for cond in ["default-low", "default-high"]:
            for ax in AXES:
                base = long[(long["gen"] == gen) & (long["condition"] == cond)]
                for strong in STRONG_NAMES:
                    s_df = base[base["judge"] == strong][join_keys + [ax]].rename(columns={ax: "score_strong"})
                    for weak in WEAK_NAMES:
                        w_df = base[base["judge"] == weak][join_keys + [ax]].rename(columns={ax: "score_weak"})
                        merged = s_df.merge(w_df, on=join_keys, how="inner").dropna()
                        if len(merged) < 30:
                            continue
                        r = float(np.corrcoef(merged["score_strong"], merged["score_weak"])[0, 1])
                        rows.append({
                            "gen": gen, "condition": cond, "axis": ax,
                            "strong": strong, "weak": weak,
                            "n": int(len(merged)),
                            "r": r,
                            "passes_min_0_21": r >= CALIB_MIN_THRESHOLD,
                        })
    return pd.DataFrame(rows)



@dataclass
class PathDecision:
    path: str
    trend_pass: bool
    trend_violations: list[str]
    calib_mean_per_pair: dict[str, float]
    calib_min_per_pair: dict[str, float]
    calib_pass: bool
    calib_min_violations: list[str]
    notes: str


def decide_path(trend: pd.DataFrame, calib: pd.DataFrame) -> PathDecision:
    # --- Criterion 1: trend — every strong judge across every (gen × axis)
    strong_trend = trend[trend["judge"].isin(STRONG_NAMES)]
    failed = strong_trend[~strong_trend["passes_trend_lt_0_5"]]
    trend_violations = [
        f"{row.judge} × {row.gen} × {row.axis} (Δ={row.delta_high_low:+.2f})"
        for row in failed.itertuples()
    ]
    trend_pass = len(failed) == 0

    # --- Criterion 2: calibration — per (strong, weak) pair, mean r across (gen × axis) ≥ 0.30, min ≥ 0.21
    calib_mean_per_pair: dict[str, float] = {}
    calib_min_per_pair: dict[str, float] = {}
    calib_min_viol: list[str] = []
    for (strong, weak), grp in calib.groupby(["strong", "weak"]):
        rs = grp["r"].dropna()
        m = float(rs.mean()) if len(rs) else float("nan")
        lo = float(rs.min()) if len(rs) else float("nan")
        calib_mean_per_pair[f"{strong}__{weak}"] = m
        calib_min_per_pair[f"{strong}__{weak}"] = lo
        if lo < CALIB_MIN_THRESHOLD:
            below = grp[grp["r"] < CALIB_MIN_THRESHOLD]
            for row in below.itertuples():
                calib_min_viol.append(
                    f"{strong}×{weak} × {row.gen} × {row.condition} × {row.axis} (r={row.r:.2f})"
                )

    # Calibration pass: every pair's mean r ≥ 0.30 AND no r below 0.21
    calib_pass = (
        all(m >= CALIB_MEAN_TARGET for m in calib_mean_per_pair.values())
        and len(calib_min_viol) == 0
    )

    # --- Path
    if not trend_pass:
        path = "C"
        notes = (
            "Trend criterion violated for at least one (strong judge × gen × axis). "
            "Original null framing in §3 needs to be limited: strong judges detect a quality shift "
            "between default-low and default-high that weak judges missed."
        )
    elif trend_pass and calib_pass:
        path = "A"
        notes = (
            "Fully agree. 5-judge ensemble decisively refutes W3: all 3 strong judges reproduce the "
            "null trend (|Δ_high-low| < 0.5pt) AND per-idea agreement with weak judges is at least "
            "as strong as the weak-weak baseline."
        )
    else:
        path = "B"
        notes = (
            "Trend agrees, calibration weak. Strong judges reproduce the null trend, but per-idea "
            "calibration with weak judges falls below the pre-specified threshold (mean r ≥ 0.30, "
            "min r ≥ 0.21). Trend-only claim is supported; the weakness is disclosed as a limitation."
        )

    return PathDecision(
        path=path,
        trend_pass=trend_pass,
        trend_violations=trend_violations,
        calib_mean_per_pair=calib_mean_per_pair,
        calib_min_per_pair=calib_min_per_pair,
        calib_pass=calib_pass,
        calib_min_violations=calib_min_viol,
        notes=notes,
    )



def main() -> None:
    print("Loading per-idea long table…")
    long = build_per_idea_long()
    print(f"  total rows: {len(long):,}")
    # Persist long parquet for downstream notebook exploration
    long.to_csv(OUT_DIR / "per_idea_long.csv.gz", index=False, compression="gzip")

    print("Computing per-condition means…")
    means = condition_means(long)
    means.to_csv(OUT_DIR / "condition_mean.csv", index=False)
    print(f"  condition_mean.csv written: {len(means)} rows")

    print("Computing trend Δ_high-low…")
    trend = trend_delta(means)
    trend.to_csv(OUT_DIR / "trend_delta.csv", index=False)
    print(f"  trend_delta.csv written: {len(trend)} rows")

    print("Computing VS-low Δ vs default-low…")
    vs = vs_low_delta(means)
    vs.to_csv(OUT_DIR / "vs_low_delta.csv", index=False)
    print(f"  vs_low_delta.csv written: {len(vs)} rows")

    print("Computing per-idea calibration r…")
    calib = calibration_r(long)
    calib.to_csv(OUT_DIR / "calibration_r.csv", index=False)
    print(f"  calibration_r.csv written: {len(calib)} rows")

    print("Applying pre-specified criteria…")
    decision = decide_path(trend, calib)
    summary_rows = [
        {"item": "path", "value": decision.path},
        {"item": "trend_criterion_pass", "value": decision.trend_pass},
        {"item": "trend_violation_count", "value": len(decision.trend_violations)},
        {"item": "calibration_criterion_pass", "value": decision.calib_pass},
        {"item": "calibration_min_violation_count", "value": len(decision.calib_min_violations)},
    ]
    for pair, m in decision.calib_mean_per_pair.items():
        summary_rows.append({"item": f"calib_mean_r__{pair}", "value": round(m, 4)})
    for pair, lo in decision.calib_min_per_pair.items():
        summary_rows.append({"item": f"calib_min_r__{pair}", "value": round(lo, 4)})
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "path_decision.csv", index=False)
    print("  path_decision.csv written")

    print("=" * 60)
    print(f"PATH: {decision.path}")
    print(f"  trend pass: {decision.trend_pass}  (violations: {len(decision.trend_violations)})")
    if decision.trend_violations:
        print("  trend violations:")
        for v in decision.trend_violations[:10]:
            print(f"    - {v}")
        if len(decision.trend_violations) > 10:
            print(f"    … and {len(decision.trend_violations) - 10} more")
    print(f"  calib pass: {decision.calib_pass}")
    for pair, m in decision.calib_mean_per_pair.items():
        lo = decision.calib_min_per_pair[pair]
        ok = "OK" if (m >= CALIB_MEAN_TARGET and lo >= CALIB_MIN_THRESHOLD) else "FAIL"
        print(f"    {pair:>40s}: mean r = {m:.3f}  min r = {lo:.3f}  [{ok}]")
    if decision.calib_min_violations:
        print("  calib min-r violations:")
        for v in decision.calib_min_violations[:10]:
            print(f"    - {v}")
        if len(decision.calib_min_violations) > 10:
            print(f"    … and {len(decision.calib_min_violations) - 10} more")
    print()
    print(f"NOTES: {decision.notes}")
    print("=" * 60)


if __name__ == "__main__":
    main()
