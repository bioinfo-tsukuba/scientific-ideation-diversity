# Reproducing the paper's items

All paths are repository-relative. Scripts are run from the repository root:

```bash
uv run python <script>
```

Analysis scripts resolve their inputs against `REPO_ROOT / "results" / "effort_diversity"`,
so the Hugging Face dataset must be materialised under `results/` first (see
[README.md](README.md)). Run-directory names below are the internal names used by
the scripts; the public dataset uses the mapped names documented in
`tools/package_hf_dataset.py`.

Notation used in the input column:
- `<phase2>` = `results/effort_diversity/phase2_{claude_1180kw_30x4,gpt54_1180kw_30x4,gemini31pro_1180kw_30x3}_facet_seed42` (the three effort-axis runs).
- `<ps>` = `results/effort_diversity/prompt_sensitivity` (prompt-axis aggregates and the `{claude,gpt54,gemini31pro}_{vs,ssot}_{low,high}` cells).
- `<run>` = any one of the 15 run/cell directories the Stage 0 scripts below operate on identically: the three `<phase2>` dirs, or one of the twelve `<ps>/<cell>` dirs (`{claude,gpt54,gemini31pro}_{vs,ssot}_{low,high}`).

## Stage 0 — rebuild intermediates from raw data

The Hugging Face dataset publishes only raw files: generated ideas
(`samples.jsonl`), raw judge logs (`pairwise.jsonl`, `quality.jsonl`, and their
`*.errors.jsonl` siblings), and fp32 embeddings (`.npy`). No aggregated or
intermediate CSV is published. After running
`tools/materialize_hf_dataset.py` (see [README.md](README.md)), the table
below rebuilds every intermediate CSV that the per-item rows further down
this document read, purely from those raw files (plus, in one case, a live
API call — flagged below). Rows are ordered so that running them top to
bottom satisfies all dependencies; a row's "raw inputs" column may itself
reference an intermediate built by an earlier row.

All scripts are run from the repository root as `uv run python <script> ...`.
`<judge>` below ranges over the two cross-checked judges used elsewhere in
this document, `gpt_4_1` and `claude_haiku_4_5_20251001` (CLI value
`gpt-4.1` / `claude-haiku-4-5-20251001`); `<model>` ranges over
`claude` / `gpt54` / `gemini31pro`.

| Intermediate file | Builder script | Raw inputs | Notes |
| --- | --- | --- | --- |
| `<run>/<embedder>/{summary_by_keyword_effort.csv, summary.csv, summary_by_category_effort.csv, centroid_distances.csv, pca_coordinates.csv}` | `scripts/run_diversity_experiment.py --resume-output-dir <phase2> --idea-model <model> --provider <provider> --embed-models <embedder...>` (effort-axis) or `scripts/experiment/prompt_sensitivity/run_prompt_sensitivity.py --model <model> --prompt <prompt> --effort <effort> --resume-output-dir <ps>/<cell>` (prompt-axis) | `<run>/samples.jsonl`, `<run>/<embedder>/embeddings.npy` | Resume mode skips generation. The embedding step reuses the cached `.npy` array whenever its row count matches `samples.jsonl` (true for the published data), so no generation or embedding API calls are made — only the aggregation CSVs are recomputed. `--idea-model`/`--provider` (effort-axis) and `--model`/`--prompt`/`--effort` (prompt-axis) must match the run; see `src/model_registry.py` and the on-disk cell name. |
| `<phase2>/reasoning_token_counts.csv` (Claude run only) | `scripts/experiment/data/count_claude_reasoning_tokens.py <phase2_claude_dir>` | `<phase2>/samples.jsonl` | Calls the live Anthropic `count_tokens` API once per sample to isolate reasoning tokens by subtraction (see the script's docstring). Needs network access and Anthropic credentials — the only row in this table that is not a pure offline recompute. GPT-5.4 / Gemini 3.1 Pro reasoning tokens are read directly out of `samples.jsonl`'s raw API response fields, no separate CSV needed. |
| `<run>/llm_judge_<judge>/{abcd_by_stratum_effort.csv, effort_effect.csv, distance_by_label.csv, position_bias.csv, rank_saturation.csv, per_keyword_frac_a.csv}` | `scripts/experiment/llm_judge/analyze_llm_judge.py --run-dir <run> --judge-model <judge>` | `<run>/llm_judge_<judge>/pairwise.jsonl` | |
| `<run>/llm_judge_<judge>/calibration.csv` | `scripts/experiment/llm_judge/analyze_llm_judge_calibration.py --run-dir <run> --judge-model <judge>` | `<run>/llm_judge_<judge>/pairwise.jsonl` | |
| `<run>/llm_judge_<judge>/facet_attribution.csv` | `scripts/experiment/llm_judge/analyze_llm_judge_facet_attribution.py --run-dir <run> --judge-model <judge>` | `<run>/llm_judge_<judge>/pairwise.jsonl`, `<run>/samples.jsonl` (run-level), `<run>/text-embedding-3-large/embeddings_{purpose,mechanism,evaluation}.npy` | |
| `<run>/llm_judge_quality_<judge>/{by_effort.csv, per_keyword_effort.csv}` | `scripts/experiment/llm_judge/aggregate_llm_judge_quality_per_keyword.py --run-dir <run> --judge-model <judge>` | `<run>/llm_judge_quality_<judge>/quality.jsonl` | |
| `<run>/cross_judge_gpt_4_1_vs_claude_haiku_4_5_20251001/{pairwise_agreement_summary.csv, pairwise_confusion.csv, pairwise_by_stratum.csv}` | `scripts/experiment/llm_judge/analyze_cross_judge_agreement.py --run-dir <run> --judge-a gpt-4.1 --judge-b claude-haiku-4-5-20251001 --mode pairwise` | `<run>/llm_judge_gpt_4_1/pairwise.jsonl`, `<run>/llm_judge_claude_haiku_4_5_20251001/pairwise.jsonl` | |
| `<run>/cross_judge_gpt_4_1_vs_claude_haiku_4_5_20251001/{by_effort_gpt_4_1.csv, by_effort_claude_haiku_4_5_20251001.csv, per_keyword_effort_gpt_4_1.csv, per_keyword_effort_claude_haiku_4_5_20251001.csv, quality_agreement.csv, quality_score_matrix_{originality,feasibility,clarity}.csv}` | `scripts/experiment/llm_judge/analyze_cross_judge_agreement.py --run-dir <run> --judge-a gpt-4.1 --judge-b claude-haiku-4-5-20251001 --mode quality` | `<run>/llm_judge_quality_gpt_4_1/quality.jsonl`, `<run>/llm_judge_quality_claude_haiku_4_5_20251001/quality.jsonl` | |
| `<ps>/q1_q4_{claude,gpt54,gemini31pro}.csv`, `<ps>/q1_q4_all_models.csv` | `scripts/experiment/prompt_sensitivity/select_per_model_q1_q4.py --phase2-root results/effort_diversity --n-per-stratum 50 --seed 42 --output-dir <ps>` | `<phase2>/text-embedding-3-large/summary_by_keyword_effort.csv` (the `summary_by_keyword_effort.csv` row above, for all three models) | The `summary_by_keyword_effort.csv` row above must have been run for all three `<phase2>` dirs first. |
| `<ps>/judge_keyword_filters/q1q4_{claude,gpt54,gemini31pro}.csv` | `scripts/experiment/data/build_judge_keyword_filters.py` | `<ps>/q1_q4_all_models.csv` (row above) | |
| `<ps>/{model}_token_pareto.csv` | `scripts/experiment/prompt_sensitivity/analyze_prompt_sensitivity_token_pareto.py --model <model>` | `<phase2>/text-embedding-3-large/summary_by_keyword_effort.csv` and the model's four `<ps>/{model}_{vs,ssot}_{low,high}/text-embedding-3-large/summary_by_keyword_effort.csv` (the `summary_by_keyword_effort.csv` row above), `<ps>/q1_q4_{model}.csv` (row above), `<run>/samples.jsonl` | |
| `<ps>/{model}_embedding_purity.csv`, `<ps>/{model}_embedding_purity__{amazontitan-embed-text-v20,allenaispecter2-adhoc-query}.csv` | `scripts/experiment/prompt_sensitivity/analyze_prompt_sensitivity_embedding_purity.py --model <model> [--embedding-model <embedder>]` | `<ps>/q1_q4_{model}.csv` (row above), each relevant cell's `<embedder>/{samples.jsonl,embeddings.npy}` | Default embedder is `text-embedding-3-large` (no filename suffix); pass `--embedding-model` for the other two. |
| `<ps>/cross_embedder_purity_qb.csv` | `scripts/figures/aggregate_cross_embedder_purity.py` | `<ps>/{claude,gpt54,gemini31pro}_embedding_purity[__{amazontitan-embed-text-v20,allenaispecter2-adhoc-query}].csv` (row above; 9 files) | |
| `<ps>/cross_judge_kappa_summary.csv` | `scripts/figures/aggregate_cross_judge_kappa.py` | each `<ps>/<cell>/cross_judge_gpt_4_1_vs_claude_haiku_4_5_20251001/pairwise_agreement_summary.csv` (row above; 12 cells) | |
| `<ps>/paper_table_quality_per_prompt_cell.csv` | `scripts/figures/aggregate_quality_per_prompt_cell.py` | `<ps>/q1_q4_{model}.csv` (row above) and each of `<phase2>` + the model's 4 cells' `cross_judge_gpt_4_1_vs_claude_haiku_4_5_20251001/per_keyword_effort_gpt_4_1.csv` (row above) | |

## Per-item mapping

Once Stage 0 has been run, every input path below is either a raw HF dataset
file (as materialized by `tools/materialize_hf_dataset.py`) or an
intermediate CSV built by one of the Stage 0 rows above.

| Paper item | Script | Input data | Output |
| --- | --- | --- | --- |
| Fig. 1a | `scripts/figures/plot_fig1_pair_distance.py` | `<phase2>/reasoning_token_counts.csv`, `<phase2>/text-embedding-3-large/summary_by_keyword_effort.csv`, `<phase2>/samples.jsonl` | `outputs/figures/pair_distance_vs_reasoning_tokens.pdf` |
| Fig. 1b | `scripts/figures/plot_fig_prompt_sensitivity_pareto.py` | `<ps>/{claude,gpt54,gemini31pro}_token_pareto.csv` | `outputs/figures/prompt_sensitivity_token_pareto.pdf` |
| Fig. 1 (composite a+b) | `scripts/figures/build_fig1_composite.py` | the two PDFs produced by the two rows above | `outputs/figures/fig1_composite.pdf` |
| Fig. 2 | `scripts/figures/plot_fig_prompt_sensitivity_qb_purity.py` | `<ps>/cross_embedder_purity_qb.csv` (built by `scripts/figures/aggregate_cross_embedder_purity.py`) | `--output`, default `outputs/figures/prompt_sensitivity_qb_purity.png` |
| Fig. 3 | `scripts/figures/plot_fig5_quality_two_judges.py` | `<phase2>/llm_judge_quality_gpt_4_1/by_effort.csv`, `<phase2>/llm_judge_quality_claude_haiku_4_5_20251001/by_effort.csv` | `outputs/figures/quality_two_judges.pdf` |
| Table 1 + Table 3 | `scripts/figures/compute_per_facet_pair_distance.py` | `<phase2>/{text-embedding-3-large,amazontitan-embed-text-v20,allenaispecter2-adhoc-query}/embeddings.npy` and `embeddings_{purpose,mechanism,evaluation}.npy`, plus `<phase2>/text-embedding-3-large/samples.jsonl` (read from the embedder subdir, not the run's top-level copy — needs the per-embedder `samples.jsonl` symlink `tools/materialize_hf_dataset.py` creates) | `outputs/tables/per_facet_pair_distance_summary.csv`, `outputs/tables/table2_per_facet_distance_low_high.csv`, `outputs/tables/app_per_facet_per_embedding.csv` |
| Table 2 | `scripts/analyze_lexical_diversity.py` | `--run-dir` (required): one effort-axis or prompt-axis run dir, reading its `samples.jsonl` | `lexical_diversity_aggregate.csv`, `lexical_diversity_per_keyword.csv` in `--output-dir` (defaults to the run dir) |
| Vendi scores | `scripts/experiment/diversity/compute_vendi_score.py` and `scripts/experiment/diversity/compute_vendi_score_q1q4.py` | effort axis: `<phase2>/<embedder>/{samples.jsonl,embeddings.npy}`. Q1/Q4 prompt axis: same files under `<ps>/<cell>/<embedder>/`, plus `<ps>/q1_q4_all_models.csv` | `results/effort_diversity/vendi_score/vendi_score_by_keyword_effort.csv` and `.../vendi_score_q1q4_by_keyword.csv` |
| cross-judge kappa | `scripts/figures/aggregate_cross_judge_kappa.py` | `<ps>/<cell>/cross_judge_gpt_4_1_vs_claude_haiku_4_5_20251001/pairwise_agreement_summary.csv` | `<ps>/cross_judge_kappa_summary.csv` |
| quality-per-prompt | `scripts/figures/aggregate_quality_per_prompt_cell.py` | `<ps>/q1_q4_{claude,gpt54,gemini31pro}.csv`, `<ps>/<cell>/cross_judge_gpt_4_1_vs_claude_haiku_4_5_20251001/per_keyword_effort_gpt_4_1.csv`, and the matching `<phase2>/cross_judge_gpt_4_1_vs_claude_haiku_4_5_20251001/per_keyword_effort_gpt_4_1.csv` | `<ps>/paper_table_quality_per_prompt_cell.csv` |
| strong-judge check | `scripts/experiment/llm_judge/analyze_strong_judge_quality.py` | `<phase2>/llm_judge_quality_*/quality.jsonl` and `<ps>/<cell>/llm_judge_quality_*/quality.jsonl` for the weak and strong judge slugs, filtered by `<ps>/judge_keyword_filters/q1q4_{claude,gpt54,gemini31pro}.csv` | `results/effort_diversity/rebuttal_strong_judge_path_decision/{condition_mean,trend_delta,vs_low_delta,calibration_r,path_decision}.csv` |
| token-diversity correlation | `scripts/figures/token_diversity_correlation.py` | `<phase2>/reasoning_token_counts.csv` (Claude), `<phase2>/samples.jsonl` (GPT-5.4 / Gemini reasoning-token subsets), `<phase2>/text-embedding-3-large/summary_by_keyword_effort.csv` | `outputs/tables/token_vs_diversity_correlation.csv` |
| 1-NN purity | `scripts/figures/aggregate_cross_embedder_purity.py` | `<ps>/{model}_embedding_purity[__<embedder>].csv` (produced by `scripts/experiment/prompt_sensitivity/analyze_prompt_sensitivity_embedding_purity.py`) | `<ps>/cross_embedder_purity_qb.csv` |
| Fluency / ABCD aggregation | `scripts/figures/build_fluency_aggregation.py` | `<phase2>/llm_judge_*/abcd_by_stratum_effort.csv` | aggregated fluency table |

## Notes

1. The per-model Q1/Q4 *selector* is
   `scripts/experiment/prompt_sensitivity/select_per_model_q1_q4.py` (Stage 0);
   there is no separate figure-side Q1/Q4 script in this paper's item list.
