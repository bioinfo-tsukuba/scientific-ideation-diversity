# NOTICE

## Third-party material: LiveIdeaBench

The following files in this repository are derived from **LiveIdeaBench**
(<https://github.com/x66ccff/liveideabench>), MIT License,
Copyright (c) 2025 Kai Ruan, pinned at commit
`fc95111c7920cb4ca02e27f5ab9730f3886ca07c`:

- `prompts/judge_quality_prompt.json` — the `critic_prompt` entry, extracted
  verbatim from `utils/prompts.json`.
- `prompts/judge_diversity_prompt.json` — the `fluency_critic_prompt` entry,
  extracted verbatim from `utils/prompts.json`.
- The keyword subset CSVs in `data/benchmarks/`
  (`keyword_subset_1_per_category_seed42.csv`,
  `keyword_subset_5_per_category_seed42.csv`) — stratified subsets sampled from
  LiveIdeaBench's `csvs/keyword_classifications.csv`.
  `data/benchmarks/liveideabench_manifest.json` records the upstream commit and
  the SHA-256 digests of the upstream files these subsets were drawn from.

All other files in this repository are covered by the top-level [LICENSE](LICENSE).

## LiveIdeaBench license

MIT License

Copyright (c) 2025 Kai Ruan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
