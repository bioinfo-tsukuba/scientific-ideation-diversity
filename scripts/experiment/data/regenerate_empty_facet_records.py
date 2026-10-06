#!/usr/bin/env python3
"""One-off helper: regenerate samples whose structured_output has empty facets.

The main generator (``scripts/run_diversity_experiment.py``) deliberately has
*no* sample-level retry (see ``memory/feedback_no_sample_retry.md``).  Failed /
degenerate samples land in the run's ``errors.jsonl`` or — as with the empty-
facet case — sneak past validation because the current schema accepts empty
strings.  This script fills the resulting gaps for a completed run without
touching the main pipeline's retry policy: the caller sets an explicit
``--max-attempts`` budget per record, so cost is bounded.

Flow
----
1. Scan ``{run_dir}/samples.jsonl`` for records with any empty facet
   (``purpose`` / ``mechanism`` / ``evaluation`` after ``.strip()``).
2. For each such record, re-invoke ``generate_one_sample`` with the same
   ``(keyword, effort, sample_index, idea_model, provider, prompt_style)``,
   up to ``--max-attempts`` times.  Accept the first attempt whose structured
   output has all facets non-empty.  If every attempt degenerates, leave the
   original record untouched and report it.
3. Rewrite ``samples.jsonl`` with successfully regenerated records substituted
   in at the same dedup key.  A ``samples.jsonl.bak-<ISO>`` is left next to it.

Usage
-----
::

    uv run python scripts/experiment/data/regenerate_empty_facet_records.py \\
        --run-dir results/effort_diversity/phase2_claude_1180kw_30x4_facet_seed42 \\
        --prompt-file external/liveideabench/utils/prompts.json \\
        --prompt-style structured_facet \\
        --max-attempts 3
"""

from __future__ import annotations

import argparse
import json
import shutil

# Import the single-sample generator from the main runner.  Rely on Python's
# import machinery rather than duplicating it here so any fix to the request
# construction automatically applies.
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.model_registry import IDEA_MODEL_PROVIDER, IdeaModelName, PromptStyleName

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
from run_diversity_experiment import (  # type: ignore[import-not-found]  # noqa: E402
    generate_one_sample,
    resolve_prompt_template,
)

FACETS: tuple[str, ...] = ("purpose", "mechanism", "evaluation")
BG_IDEA_FIELDS: tuple[str, ...] = ("background", "idea")


def _has_empty_field(structured_output: dict, fields: tuple[str, ...]) -> list[str]:
    return [f for f in fields if not structured_output.get(f, "").strip()]


def _required_fields(prompt_style: PromptStyleName) -> tuple[str, ...]:
    if prompt_style == PromptStyleName.STRUCTURED_FACET:
        return FACETS
    return BG_IDEA_FIELDS


def _record_has_empty(record: dict, required: tuple[str, ...]) -> list[str]:
    return _has_empty_field(record.get("structured_output", {}), required)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument(
        "--prompt-style",
        type=PromptStyleName,
        default=PromptStyleName.STRUCTURED_FACET,
        choices=list(PromptStyleName),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Per-record attempt budget.  Each attempt is one full API call (cost bounded).  Default 3.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir: Path = args.run_dir.resolve()
    samples_path = run_dir / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(f"samples.jsonl not found: {samples_path}")

    required = _required_fields(args.prompt_style)
    idea_prompt_template, structured_output_schema = resolve_prompt_template(
        prompt_file=args.prompt_file,
        prompt_style=args.prompt_style,
    )

    records: list[dict] = [json.loads(line) for line in open(samples_path, encoding="utf-8")]
    bad_indices = [idx for idx, r in enumerate(records) if _record_has_empty(r, required)]
    print(f"Loaded {len(records)} records; {len(bad_indices)} have empty {required} fields.")
    if not bad_indices:
        print("Nothing to regenerate.")
        return

    regenerated = 0
    still_bad: list[tuple[int, list[str]]] = []

    for idx in bad_indices:
        orig = records[idx]
        print(
            f"\n=== record #{idx} :: kw={orig['keyword']!r} "
            f"effort={orig['effort']!r} sample_index={orig['sample_index']} ==="
        )
        orig_empty = _record_has_empty(orig, required)
        print(f"  original empty fields: {orig_empty}")

        winning: dict | None = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                provider = IDEA_MODEL_PROVIDER[IdeaModelName(orig["idea_model"])].value
                new_record = generate_one_sample(
                    keyword=orig["keyword"],
                    category=orig.get("category"),
                    idea_model=orig["idea_model"],
                    provider=provider,
                    effort_name=orig["effort"],
                    sample_index=orig["sample_index"],
                    idea_prompt_template=idea_prompt_template,
                    structured_output_schema=structured_output_schema,
                    embedding_model=orig["embedding_model"],
                    prompt_style=orig["prompt_style"],
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  attempt {attempt}: generation raised {type(exc).__name__}: {exc}")
                continue

            new_dict = new_record.model_dump(mode="json")
            new_empty = _record_has_empty(new_dict, required)
            if not new_empty:
                winning = new_dict
                print(f"  attempt {attempt}: OK (all fields non-empty)")
                break
            print(f"  attempt {attempt}: still empty in {new_empty}")

        if winning is None:
            print(f"  GAVE UP after {args.max_attempts} attempts, keeping original")
            still_bad.append((idx, orig_empty))
            continue

        records[idx] = winning
        regenerated += 1

    print(f"\nRegenerated {regenerated}/{len(bad_indices)} records.")
    if still_bad:
        print("Remaining problem records (kept as-is):")
        for idx, empty in still_bad:
            r = records[idx]
            print(
                f"  idx={idx} kw={r['keyword']!r} effort={r['effort']!r} sample_index={r['sample_index']} empty={empty}"
            )

    if regenerated == 0:
        print("No records changed; leaving samples.jsonl untouched.")
        return

    backup = samples_path.with_name(f"samples.jsonl.bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    shutil.copy2(samples_path, backup)
    print(f"Wrote backup: {backup.relative_to(run_dir)}")

    with open(samples_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Rewrote {samples_path.relative_to(run_dir)} ({len(records)} records).")


if __name__ == "__main__":
    main()
