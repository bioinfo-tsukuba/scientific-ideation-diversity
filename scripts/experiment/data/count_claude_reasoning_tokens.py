"""Count Claude Sonnet 4.6 reasoning tokens per sample by subtraction.

For each sample in a ``samples.jsonl`` produced by the effort-diversity
generation pipeline, derive the reasoning-only (thinking) token count via::

    reasoning_tokens = output_tokens - count_tokens(full_response) + system_overhead

Bedrock / Anthropic APIs bill thinking tokens as part of ``output_tokens``
without exposing a subset field (see issue #110 for the exhaustive audit of
alternative routes), so the only way to isolate reasoning tokens is to call
``client.messages.count_tokens`` on the verbatim visible answer (the JSON
that ``full_response`` already stores) and subtract it from the billed
total.

Anthropic's doc notes::

    Token counts may include tokens added automatically by Anthropic for
    system optimizations.  You are not billed for system-added tokens.

i.e.  ``count_tokens(s) = true_tokens(s) + system_overhead`` while
``output_tokens`` is the *billed* value (no overhead).  Therefore::

    output_tokens - count_tokens(full_response)
      = (true_thinking + true_visible) - (true_visible + system_overhead)
      = true_thinking - system_overhead

and we must add ``system_overhead`` back to recover ``true_thinking``.

``system_overhead`` is calibrated once at startup as
``count_tokens(".") - 1`` (a single period tokenizes to exactly one BPE
token in every Claude tokenizer variant; the remainder is the per-call
boilerplate the API adds around the ``user`` message).  The value (typically
7-10) is logged and also stored as a constant column in the output CSV for
downstream auditability.

Ground-truth clamp: when the generation response lacks a thinking block
(``has_reasoning_block=False``) there were no reasoning tokens by
definition, regardless of what the subtraction arithmetic produces.  The
derived value for those samples is stored as ``reasoning_tokens_raw`` for
audit and the published ``reasoning_tokens`` column is forced to 0.

Usage::

    uv run python scripts/experiment/data/count_claude_reasoning_tokens.py \\
        results/effort_diversity/phase2_claude_1180kw_30x4_facet_seed42 \\
        --max-workers 20

The script is idempotent: lines already present in ``reasoning_token_counts.csv``
are skipped, so reruns after partial failure pick up where they left off.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from anthropic import Anthropic

from src.model_registry import IDEA_MODEL_ANTHROPIC_NATIVE_ID, IdeaModelName
from src.providers.llm_judge.anthropic_count_tokens import (
    count_visible_answer_tokens,
    measure_system_overhead,
)

# Claude generation model whose samples this script processes.  Used
# both as the Anthropic-native id for ``count_tokens`` and as the key
# into :data:`IDEA_MODEL_ANTHROPIC_NATIVE_ID` (single-source).
_CLAUDE_MODEL = IdeaModelName.CLAUDE_SONNET_4_6
CLAUDE_API_MODEL = IDEA_MODEL_ANTHROPIC_NATIVE_ID[_CLAUDE_MODEL]
CSV_NAME = "reasoning_token_counts.csv"
ERRORS_NAME = "reasoning_token_counts.errors.jsonl"

CSV_FIELDS = [
    "sample_jsonl_index",
    "keyword",
    "effort",
    "sample_index",
    "output_tokens",
    "visible_answer_tokens_raw",
    "system_overhead",
    "has_reasoning_block",
    "reasoning_tokens_raw",
    "reasoning_tokens",
]

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("count_claude_reasoning_tokens")


@dataclass(frozen=True)
class Sample:
    index: int
    keyword: str
    effort: str
    sample_index: int
    output_tokens: int
    has_reasoning_block: bool
    full_response: str


def iter_samples(samples_path: Path):
    with samples_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            yield Sample(
                index=i,
                keyword=rec["keyword"],
                effort=rec["effort"],
                sample_index=int(rec["sample_index"]),
                output_tokens=int(rec["output_tokens"]),
                has_reasoning_block=bool(rec["has_reasoning_block"]),
                full_response=rec["full_response"],
            )


def load_done_indices(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    done: set[int] = set()
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(int(row["sample_jsonl_index"]))
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_dir",
        type=Path,
        help="run directory containing samples.jsonl "
        "(e.g., results/effort_diversity/phase2_claude_1180kw_30x4_facet_seed42)",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="ThreadPoolExecutor concurrency (default: 20; raise to ~100 on Tier 4)",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="log progress every N completions",
    )
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    samples_path = run_dir / "samples.jsonl"
    csv_path = run_dir / CSV_NAME
    errors_path = run_dir / ERRORS_NAME

    if not samples_path.exists():
        logger.error("samples.jsonl not found at %s", samples_path)
        return 2
    if "ANTHROPIC_API_KEY" not in os.environ:
        logger.error("ANTHROPIC_API_KEY is not set")
        return 2

    client = Anthropic()

    overhead = measure_system_overhead(client, model=_CLAUDE_MODEL)
    logger.info(
        "system overhead = %d tokens (= count_tokens('.') - 1); "
        "added back into derived reasoning_tokens to recover true_thinking",
        overhead,
    )

    done = load_done_indices(csv_path)
    logger.info("resume: %d samples already in %s", len(done), csv_path.name)

    total_samples = sum(1 for _ in samples_path.open(encoding="utf-8"))
    pending = [s for s in iter_samples(samples_path) if s.index not in done]
    logger.info(
        "pending: %d / %d samples (max_workers=%d)",
        len(pending),
        total_samples,
        args.max_workers,
    )
    if not pending:
        return 0

    is_new_csv = not csv_path.exists()
    n_ok = 0
    n_err = 0

    with csv_path.open("a", encoding="utf-8", newline="") as csv_f, errors_path.open("a", encoding="utf-8") as err_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
        if is_new_csv:
            writer.writeheader()
            csv_f.flush()

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    count_visible_answer_tokens, client, s.full_response, model=_CLAUDE_MODEL
                ): s
                for s in pending
            }
            for i, fut in enumerate(as_completed(futures), start=1):
                sample = futures[fut]
                try:
                    visible_raw = fut.result()
                except Exception as exc:  # noqa: BLE001
                    err_f.write(
                        json.dumps(
                            {
                                "sample_jsonl_index": sample.index,
                                "keyword": sample.keyword,
                                "effort": sample.effort,
                                "sample_index": sample.sample_index,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        + "\n"
                    )
                    err_f.flush()
                    n_err += 1
                else:
                    # visible_raw = true_visible + system_overhead
                    # output_tokens = true_thinking + true_visible (billed, no overhead)
                    # => true_thinking = output_tokens - visible_raw + system_overhead
                    reasoning_raw = sample.output_tokens - visible_raw + overhead
                    # has_reasoning_block=False means the generation response had
                    # no thinking block, so the true reasoning token count is 0
                    # by definition — any positive derived value on those samples
                    # is tokenizer-boundary noise, and we keep it in the "_raw"
                    # column for audit but clamp the published value to 0.
                    reasoning = reasoning_raw if sample.has_reasoning_block else 0
                    writer.writerow(
                        {
                            "sample_jsonl_index": sample.index,
                            "keyword": sample.keyword,
                            "effort": sample.effort,
                            "sample_index": sample.sample_index,
                            "output_tokens": sample.output_tokens,
                            "visible_answer_tokens_raw": visible_raw,
                            "system_overhead": overhead,
                            "has_reasoning_block": sample.has_reasoning_block,
                            "reasoning_tokens_raw": reasoning_raw,
                            "reasoning_tokens": reasoning,
                        }
                    )
                    csv_f.flush()
                    n_ok += 1

                if i % args.progress_every == 0 or i == len(pending):
                    logger.info(
                        "progress: %d/%d (ok=%d, err=%d)",
                        i,
                        len(pending),
                        n_ok,
                        n_err,
                    )

    logger.info("done: %d succeeded, %d failed", n_ok, n_err)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
