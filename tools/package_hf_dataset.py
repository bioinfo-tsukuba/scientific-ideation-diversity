"""Package the Hugging Face dataset release from the research repo's results tree.

This is **release tooling**, not part of the paper's research pipeline. It reads
the internal experiment output tree (which lives outside this repository, passed
via the required ``--source-root``) and writes a public, renamed, compressed
copy laid out for upload to Hugging Face.

Output layout under ``--out-dir``::

    ideas/effort-axis/<model>.jsonl.gz
    ideas/prompt-axis/<cell>.jsonl.gz
    judge/diversity/effort-axis/<model>__judge-<judge>.jsonl.gz
    judge/diversity/effort-axis/<model>__judge-<judge>.errors.jsonl.gz
    judge/diversity/prompt-axis/<cell>__judge-<judge>.jsonl.gz
    judge/quality/effort-axis/<model>__judge-<judge>.jsonl.gz
    judge/quality/prompt-axis/<cell>__judge-<judge>.jsonl.gz
    errors/effort-axis/<model>.errors.jsonl.gz
    errors/prompt-axis/<cell>.errors.jsonl.gz
    embeddings/effort-axis/<model>/<embedder>/<name>.npy
    embeddings/effort-axis/<model>/<embedder>/row_keys.jsonl.gz
    embeddings/prompt-axis/<cell>/<embedder>/<name>.npy
    embeddings/prompt-axis/<cell>/<embedder>/row_keys.jsonl.gz

JSONL files are stream-copied verbatim into a gzip stream (no record is parsed or
rewritten), with one exception: when a run has a ``backfill_successes.jsonl``
(records recovered by a ``--resume-generation-output-dir`` pass), the published
ideas file is ``samples.jsonl`` followed by every backfill line whose sample key
is not already present -- the same merge + dedup as
``src.artifacts.load_run_sample_records``, with the original lines kept byte for
byte. ``.npy`` embedding arrays are re-saved, downcasting float64 to float32;
any other dtype is passed through unchanged.

``row_keys.jsonl.gz`` records, for each row of the embedder's ``embeddings*.npy``
(all arrays in one embedder directory share the row order), the sample key
``{category, keyword, prompt_style, effort, sample_index}`` taken from that
embedder's own ``samples.jsonl``. It is what lets
``tools/materialize_hf_dataset.py`` rebuild a per-embedder ``samples.jsonl`` in
array row order, including the arrays that cover only a subset of the run's
samples (e.g. GPT-5.4 prompt-axis SPECTER2, which omits ``bioterrorism``).
Gzip headers carry no timestamp, so re-running the packer is byte-reproducible.

Usage::

    python tools/package_hf_dataset.py                      # dry run (default)
    python tools/package_hf_dataset.py --no-dry-run --out-dir ./hf_dataset_out
    python tools/package_hf_dataset.py --no-dry-run --categories ideas row-keys ...  # partial rebuild
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_OUT_DIR = Path("./hf_dataset_out")

# Effort-axis runs: internal run-directory name -> public dataset name.
# Effort (low/medium/high) is a *field inside* each record of ``samples.jsonl``,
# not a subdirectory, so one run directory maps to one public file per artifact.
EFFORT_RUN_PUBLIC_NAME: dict[str, str] = {
    "phase2_claude_1180kw_30x4_facet_seed42": "claude-sonnet-4-6",
    "phase2_gpt54_1180kw_30x4_facet_seed42": "gpt-5-4",
    "phase2_gemini31pro_1180kw_30x3_facet_seed42": "gemini-3-1-pro",
}

# Prompt-axis runs live in a single ``prompt_sensitivity/`` directory whose cell
# subdirectories are named ``<generator>_<prompt>_<effort>``. The generator slugs
# on disk are ``claude`` / ``gpt54`` / ``gemini31pro`` (not ``sonnet`` / ``gpt5``).
# Prompt-axis cells keep their on-disk name as their public name.
PROMPT_SENSITIVITY_DIRNAME = "prompt_sensitivity"
PROMPT_CELL_PATTERN = re.compile(r"^(?:claude|gpt54|gemini31pro)_(?:vs|ssot)_(?:low|high)$")

# Embedder subdirectory names, verified against the on-disk tree: all four are
# present in every effort-axis run and every prompt-axis cell, spelled exactly
# as below (lower-case, hyphenated, ``:`` stripped from the Titan model id).
EMBEDDER_DIRS: tuple[str, ...] = (
    "amazontitan-embed-text-v20",
    "text-embedding-3-large",
    "allenaispecter2",
    "allenaispecter2-adhoc-query",
)

# Judge directories are discovered by glob. ``llm_judge_*`` also matches
# ``llm_judge_quality_*``, so the diversity (pairwise) scan filters those out.
# Directories prefixed with ``_`` (``_archive_smoke_*``, ``_pr261_smoke_*``) are
# development smoke tests and are excluded by the globs below.
DIVERSITY_JUDGE_GLOB = "llm_judge_*"
QUALITY_JUDGE_PREFIX = "llm_judge_quality_"
DIVERSITY_JUDGE_PREFIX = "llm_judge_"

COPY_BUFFER_BYTES = 1 << 22  # 4 MiB
CATEGORIES: tuple[str, ...] = ("ideas", "judge", "errors", "embeddings", "row-keys")

# Sample identity, as in ``src.artifacts.ROW_KEY_FIELDS``; the dedup key of
# ``load_run_sample_records`` is the same tuple without ``category``.
ROW_KEY_FIELDS: tuple[str, ...] = ("category", "keyword", "prompt_style", "effort", "sample_index")
DEDUP_KEY_FIELDS: tuple[str, ...] = ("keyword", "prompt_style", "effort", "sample_index")
BACKFILL_FILENAME = "backfill_successes.jsonl"


@dataclass(frozen=True)
class PlannedFile:
    """One source file and where it goes in the public dataset."""

    source: Path
    dest_rel: Path
    kind: str  # "jsonl-gz", "jsonl-merge-gz", "row-keys" or "npy"
    category: str  # one of CATEGORIES
    extra_source: Path | None = None  # backfill file (merge) or embeddings.npy (row-keys)

    @property
    def source_bytes(self) -> int:
        return self.source.stat().st_size


def axis_dir(axis: str) -> str:
    """Map an internal axis key to its public directory segment."""
    return {"effort": "effort-axis", "prompt": "prompt-axis"}[axis]


def plan_ideas(run_dir: Path, axis: str, public_name: str) -> list[PlannedFile]:
    """Plan the top-level ``samples.jsonl`` (merged with ``backfill_successes.jsonl``).

    Only the run's top-level copy is taken; each embedder subdirectory holds a
    re-ordered copy of the same records which is not published -- its row order
    is published instead as ``row_keys.jsonl.gz`` (see :func:`plan_row_keys`).
    """
    samples = run_dir / "samples.jsonl"
    if not samples.is_file():
        return []
    backfill = run_dir / BACKFILL_FILENAME
    return [
        PlannedFile(
            source=samples,
            dest_rel=Path("ideas") / axis_dir(axis) / f"{public_name}.jsonl.gz",
            kind="jsonl-merge-gz" if backfill.is_file() else "jsonl-gz",
            category="ideas",
            extra_source=backfill if backfill.is_file() else None,
        )
    ]


def plan_generation_errors(run_dir: Path, axis: str, public_name: str) -> list[PlannedFile]:
    """Plan the run's top-level generation ``errors.jsonl`` (absent for some runs)."""
    errors = run_dir / "errors.jsonl"
    if not errors.is_file():
        return []
    return [
        PlannedFile(
            source=errors,
            dest_rel=Path("errors") / axis_dir(axis) / f"{public_name}.errors.jsonl.gz",
            kind="jsonl-gz",
            category="errors",
        )
    ]


def _plan_judge_family(
    run_dir: Path,
    axis: str,
    public_name: str,
    judge_dirs: list[Path],
    prefix: str,
    record_name: str,
    errors_name: str,
    family: str,
) -> list[PlannedFile]:
    """Plan one judge family (``diversity`` or ``quality``) for a run."""
    planned: list[PlannedFile] = []
    out_dir = Path("judge") / family / axis_dir(axis)
    for judge_dir in sorted(judge_dirs):
        judge = judge_dir.name[len(prefix) :]
        records = judge_dir / record_name
        if not records.is_file():
            continue
        planned.append(
            PlannedFile(
                source=records,
                dest_rel=out_dir / f"{public_name}__judge-{judge}.jsonl.gz",
                kind="jsonl-gz",
                category="judge",
            )
        )
        errors = judge_dir / errors_name
        if errors.is_file():
            planned.append(
                PlannedFile(
                    source=errors,
                    dest_rel=out_dir / f"{public_name}__judge-{judge}.errors.jsonl.gz",
                    kind="jsonl-gz",
                    category="judge",
                )
            )
    return planned


def plan_judges(run_dir: Path, axis: str, public_name: str) -> list[PlannedFile]:
    """Plan both judge families for a run."""
    all_judge_dirs = [p for p in run_dir.glob(DIVERSITY_JUDGE_GLOB) if p.is_dir()]
    quality_dirs = [p for p in all_judge_dirs if p.name.startswith(QUALITY_JUDGE_PREFIX)]
    diversity_dirs = [p for p in all_judge_dirs if not p.name.startswith(QUALITY_JUDGE_PREFIX)]

    planned = _plan_judge_family(
        run_dir,
        axis,
        public_name,
        diversity_dirs,
        DIVERSITY_JUDGE_PREFIX,
        "pairwise.jsonl",
        "pairwise.errors.jsonl",
        "diversity",
    )
    planned += _plan_judge_family(
        run_dir,
        axis,
        public_name,
        quality_dirs,
        QUALITY_JUDGE_PREFIX,
        "quality.jsonl",
        "quality.errors.jsonl",
        "quality",
    )
    return planned


def plan_embeddings(run_dir: Path, axis: str, public_name: str) -> list[PlannedFile]:
    """Plan every ``embeddings*.npy`` under each known embedder subdirectory."""
    planned: list[PlannedFile] = []
    for embedder in EMBEDDER_DIRS:
        embedder_dir = run_dir / embedder
        if not embedder_dir.is_dir():
            continue
        for npy in sorted(embedder_dir.glob("embeddings*.npy")):
            planned.append(
                PlannedFile(
                    source=npy,
                    dest_rel=Path("embeddings") / axis_dir(axis) / public_name / embedder / npy.name,
                    kind="npy",
                    category="embeddings",
                )
            )
    return planned


def plan_row_keys(run_dir: Path, axis: str, public_name: str) -> list[PlannedFile]:
    """Plan ``row_keys.jsonl.gz`` for every embedder directory with an ``embeddings.npy``."""
    planned: list[PlannedFile] = []
    for embedder in EMBEDDER_DIRS:
        embedder_dir = run_dir / embedder
        samples = embedder_dir / "samples.jsonl"
        npy = embedder_dir / "embeddings.npy"
        if not (samples.is_file() and npy.is_file()):
            continue
        planned.append(
            PlannedFile(
                source=samples,
                dest_rel=Path("embeddings") / axis_dir(axis) / public_name / embedder / "row_keys.jsonl.gz",
                kind="row-keys",
                category="row-keys",
                extra_source=npy,
            )
        )
    return planned


def plan_run(run_dir: Path, axis: str, public_name: str) -> list[PlannedFile]:
    """Plan every public artifact derived from a single run directory."""
    return (
        plan_ideas(run_dir, axis, public_name)
        + plan_judges(run_dir, axis, public_name)
        + plan_generation_errors(run_dir, axis, public_name)
        + plan_embeddings(run_dir, axis, public_name)
        + plan_row_keys(run_dir, axis, public_name)
    )


def discover_prompt_cells(source_root: Path) -> list[tuple[Path, str]]:
    """Return (cell_dir, public_name) for each prompt-axis cell directory."""
    ps_root = source_root / PROMPT_SENSITIVITY_DIRNAME
    cells: list[tuple[Path, str]] = []
    for child in sorted(ps_root.iterdir()):
        if child.is_dir() and PROMPT_CELL_PATTERN.match(child.name):
            cells.append((child, child.name))
    return cells


def build_plan(source_root: Path) -> list[PlannedFile]:
    """Plan the whole release from the source results tree."""
    planned: list[PlannedFile] = []
    for run_name, public_name in EFFORT_RUN_PUBLIC_NAME.items():
        planned += plan_run(source_root / run_name, "effort", public_name)
    for cell_dir, public_name in discover_prompt_cells(source_root):
        planned += plan_run(cell_dir, "prompt", public_name)
    return planned


@contextmanager
def _open_gz_for_write(dest: Path) -> Iterator[gzip.GzipFile]:
    """Gzip writer with an empty filename and zero mtime in the header (reproducible bytes)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
        yield out


def write_jsonl_gz(source: Path, dest: Path) -> None:
    """Stream-copy a JSONL file into a gzip stream, byte for byte."""
    with source.open("rb") as src, _open_gz_for_write(dest) as out:
        shutil.copyfileobj(src, out, COPY_BUFFER_BYTES)


def _key(payload: dict, fields: tuple[str, ...]) -> tuple:
    return tuple(payload.get(field) for field in fields)


def write_merged_jsonl_gz(samples: Path, backfill: Path, dest: Path) -> tuple[int, int]:
    """Write ``samples.jsonl`` + the backfill lines whose sample key is new.

    Equivalent to ``load_run_sample_records`` (which keeps the first record per
    ``(keyword, prompt_style, effort, sample_index)``, samples before backfill),
    but keeps every published line byte for byte. Returns (samples_kept, backfill_added).
    """
    seen: set[tuple] = set()
    kept = added = 0
    with _open_gz_for_write(dest) as out:
        for path, is_backfill in ((samples, False), (backfill, True)):
            with path.open("rb") as src:
                for line in src:
                    if not line.strip():
                        continue
                    key = _key(json.loads(line), DEDUP_KEY_FIELDS)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.write(line if line.endswith(b"\n") else line + b"\n")
                    if is_backfill:
                        added += 1
                    else:
                        kept += 1
    return kept, added


def write_row_keys_gz(samples: Path, npy: Path, dest: Path) -> int:
    """Write one ``ROW_KEY_FIELDS`` object per line of an embedder's ``samples.jsonl``."""
    n_rows = np.load(npy, mmap_mode="r").shape[0]
    n = 0
    with samples.open("rb") as src, _open_gz_for_write(dest) as out:
        for line in src:
            if not line.strip():
                continue
            payload = json.loads(line)
            row = {field: payload.get(field) for field in ROW_KEY_FIELDS}
            out.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
            n += 1
    if n != n_rows:
        dest.unlink()
        raise ValueError(f"{samples} has {n} records but {npy} has {n_rows} rows")
    return n


def write_npy(source: Path, dest: Path) -> None:
    """Re-save an embedding array, downcasting float64 to float32.

    The array is memory-mapped for reading; the float64 cast materialises one
    float32 copy of the array in RAM (roughly half the source file size).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    array = np.load(source, mmap_mode="r")
    if array.dtype == np.float64:
        np.save(dest, array.astype(np.float32))
    else:
        np.save(dest, array)


def execute(planned: list[PlannedFile], out_dir: Path) -> None:
    """Write every planned file into ``out_dir``."""
    for item in planned:
        dest = out_dir / item.dest_rel
        if item.kind == "jsonl-gz":
            write_jsonl_gz(item.source, dest)
        elif item.kind == "jsonl-merge-gz":
            kept, added = write_merged_jsonl_gz(item.source, item.extra_source, dest)
            print(f"  merged {item.dest_rel}: {kept} samples + {added} backfill")
        elif item.kind == "row-keys":
            write_row_keys_gz(item.source, item.extra_source, dest)
        else:
            write_npy(item.source, dest)


def print_plan(planned: list[PlannedFile]) -> None:
    """Print the dry-run plan: source size and destination for every file."""
    print("Planned files (dry run — nothing written):")
    for item in planned:
        print(f"  {item.source_bytes:>14,d} B  {item.source}")
        print(f"  {'':>14}    -> {item.dest_rel}")


def print_manifest(planned: list[PlannedFile], out_dir: Path, dry_run: bool) -> None:
    """Print the output manifest with per-file sizes and per-category totals."""
    size_label = "source bytes (output not written)" if dry_run else "output bytes"
    print(f"\nManifest ({size_label}):")

    counts: dict[str, int] = {c: 0 for c in CATEGORIES}
    totals: dict[str, int] = {c: 0 for c in CATEGORIES}

    for item in planned:
        dest = out_dir / item.dest_rel
        size = item.source_bytes if dry_run else dest.stat().st_size
        counts[item.category] += 1
        totals[item.category] += size
        print(f"  {size:>16,d} B  {item.dest_rel}")

    print("\nTotals by category:")
    for category in CATEGORIES:
        print(f"  {category:<12} {counts[category]:>5,d} files  {totals[category]:>18,d} B")
    print(f"  {'ALL':<12} {sum(counts.values()):>5,d} files  {sum(totals.values()):>18,d} B")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Internal results/effort_diversity tree to package.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Destination directory for the packaged dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=CATEGORIES,
        default=list(CATEGORIES),
        help="Only plan/write these file categories (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Print the planned file list without writing anything (default).",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Actually write the packaged dataset.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root: Path = args.source_root
    out_dir: Path = args.out_dir
    dry_run: bool = args.dry_run

    print(f"source-root: {source_root}")
    print(f"out-dir:     {out_dir}")
    print(f"mode:        {'dry run' if dry_run else 'write'}\n")

    planned = [item for item in build_plan(source_root) if item.category in set(args.categories)]

    if dry_run:
        print_plan(planned)
    else:
        execute(planned, out_dir)

    print_manifest(planned, out_dir, dry_run)


if __name__ == "__main__":
    main()
