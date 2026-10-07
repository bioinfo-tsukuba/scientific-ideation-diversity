"""Materialize a downloaded Hugging Face dataset tree into the internal results layout.

This is the **inverse** of ``tools/package_hf_dataset.py``: given a local copy of
the public dataset (in the layout ``package_hf_dataset.py`` produces), it
reconstructs the internal directory tree that the analysis scripts under
``scripts/`` expect, so the paper's figures/tables can be rebuilt starting from
the published raw data alone. This is **release tooling**, not part of the
paper's research pipeline.

Internal layout written under ``--results-root`` (default: ``results/effort_diversity``
relative to the current working directory)::

    <internal_run>/samples.jsonl
    <internal_run>/errors.jsonl
    <internal_run>/llm_judge_<judge>/{pairwise.jsonl,pairwise.errors.jsonl}
    <internal_run>/llm_judge_quality_<judge>/{quality.jsonl,quality.errors.jsonl}
    <internal_run>/<embedder>/embeddings*.npy
    <internal_run>/<embedder>/samples.jsonl              (run records, in .npy row order)
    <internal_run>/<embedder>/embedding_row_keys.jsonl   (one sample key per .npy row)
    prompt_sensitivity/<cell>/...                        (same sub-layout as one run)

``<internal_run>`` is one of the three effort-axis run directories (recovered
from the public model name via the inverse of
``package_hf_dataset.EFFORT_RUN_PUBLIC_NAME``); prompt-axis cells keep their
on-disk name as their public name, so ``prompt_sensitivity/<cell>`` needs no
lookup table.

JSONL files are gunzipped byte for byte (no record is parsed or rewritten).
``.npy`` embedding arrays are copied as-is: the published arrays are already
float32 (``package_hf_dataset.py`` downcasts on the way *out*), so no dtype
conversion is needed on the way back in.

Several analysis scripts additionally expect a ``samples.jsonl`` inside each
embedder subdirectory, next to ``embeddings.npy``, and index the arrays by its
line number. The run-level file is *not* in that order (it is in generation
order, and some arrays cover only a subset of the run, e.g. GPT-5.4 prompt-axis
SPECTER2 without ``bioterrorism``), so a symlink to it would silently misalign
rows. Each embedder's ``samples.jsonl`` is therefore written as a real file:
the run-level lines (byte for byte) re-ordered by the dataset's
``row_keys.jsonl.gz``. For a dataset copy without ``row_keys.jsonl.gz`` the
canonical ``src.artifacts.sort_sample_records`` order is used, which is the
order every published array was written in, and only when the row count
matches. The keys are also written to ``embedding_row_keys.jsonl``, which
``scripts/run_diversity_experiment.py`` checks before reusing a published array
in resume mode.

Usage::

    python tools/materialize_hf_dataset.py --dataset-dir ./hf_dataset_out           # dry run (default)
    python tools/materialize_hf_dataset.py --dataset-dir ./hf_dataset_out --no-dry-run
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from package_hf_dataset import (
    DIVERSITY_JUDGE_PREFIX,
    EFFORT_RUN_PUBLIC_NAME,
    EMBEDDER_DIRS,
    PROMPT_SENSITIVITY_DIRNAME,
    QUALITY_JUDGE_PREFIX,
    axis_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.artifacts import EMBEDDING_ROW_KEYS_FILENAME, ROW_KEY_FIELDS, row_key_from_payload  # noqa: E402
from src.model_registry import effort_sort_key  # noqa: E402

DEFAULT_RESULTS_ROOT = Path("results/effort_diversity")
ROW_KEYS_PUBLIC_NAME = "row_keys.jsonl.gz"

# Public model name -> internal run-directory name (inverse of EFFORT_RUN_PUBLIC_NAME).
INTERNAL_RUN_BY_PUBLIC_NAME: dict[str, str] = {public: run for run, public in EFFORT_RUN_PUBLIC_NAME.items()}

COPY_BUFFER_BYTES = 1 << 22  # 4 MiB
CATEGORIES: tuple[str, ...] = ("ideas", "judge", "errors", "embeddings", "embedder-samples")


@dataclass(frozen=True)
class PlannedFile:
    """One file to materialize into the internal results tree.

    ``source`` is the HF dataset file to read. For ``kind == "embedder-samples"``
    it is the embedder's ``row_keys.jsonl.gz`` (``None`` if the dataset copy has
    none), ``run_samples_rel`` is the run-level ``samples.jsonl`` this same plan
    creates, and ``npy`` is the dataset array whose rows the output must match.
    ``dest_rel`` is then the embedder directory.
    """

    dest_rel: Path
    kind: str  # "gunzip", "npy-copy", or "embedder-samples"
    category: str  # one of CATEGORIES
    source: Path | None = None
    run_samples_rel: Path | None = None
    npy: Path | None = None

    @property
    def source_bytes(self) -> int:
        """Size of the HF dataset source file; 0 when there is none."""
        return 0 if self.source is None else self.source.stat().st_size


def run_rel_dir(axis: str, public_name: str) -> Path:
    """Internal run/cell directory, relative to --results-root."""
    if axis == "effort":
        return Path(INTERNAL_RUN_BY_PUBLIC_NAME[public_name])
    return Path(PROMPT_SENSITIVITY_DIRNAME) / public_name


def plan_ideas(dataset_dir: Path, axis: str, public_name: str) -> list[PlannedFile]:
    """Plan the run-level ``samples.jsonl`` from ``ideas/<axis-dir>/<public>.jsonl.gz``."""
    source = dataset_dir / "ideas" / axis_dir(axis) / f"{public_name}.jsonl.gz"
    if not source.is_file():
        return []
    return [
        PlannedFile(
            dest_rel=run_rel_dir(axis, public_name) / "samples.jsonl",
            kind="gunzip",
            category="ideas",
            source=source,
        )
    ]


def plan_generation_errors(dataset_dir: Path, axis: str, public_name: str) -> list[PlannedFile]:
    """Plan the run-level ``errors.jsonl`` from ``errors/<axis-dir>/<public>.errors.jsonl.gz``."""
    source = dataset_dir / "errors" / axis_dir(axis) / f"{public_name}.errors.jsonl.gz"
    if not source.is_file():
        return []
    return [
        PlannedFile(
            dest_rel=run_rel_dir(axis, public_name) / "errors.jsonl",
            kind="gunzip",
            category="errors",
            source=source,
        )
    ]


def _plan_judge_family(
    dataset_dir: Path,
    axis: str,
    public_name: str,
    run_dir: Path,
    family: str,
    prefix: str,
    record_name: str,
    errors_name: str,
) -> list[PlannedFile]:
    """Plan one judge family (``diversity`` or ``quality``) for a run/cell.

    Discovers judges by globbing the public filenames rather than by listing
    the (nonexistent, pre-materialization) internal judge directories --
    the inverse of how ``package_hf_dataset.py`` discovers them by globbing
    ``llm_judge_*`` on the internal tree.
    """
    in_dir = dataset_dir / "judge" / family / axis_dir(axis)
    file_prefix = f"{public_name}__judge-"
    planned: list[PlannedFile] = []
    for path in sorted(in_dir.glob(f"{file_prefix}*.jsonl.gz")):
        name = path.name[len(file_prefix) :]
        if name.endswith(".errors.jsonl.gz"):
            judge = name[: -len(".errors.jsonl.gz")]
            record_filename = errors_name
        else:
            judge = name[: -len(".jsonl.gz")]
            record_filename = record_name
        planned.append(
            PlannedFile(
                dest_rel=run_dir / f"{prefix}{judge}" / record_filename,
                kind="gunzip",
                category="judge",
                source=path,
            )
        )
    return planned


def plan_judges(dataset_dir: Path, axis: str, public_name: str) -> list[PlannedFile]:
    """Plan both judge families for a run/cell."""
    run_dir = run_rel_dir(axis, public_name)
    planned = _plan_judge_family(
        dataset_dir,
        axis,
        public_name,
        run_dir,
        "diversity",
        DIVERSITY_JUDGE_PREFIX,
        "pairwise.jsonl",
        "pairwise.errors.jsonl",
    )
    planned += _plan_judge_family(
        dataset_dir,
        axis,
        public_name,
        run_dir,
        "quality",
        QUALITY_JUDGE_PREFIX,
        "quality.jsonl",
        "quality.errors.jsonl",
    )
    return planned


def plan_embeddings(dataset_dir: Path, axis: str, public_name: str) -> list[PlannedFile]:
    """Plan every ``embeddings*.npy`` plus a row-aligned ``samples.jsonl``, per embedder."""
    run_dir = run_rel_dir(axis, public_name)
    planned: list[PlannedFile] = []
    for embedder in EMBEDDER_DIRS:
        embedder_in_dir = dataset_dir / "embeddings" / axis_dir(axis) / public_name / embedder
        if not embedder_in_dir.is_dir():
            continue
        embedder_dest_dir = run_dir / embedder
        npy_files = sorted(embedder_in_dir.glob("*.npy"))
        for npy in npy_files:
            planned.append(
                PlannedFile(
                    dest_rel=embedder_dest_dir / npy.name,
                    kind="npy-copy",
                    category="embeddings",
                    source=npy,
                )
            )
        if (embedder_in_dir / "embeddings.npy").is_file():
            row_keys = embedder_in_dir / ROW_KEYS_PUBLIC_NAME
            planned.append(
                PlannedFile(
                    dest_rel=embedder_dest_dir,
                    kind="embedder-samples",
                    category="embedder-samples",
                    source=row_keys if row_keys.is_file() else None,
                    run_samples_rel=run_dir / "samples.jsonl",
                    npy=embedder_in_dir / "embeddings.npy",
                )
            )
    return planned


def plan_run(dataset_dir: Path, axis: str, public_name: str) -> list[PlannedFile]:
    """Plan every internal artifact derived from one run's/cell's public files."""
    return (
        plan_ideas(dataset_dir, axis, public_name)
        + plan_judges(dataset_dir, axis, public_name)
        + plan_generation_errors(dataset_dir, axis, public_name)
        + plan_embeddings(dataset_dir, axis, public_name)
    )


def discover_prompt_cells(dataset_dir: Path) -> list[str]:
    """Return the public cell names present under ``ideas/prompt-axis/``."""
    in_dir = dataset_dir / "ideas" / axis_dir("prompt")
    if not in_dir.is_dir():
        return []
    return sorted(p.name[: -len(".jsonl.gz")] for p in in_dir.glob("*.jsonl.gz"))


def build_plan(dataset_dir: Path) -> list[PlannedFile]:
    """Plan the full materialization from a downloaded HF dataset tree."""
    planned: list[PlannedFile] = []
    for public_name in EFFORT_RUN_PUBLIC_NAME.values():
        planned += plan_run(dataset_dir, "effort", public_name)
    for cell_name in discover_prompt_cells(dataset_dir):
        planned += plan_run(dataset_dir, "prompt", cell_name)
    return planned


def write_gunzip(source: Path, dest: Path) -> None:
    """Gunzip a ``.jsonl.gz`` file byte for byte."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rb") as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out, COPY_BUFFER_BYTES)


def write_npy_copy(source: Path, dest: Path) -> None:
    """Copy an embedding array as-is (already float32 in the public dataset)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def _canonical_sort_key(key: tuple) -> tuple:
    """``src.artifacts.sort_sample_records`` order, on a ``row_key_from_payload`` tuple."""
    category, keyword, prompt_style, effort, sample_index = key
    return (category, keyword, prompt_style, effort_sort_key(effort), sample_index)


class RunLines:
    """Run-level ``samples.jsonl`` lines indexed by sample key (loaded once per run)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.by_key: dict[tuple, bytes] = {}
        with path.open("rb") as f:
            for line in f:
                if not line.strip():
                    continue
                key = row_key_from_payload(json.loads(line))
                if key in self.by_key:
                    raise ValueError(f"duplicate sample key {key} in {path}")
                self.by_key[key] = line if line.endswith(b"\n") else line + b"\n"


def write_embedder_samples(item: PlannedFile, results_root: Path, cache: dict[Path, RunLines]) -> str:
    """Write ``<embedder>/samples.jsonl`` and ``embedding_row_keys.jsonl`` in array row order."""
    dest_dir = results_root / item.dest_rel
    run_samples = results_root / item.run_samples_rel
    if run_samples not in cache:
        cache.clear()  # one run at a time keeps memory bounded
        cache[run_samples] = RunLines(run_samples)
    run_lines = cache[run_samples]
    n_rows = np.load(item.npy, mmap_mode="r").shape[0]

    if item.source is not None:
        with gzip.open(item.source, "rb") as f:
            keys = [row_key_from_payload(json.loads(line)) for line in f if line.strip()]
        how = "row_keys"
    else:
        keys = sorted(run_lines.by_key, key=_canonical_sort_key)
        how = "canonical order"
    if len(keys) != n_rows:
        raise ValueError(
            f"{item.npy}: {n_rows} rows but {len(keys)} keys ({how}); "
            f"cannot write a row-aligned {dest_dir / 'samples.jsonl'}"
        )
    missing = [k for k in keys if k not in run_lines.by_key]
    if missing:
        raise ValueError(f"{len(missing)} row keys of {item.npy} are not in {run_samples}, e.g. {missing[0]}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    samples_dest = dest_dir / "samples.jsonl"
    if samples_dest.is_symlink():  # left behind by an older materializer
        samples_dest.unlink()
    with samples_dest.open("wb") as out:
        for key in keys:
            out.write(run_lines.by_key[key])
    with (dest_dir / EMBEDDING_ROW_KEYS_FILENAME).open("w", encoding="utf-8") as out:
        for key in keys:
            out.write(json.dumps(dict(zip(ROW_KEY_FIELDS, key)), ensure_ascii=False) + "\n")
    return f"{n_rows} rows ({how})"


def execute(planned: list[PlannedFile], results_root: Path) -> list[str]:
    """Write every planned file into ``results_root``.

    Returns one message per embedder directory whose ``samples.jsonl`` could not
    be aligned with its arrays; those directories get no ``samples.jsonl``, so
    scripts that need it fail loudly instead of reading misaligned rows.
    """
    cache: dict[Path, RunLines] = {}
    failures: list[str] = []
    for item in planned:
        dest = results_root / item.dest_rel
        if item.kind == "gunzip":
            write_gunzip(item.source, dest)
        elif item.kind == "npy-copy":
            write_npy_copy(item.source, dest)
        else:
            try:
                print(f"  {item.dest_rel}/samples.jsonl: {write_embedder_samples(item, results_root, cache)}")
            except ValueError as exc:
                stale = dest / "samples.jsonl"
                if stale.is_symlink() or stale.exists():
                    stale.unlink()
                failures.append(str(exc))
                print(f"  WARNING {item.dest_rel}: {exc}")
    return failures


def print_plan(planned: list[PlannedFile]) -> None:
    """Print the dry-run plan: source and destination for every file."""
    print("Planned files (dry run — nothing written):")
    for item in planned:
        if item.kind == "embedder-samples":
            print(f"  {'<re-ordered>':>14}  {item.run_samples_rel} by {item.source or 'canonical order'}")
            print(f"  {'':>14}    -> {item.dest_rel / 'samples.jsonl'}")
            continue
        else:
            print(f"  {item.source_bytes:>14,d} B  {item.source}")
        print(f"  {'':>14}    -> {item.dest_rel}")


def print_manifest(planned: list[PlannedFile], results_root: Path, dry_run: bool) -> None:
    """Print the output manifest with per-file sizes and per-category totals."""
    size_label = "source bytes (output not written)" if dry_run else "output bytes"
    print(f"\nManifest ({size_label}):")

    counts: dict[str, int] = {c: 0 for c in CATEGORIES}
    totals: dict[str, int] = {c: 0 for c in CATEGORIES}

    for item in planned:
        dest = results_root / item.dest_rel
        if item.kind == "embedder-samples":
            dest = dest / "samples.jsonl"
            size = 0 if dry_run or not dest.exists() else dest.stat().st_size
        else:
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
        "--dataset-dir",
        type=Path,
        required=True,
        help="Downloaded Hugging Face dataset tree, in the public layout tools/package_hf_dataset.py produces.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Internal results/effort_diversity tree to (re)create (default: %(default)s, relative to CWD).",
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
        help="Actually materialize the internal results tree.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir: Path = args.dataset_dir
    results_root: Path = args.results_root
    dry_run: bool = args.dry_run

    print(f"dataset-dir:  {dataset_dir}")
    print(f"results-root: {results_root}")
    print(f"mode:         {'dry run' if dry_run else 'write'}\n")

    planned = build_plan(dataset_dir)

    failures: list[str] = []
    if dry_run:
        print_plan(planned)
    else:
        failures = execute(planned, results_root)

    print_manifest(planned, results_root, dry_run)
    if failures:
        print(f"\n{len(failures)} embedder directories have no row-aligned samples.jsonl "
              "(is row_keys.jsonl.gz missing from this dataset copy?):", file=sys.stderr)
        for message in failures:
            print(f"  {message}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
