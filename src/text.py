"""Text normalization helpers shared by diversity modules."""

from __future__ import annotations

from .schemas.sample import SampleRecord

TEXT_COMPONENTS: list[str] = ["combined", "background", "idea"]


def slugify(text: str) -> str:
    """Convert *text* to a filename-safe slug (lowercase alphanumeric + hyphens)."""
    out: list[str] = []
    for char in text.lower():
        if char.isalnum():
            out.append(char)
        elif char in {" ", "-", "_"}:
            out.append("-")
    slug = "".join(out).strip("-")
    return slug or "keyword"


def clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").replace("\r", " ").replace("\t", " ").split())


def normalize_embedding_text(text: str) -> str:
    sanitized = text.replace("\x00", " ")
    return sanitized.encode("utf-8", "replace").decode("utf-8")


def ensure_nonempty_text(text: str) -> str:
    normalized = clean_text(text)
    if not normalized:
        raise ValueError("Text is empty after normalization")
    return normalized


def extract_record_text(record: SampleRecord, component: str) -> str:
    if component == "combined":
        return ensure_nonempty_text(record.idea)
    return ensure_nonempty_text(record.structured_output.text_for_component(component))


def extract_record_texts(records: list[SampleRecord], component: str) -> list[str]:
    return [extract_record_text(record, component) for record in records]


def assemble_specter2_text(record: SampleRecord, *, sep_token: str, is_adhoc: bool) -> str:
    """Assemble a SPECTER2 embedding input from a sample record.

    Adhoc adapters take the schema's full combined text as a single free-form
    query; proximity adapters take the two ``specter2_parts()`` parts joined
    by the tokenizer's sep_token (title[SEP]abstract-style input).
    """
    if is_adhoc:
        return clean_text(record.structured_output.combined_text())
    part_a, part_b = record.structured_output.specter2_parts()
    if not part_b:
        raise ValueError("structured_output text part is empty after normalization")
    return f"{part_a}{sep_token}{part_b}" if part_a else part_b


def assemble_specter2_texts(
    records: list[SampleRecord], *, sep_token: str, is_adhoc: bool
) -> list[str]:
    """Assemble SPECTER2 embedding inputs for a list of records."""
    return [
        assemble_specter2_text(record, sep_token=sep_token, is_adhoc=is_adhoc)
        for record in records
    ]
