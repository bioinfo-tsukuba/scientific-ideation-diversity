"""Pure diversity metric helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .model_registry import EFFORT_ORDER, effort_sort_key
from .schemas.sample import SampleRecord
from .schemas.validation import PydanticValidationStatus


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((0, 0), dtype=float)
    normalized = normalize_rows(np.asarray(matrix, dtype=np.float64))
    similarity = normalized @ normalized.T
    return np.clip(1.0 - similarity, 0.0, 2.0)


def vendi_score(matrix: np.ndarray) -> float:
    """Vendi Score~\\citep{friedmanVendiScoreDiversity2023} under the cosine kernel.

    Returns the effective number of unique elements in the embedding set,
    computed as exp(Shannon entropy of eigenvalues of K/n) where K is the
    cosine similarity matrix. The returned value is in [1, n] for n samples.
    """
    n = int(matrix.shape[0])
    if n < 2:
        return float(n)

    normalized = normalize_rows(np.asarray(matrix, dtype=np.float64))
    similarity = normalized @ normalized.T
    K_over_n = similarity / n  # eigenvalues sum to trace(K)/n = 1

    eigenvalues = np.linalg.eigvalsh(K_over_n)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    nonzero = eigenvalues[eigenvalues > 1e-12]
    if nonzero.size == 0:
        return 1.0
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    return float(np.exp(entropy))


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.empty((0,), dtype=float)
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def population_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.std(ddof=0))


def _mean_int(values: list[int]) -> float:
    arr = np.fromiter(values, dtype=np.float64)
    return float(arr.mean()) if arr.size else 0.0


def _rate(records: list[SampleRecord], predicate: Any) -> float:
    values = np.fromiter((1.0 if predicate(record) else 0.0 for record in records), dtype=np.float64)
    return float(values.mean()) if values.size else 0.0


def _build_keyword_effort_row(
    *,
    category: str,
    keyword: str,
    effort: str,
    effort_records: list[SampleRecord],
    effort_matrix: np.ndarray,
) -> dict[str, Any]:
    distance_matrix = cosine_distance_matrix(effort_matrix)
    pairwise = upper_triangle_values(distance_matrix)
    centroid = effort_matrix.mean(axis=0, keepdims=True)
    centroid_distances = 1.0 - (normalize_rows(effort_matrix) @ normalize_rows(centroid).T).reshape(-1)

    return {
        "category": category,
        "keyword": keyword,
        "prompt_style": effort_records[0].prompt_style,
        "effort": effort,
        "n": len(effort_records),
        "mean_word_count": float(np.mean([record.word_count for record in effort_records])),
        "mean_char_count": float(np.mean([record.char_count for record in effort_records])),
        "reasoning_block_rate": _rate(effort_records, lambda record: record.has_reasoning_block),
        "max_tokens_stop_rate": _rate(
            effort_records,
            lambda record: (record.stop_reason or "").lower() in {"max_tokens", "maxtokens"},
        ),
        "pydantic_validation_pass_rate": _rate(
            effort_records,
            lambda record: record.pydantic_validation_status == PydanticValidationStatus.PASSED,
        ),
        "mean_input_tokens": _mean_int([record.input_tokens for record in effort_records]),
        "mean_output_tokens": _mean_int([record.output_tokens for record in effort_records]),
        "mean_total_tokens": _mean_int([record.total_tokens for record in effort_records]),
        "max_output_tokens": max(record.output_tokens for record in effort_records),
        "mean_pairwise_cosine_distance": float(np.mean(pairwise)) if pairwise.size else 0.0,
        "median_pairwise_cosine_distance": float(np.median(pairwise)) if pairwise.size else 0.0,
        "mean_distance_to_centroid": float(np.mean(centroid_distances)),
        "std_distance_to_centroid": float(np.std(centroid_distances)),
    }


def pca_2d(matrix: np.ndarray) -> tuple[np.ndarray, list[float]]:
    if matrix.shape[0] == 1:
        return np.zeros((1, 2), dtype=float), [1.0, 0.0]

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T

    if singular_values.size == 0:
        return np.zeros((matrix.shape[0], 2), dtype=float), [0.0, 0.0]

    variances = singular_values**2
    explained = variances / variances.sum()
    explained_list = explained[:2].tolist()
    while len(explained_list) < 2:
        explained_list.append(0.0)
    return coords, explained_list


def build_centroid_distance_rows(
    records: list[SampleRecord],
    embeddings: np.ndarray,
) -> tuple[list[str], list[list[Any]]]:
    group_labels = []
    centroids: dict[str, np.ndarray] = {}

    grouped_records = sorted(
        {(record.category, record.keyword, record.effort) for record in records},
        key=lambda item: (item[0] or "", item[1], effort_sort_key(item[2])),
    )

    for category, keyword, effort in grouped_records:
        label = f"{category or 'NA'}|{keyword}|{effort}"
        indices = [
            i
            for i, record in enumerate(records)
            if record.category == category and record.keyword == keyword and record.effort == effort
        ]
        centroids[label] = embeddings[indices].mean(axis=0)
        group_labels.append(label)

    header = ["group"] + group_labels
    rows: list[list[Any]] = []
    normalized = {key: normalize_rows(value.reshape(1, -1))[0] for key, value in centroids.items()}
    for group_a in group_labels:
        row: list[Any] = [group_a]
        for group_b in group_labels:
            distance = 1.0 - float(np.dot(normalized[group_a], normalized[group_b]))
            row.append(distance)
        rows.append(row)
    return header, rows


def build_summary_by_keyword_effort_rows(
    records: list[SampleRecord],
    embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = sorted(
        {(record.category, record.keyword, record.effort) for record in records},
        key=lambda item: (item[0] or "", item[1], effort_sort_key(item[2])),
    )

    for category, keyword, effort in groups:
        indices = [
            i
            for i, record in enumerate(records)
            if record.category == category and record.keyword == keyword and record.effort == effort
        ]
        effort_records = [records[i] for i in indices]
        rows.append(
            _build_keyword_effort_row(
                category=category,
                keyword=keyword,
                effort=effort,
                effort_records=effort_records,
                effort_matrix=embeddings[indices],
            )
        )

    return rows


def build_summary_by_effort_rows(
    keyword_effort_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows_by_effort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in keyword_effort_rows:
        rows_by_effort[row["effort"]].append(row)

    for effort in sorted(rows_by_effort, key=effort_sort_key):
        effort_rows = rows_by_effort[effort]
        rows.append(
            {
                "prompt_style": effort_rows[0]["prompt_style"],
                "effort": effort,
                "n_keywords": len(effort_rows),
                "total_samples": int(sum(row["n"] for row in effort_rows)),
                "mean_word_count": float(np.mean([row["mean_word_count"] for row in effort_rows])),
                "mean_char_count": float(np.mean([row["mean_char_count"] for row in effort_rows])),
                "mean_reasoning_block_rate": float(np.mean([row["reasoning_block_rate"] for row in effort_rows])),
                "mean_max_tokens_stop_rate": float(np.mean([row["max_tokens_stop_rate"] for row in effort_rows])),
                "mean_pydantic_validation_pass_rate": float(
                    np.mean([row["pydantic_validation_pass_rate"] for row in effort_rows])
                ),
                "mean_input_tokens": float(np.mean([row["mean_input_tokens"] for row in effort_rows])),
                "mean_output_tokens": float(np.mean([row["mean_output_tokens"] for row in effort_rows])),
                "mean_total_tokens": float(np.mean([row["mean_total_tokens"] for row in effort_rows])),
                "max_output_tokens": int(max(row["max_output_tokens"] for row in effort_rows)),
                "mean_pairwise_cosine_distance": float(
                    np.mean([row["mean_pairwise_cosine_distance"] for row in effort_rows])
                ),
                "median_pairwise_cosine_distance": float(
                    np.mean([row["median_pairwise_cosine_distance"] for row in effort_rows])
                ),
                "mean_distance_to_centroid": float(np.mean([row["mean_distance_to_centroid"] for row in effort_rows])),
                "mean_std_distance_to_centroid": float(
                    np.mean([row["std_distance_to_centroid"] for row in effort_rows])
                ),
            }
        )

    return rows


def build_category_summary_rows(keyword_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in keyword_summary_rows:
        grouped[(row["category"], row["effort"])].append(row)

    categories = sorted({row["category"] for row in keyword_summary_rows if row.get("category")})
    rows: list[dict[str, Any]] = []
    for category in categories:
        for effort in EFFORT_ORDER:
            items = grouped.get((category, effort), [])
            if not items:
                continue
            pairwise_values = [float(item["mean_pairwise_cosine_distance"]) for item in items]
            centroid_values = [float(item["mean_distance_to_centroid"]) for item in items]
            output_token_values = [float(item["mean_output_tokens"]) for item in items]
            rows.append(
                {
                    "category": category,
                    "effort": effort,
                    "n_keywords": len(items),
                    "mean_pairwise_cosine_distance": mean(pairwise_values),
                    "std_pairwise_cosine_distance": population_std(pairwise_values),
                    "min_pairwise_cosine_distance": min(pairwise_values),
                    "max_pairwise_cosine_distance": max(pairwise_values),
                    "mean_distance_to_centroid": mean(centroid_values),
                    "std_distance_to_centroid": population_std(centroid_values),
                    "mean_output_tokens": mean(output_token_values),
                }
            )
    return rows
