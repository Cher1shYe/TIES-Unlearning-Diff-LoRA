"""Validation and metric recomputation for canonical HANS predictions."""

import math
from numbers import Integral, Real
from typing import Any, Iterable, Mapping


REQUIRED_PREDICTION_FIELDS = (
    "pair_id",
    "gold_label",
    "predicted_label",
    "entailment_probability",
    "heuristic",
    "subcase",
    "training_seed",
    "method_tag",
    "checkpoint_hash",
)
LABELS = {"entailment", "non-entailment"}
HEURISTICS = {"lexical_overlap", "subsequence", "constituent"}


def validate_hans_prediction(record: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_PREDICTION_FIELDS if field not in record]
    if missing:
        raise ValueError(f"HANS prediction is missing required field {missing[0]!r}")

    normalized = dict(record)
    for field in ("pair_id", "gold_label", "predicted_label", "heuristic", "subcase", "method_tag", "checkpoint_hash"):
        if not isinstance(normalized[field], str) or not normalized[field]:
            raise ValueError(f"HANS prediction field {field!r} must be a non-empty string")
    if normalized["gold_label"] not in LABELS:
        raise ValueError("gold_label must be 'entailment' or 'non-entailment'")
    if normalized["predicted_label"] not in LABELS:
        raise ValueError("predicted_label must be 'entailment' or 'non-entailment'")
    if normalized["heuristic"] not in HEURISTICS:
        raise ValueError(f"unsupported HANS heuristic: {normalized['heuristic']!r}")

    probability = normalized["entailment_probability"]
    if not isinstance(probability, Real) or isinstance(probability, bool):
        raise ValueError("entailment_probability must be a finite float in [0, 1]")
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("entailment_probability must be a finite float in [0, 1]")
    normalized["entailment_probability"] = probability

    seed = normalized["training_seed"]
    if not isinstance(seed, Integral) or isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("training_seed must be a non-negative integer")
    normalized["training_seed"] = int(seed)

    checkpoint_hash = normalized["checkpoint_hash"]
    if len(checkpoint_hash) != 64 or any(char not in "0123456789abcdef" for char in checkpoint_hash.lower()):
        raise ValueError("checkpoint_hash must be a 64-character SHA-256 hex digest")
    normalized["checkpoint_hash"] = checkpoint_hash.lower()
    return normalized


def _accuracy(records: list[dict[str, Any]]) -> float:
    return sum(row["gold_label"] == row["predicted_label"] for row in records) / len(records)


def _breakdown(records: list[dict[str, Any]], field: str) -> dict[str, float]:
    result = {}
    for value in sorted({row[field] for row in records}):
        group = [row for row in records if row[field] == value]
        result[value] = _accuracy(group)
    return result


def aggregate_hans_predictions(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [validate_hans_prediction(record) for record in records]
    if not normalized:
        raise ValueError("HANS aggregation requires at least one prediction")
    pair_ids = [row["pair_id"] for row in normalized]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("duplicate HANS prediction pair_id")

    entailment = [row for row in normalized if row["gold_label"] == "entailment"]
    non_entailment = [row for row in normalized if row["gold_label"] == "non-entailment"]
    if not entailment or not non_entailment:
        raise ValueError("HANS predictions must include both gold-label groups")

    return {
        "n_examples": len(normalized),
        "hans_overall": _accuracy(normalized),
        "hans_entailment": _accuracy(entailment),
        "hans_non_entailment": _accuracy(non_entailment),
        "heuristic_breakdown": _breakdown(normalized, "heuristic"),
        "subcase_breakdown": _breakdown(normalized, "subcase"),
    }
