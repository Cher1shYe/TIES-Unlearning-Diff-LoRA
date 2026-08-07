"""Shared final-evaluation schema for every canonical method."""

from typing import Any, Mapping


FINAL_EVALUATION_KEYS = (
    "mnli",
    "hans",
    "esnli",
    "anli",
    "snli_hard",
    "wanli",
)

_REQUIRED_FIELDS = {
    "mnli": ("mnli_accuracy",),
    "hans": (
        "hans_overall",
        "hans_entailment",
        "hans_non_entailment",
        "heuristic_breakdown",
        "subcase_breakdown",
    ),
    "esnli": ("esnli_accuracy",),
    "anli": ("anli_accuracy",),
    "snli_hard": ("snli_hard_accuracy",),
    "wanli": ("wanli_accuracy",),
}


def validate_final_metric_schema(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Reject method-specific drift in the frozen final evaluation battery."""
    missing_groups = [key for key in FINAL_EVALUATION_KEYS if key not in metrics]
    if missing_groups:
        raise ValueError(f"canonical final metrics are missing {missing_groups[0]!r}")
    normalized = {}
    for group in FINAL_EVALUATION_KEYS:
        payload = metrics[group]
        if not isinstance(payload, Mapping):
            raise ValueError(f"canonical final metric group {group!r} must be an object")
        missing_fields = [field for field in _REQUIRED_FIELDS[group] if field not in payload]
        if missing_fields:
            raise ValueError(
                f"canonical final metric group {group!r} is missing {missing_fields[0]!r}"
            )
        normalized[group] = dict(payload)
    return normalized
