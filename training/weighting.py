"""Phase-3 weighting contracts shared by canonical conditions."""

import math
from numbers import Integral, Real
from typing import Iterable, Mapping, Sequence


VALID_WEIGHTING_MODES = {"none", "n_guided", "class_prior"}


def resolve_weighting_mode(cfg) -> str:
    explicit = getattr(cfg, "phase3_weighting", None)
    if explicit is None:
        return "n_guided" if getattr(cfg, "phase3_debias_reweight", False) else "none"
    if explicit not in VALID_WEIGHTING_MODES:
        raise ValueError(f"unknown Phase-3 weighting mode: {explicit!r}")
    return explicit


def normalize_batch_weights(raw_weights: Iterable[Real]) -> list[float]:
    weights = [float(value) for value in raw_weights]
    if not weights:
        raise ValueError("batch weights must not be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("batch weights must be finite and non-negative")
    mean = sum(weights) / len(weights)
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError("batch weights must have a positive finite mean")
    return [value / mean for value in weights]


def compute_class_priors(
    labels: Sequence[Integral],
    gold_probabilities: Sequence[Real],
    *,
    gamma: float,
    classes: Sequence[int],
) -> dict[int, float]:
    """Compute a_c = mean((1-p_N(y|x))^gamma | y=c)."""
    if len(labels) != len(gold_probabilities):
        raise ValueError("labels and gold probabilities must have the same length")
    if not labels:
        raise ValueError("class-prior estimation requires training examples")
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be a positive finite number")
    declared = tuple(int(value) for value in classes)
    if len(set(declared)) != len(declared):
        raise ValueError("classes must not contain duplicates")
    buckets = {label: [] for label in declared}
    for index, (label_value, probability_value) in enumerate(zip(labels, gold_probabilities)):
        if not isinstance(label_value, Integral) or isinstance(label_value, bool):
            raise ValueError(f"label at index {index} is not an integer")
        label = int(label_value)
        if label not in buckets:
            raise ValueError(f"label at index {index} is outside declared classes")
        if not isinstance(probability_value, Real) or isinstance(probability_value, bool):
            raise ValueError(f"gold probability at index {index} is invalid")
        probability = float(probability_value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"gold probability at index {index} is invalid")
        buckets[label].append((1.0 - probability) ** gamma)
    missing = [label for label, values in buckets.items() if not values]
    if missing:
        raise ValueError(f"missing training examples for classes: {missing}")
    return {label: sum(values) / len(values) for label, values in buckets.items()}


def class_prior_batch_weights(
    labels: Sequence[Integral], class_priors: Mapping[int, Real]
) -> list[float]:
    raw = []
    for index, label_value in enumerate(labels):
        if not isinstance(label_value, Integral) or isinstance(label_value, bool):
            raise ValueError(f"label at index {index} is not an integer")
        label = int(label_value)
        if label not in class_priors:
            raise ValueError(f"class prior is missing label {label}")
        raw.append(float(class_priors[label]))
    return normalize_batch_weights(raw)


def torch_phase3_weights(
    mode: str,
    labels,
    *,
    n_gold_probabilities=None,
    class_priors: Mapping[int, Real] | None = None,
    gamma: float = 2.0,
):
    """Create mean-one weights while keeping PyTorch an optional import."""
    if mode == "none":
        return None
    if mode == "n_guided":
        if n_gold_probabilities is None:
            raise ValueError("n_guided weighting requires N-path gold probabilities")
        raw = (1.0 - n_gold_probabilities).clamp_min(0.0).pow(gamma)
        mean = raw.mean()
        if not bool(mean.isfinite().item()) or float(mean.item()) <= 0.0:
            raise ValueError("N-guided batch weights must have a positive finite mean")
        return raw / mean
    if mode == "class_prior":
        if class_priors is None:
            raise ValueError("class_prior weighting requires saved class priors")
        import torch

        normalized = class_prior_batch_weights(labels.detach().cpu().tolist(), class_priors)
        return torch.tensor(normalized, dtype=torch.float32, device=labels.device)
    raise ValueError(f"unknown Phase-3 weighting mode: {mode!r}")
