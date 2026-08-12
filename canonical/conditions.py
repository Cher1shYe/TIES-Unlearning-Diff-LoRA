"""The frozen six-condition matrix and run ordering for canonical v1."""

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Mapping

if TYPE_CHECKING:
    from configs.config import TrainConfig


WeightingMode = Literal["none", "n_guided", "class_prior"]

CANONICAL_TRAINING_SEEDS = (42, 123, 2024, 3407, 777)
BASE_CONDITION_ORDER = (
    "standard_lora",
    "full_sr",
    "subtraction_only",
    "reweight_only",
    "staged_neither",
    "class_prior_reweight",
)


@dataclass(frozen=True)
class CanonicalCondition:
    tag: str
    subtraction: bool
    weighting: WeightingMode
    standard_lora: bool = False

    def apply_to_config(self, base: "TrainConfig") -> "TrainConfig":
        """Return an isolated config changing only the condition factors."""
        cfg = deepcopy(base)
        cfg.no_ties_ablation = not self.subtraction
        cfg.phase3_weighting = self.weighting
        cfg.phase3_debias_reweight = self.weighting == "n_guided"
        cfg.experiment_name = self.tag
        return cfg


_CONDITIONS = {
    "standard_lora": CanonicalCondition(
        "standard_lora", subtraction=False, weighting="none", standard_lora=True
    ),
    "full_sr": CanonicalCondition("full_sr", subtraction=True, weighting="n_guided"),
    "subtraction_only": CanonicalCondition(
        "subtraction_only", subtraction=True, weighting="none"
    ),
    "reweight_only": CanonicalCondition(
        "reweight_only", subtraction=False, weighting="n_guided"
    ),
    "staged_neither": CanonicalCondition(
        "staged_neither", subtraction=False, weighting="none"
    ),
    "class_prior_reweight": CanonicalCondition(
        "class_prior_reweight", subtraction=False, weighting="class_prior"
    ),
}

CONDITIONS: Mapping[str, CanonicalCondition] = MappingProxyType(_CONDITIONS)


def rotated_condition_order(training_seed: int) -> tuple[str, ...]:
    """Return the protocol-frozen left rotation for one training seed."""
    try:
        offset = CANONICAL_TRAINING_SEEDS.index(training_seed)
    except ValueError as exc:
        raise ValueError(
            f"{training_seed!r} is not a frozen canonical training seed"
        ) from exc
    return BASE_CONDITION_ORDER[offset:] + BASE_CONDITION_ORDER[:offset]
