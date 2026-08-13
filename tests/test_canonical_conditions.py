import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import TrainConfig
from canonical.conditions import (
    BASE_CONDITION_ORDER,
    CANONICAL_TRAINING_SEEDS,
    CONDITIONS,
    rotated_condition_order,
)


class CanonicalConditionContractTest(unittest.TestCase):
    def test_legacy_seed_alias_changes_training_seed_only(self):
        cfg = TrainConfig(output_dir=".")

        cfg.seed = 123

        self.assertEqual(42, cfg.data_seed)
        self.assertEqual(42, cfg.hans_split_seed)
        self.assertEqual(123, cfg.training_seed)
        self.assertEqual(123, cfg.seed)

    def test_conditions_match_frozen_factor_matrix(self):
        got = {
            tag: (condition.subtraction, condition.weighting)
            for tag, condition in CONDITIONS.items()
        }

        self.assertEqual(
            {
                "standard_lora": (False, "none"),
                "full_sr": (True, "n_guided"),
                "subtraction_only": (True, "none"),
                "reweight_only": (False, "n_guided"),
                "staged_neither": (False, "none"),
                "class_prior_reweight": (False, "class_prior"),
            },
            got,
        )

    def test_condition_application_changes_only_declared_factors(self):
        baseline = TrainConfig(output_dir=".")
        frozen = {
            "data_seed": baseline.data_seed,
            "hans_split_seed": baseline.hans_split_seed,
            "training_seed": baseline.training_seed,
            "pos_rank": baseline.pos_rank,
            "neg_rank": baseline.neg_rank,
            "alpha": baseline.alpha,
            "beta": baseline.beta,
            "phase1_epochs": baseline.phase1_epochs,
            "phase2_epochs": baseline.phase2_epochs,
            "phase3_epochs": baseline.phase3_epochs,
        }

        for condition in CONDITIONS.values():
            with self.subTest(condition=condition.tag):
                cfg = condition.apply_to_config(baseline)
                self.assertEqual(not condition.subtraction, cfg.no_ties_ablation)
                self.assertEqual(condition.weighting, cfg.phase3_weighting)
                self.assertEqual(frozen, {name: getattr(cfg, name) for name in frozen})

    def test_seed_order_uses_frozen_left_rotations(self):
        self.assertEqual((42, 123, 2024, 3407, 777), CANONICAL_TRAINING_SEEDS)
        self.assertEqual(
            BASE_CONDITION_ORDER[1:] + BASE_CONDITION_ORDER[:1],
            rotated_condition_order(123),
        )
        self.assertEqual(
            BASE_CONDITION_ORDER[4:] + BASE_CONDITION_ORDER[:4],
            rotated_condition_order(777),
        )

    def test_unknown_training_seed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "frozen canonical training seed"):
            rotated_condition_order(999)


if __name__ == "__main__":
    unittest.main()
