import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.evaluation_policy import hans_split_for_event
from training.weighting import (
    class_prior_batch_weights,
    compute_class_priors,
    normalize_batch_weights,
    resolve_weighting_mode,
)


class CanonicalWeightingContractTest(unittest.TestCase):
    def test_class_priors_use_frozen_training_formula(self):
        labels = [0, 0, 1, 1, 2, 2]
        gold_probabilities = [0.5, 1.0, 0.0, 0.5, 0.25, 0.75]

        priors = compute_class_priors(
            labels, gold_probabilities, gamma=2.0, classes=(0, 1, 2)
        )

        self.assertEqual({0: 0.125, 1: 0.625, 2: 0.3125}, priors)

    def test_class_prior_batch_weights_are_normalized_to_mean_one(self):
        priors = {0: 0.125, 1: 0.625, 2: 0.3125}

        weights = class_prior_batch_weights([2, 0, 1], priors)

        self.assertTrue(math.isclose(1.0, sum(weights) / len(weights)))
        self.assertTrue(math.isclose(0.3125 / (1.0625 / 3), weights[0]))
        self.assertTrue(math.isclose(0.125 / (1.0625 / 3), weights[1]))
        self.assertTrue(math.isclose(0.625 / (1.0625 / 3), weights[2]))

    def test_normalization_rejects_zero_mean(self):
        with self.assertRaisesRegex(ValueError, "positive finite mean"):
            normalize_batch_weights([0.0, 0.0])

    def test_class_prior_inputs_are_validated(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            compute_class_priors([0], [0.1, 0.2], gamma=2.0, classes=(0, 1, 2))
        with self.assertRaisesRegex(ValueError, "probability"):
            compute_class_priors([0, 1, 2], [0.2, float("nan"), 0.4], gamma=2.0, classes=(0, 1, 2))
        with self.assertRaisesRegex(ValueError, "missing training examples"):
            compute_class_priors([0, 1], [0.2, 0.4], gamma=2.0, classes=(0, 1, 2))

    def test_explicit_and_legacy_weighting_modes_resolve_consistently(self):
        self.assertEqual(
            "class_prior",
            resolve_weighting_mode(SimpleNamespace(phase3_weighting="class_prior", phase3_debias_reweight=True)),
        )
        self.assertEqual(
            "n_guided",
            resolve_weighting_mode(SimpleNamespace(phase3_weighting=None, phase3_debias_reweight=True)),
        )
        self.assertEqual(
            "none",
            resolve_weighting_mode(SimpleNamespace(phase3_weighting=None, phase3_debias_reweight=False)),
        )

    def test_only_final_event_can_request_official_evaluation(self):
        for event in ("phase1_end", "phase2_end", "phase2_5", "phase3_epoch"):
            with self.subTest(event=event):
                self.assertEqual("dev", hans_split_for_event(event))
        self.assertEqual("evaluation", hans_split_for_event("final_evaluation"))

    def test_unknown_evaluation_event_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown HANS evaluation event"):
            hans_split_for_event("checkpoint_selection")


if __name__ == "__main__":
    unittest.main()
