import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.hans import aggregate_hans_predictions, validate_hans_prediction


def prediction(pair_id, gold, predicted, probability, heuristic, subcase):
    return {
        "pair_id": pair_id,
        "gold_label": gold,
        "predicted_label": predicted,
        "entailment_probability": probability,
        "heuristic": heuristic,
        "subcase": subcase,
        "training_seed": 42,
        "method_tag": "full_sr",
        "checkpoint_hash": "a" * 64,
    }


class CanonicalHansPredictionContractTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            prediction("p-1", "entailment", "entailment", 0.80, "lexical_overlap", "case-1"),
            prediction("p-2", "entailment", "non-entailment", 0.20, "lexical_overlap", "case-1"),
            prediction("p-3", "non-entailment", "non-entailment", 0.10, "subsequence", "case-2"),
            prediction("p-4", "non-entailment", "entailment", 0.75, "subsequence", "case-2"),
            prediction("p-5", "non-entailment", "non-entailment", 0.05, "constituent", "case-3"),
            prediction("p-6", "entailment", "entailment", 0.90, "constituent", "case-3"),
        ]

    def test_aggregate_metrics_are_recomputed_from_literal_rows(self):
        metrics = aggregate_hans_predictions(self.records)

        self.assertEqual(6, metrics["n_examples"])
        self.assertTrue(math.isclose(4 / 6, metrics["hans_overall"]))
        self.assertTrue(math.isclose(2 / 3, metrics["hans_entailment"]))
        self.assertTrue(math.isclose(2 / 3, metrics["hans_non_entailment"]))
        self.assertEqual(
            {"constituent": 1.0, "lexical_overlap": 0.5, "subsequence": 0.5},
            metrics["heuristic_breakdown"],
        )
        self.assertEqual(
            {"case-1": 0.5, "case-2": 0.5, "case-3": 1.0},
            metrics["subcase_breakdown"],
        )

    def test_prediction_schema_rejects_missing_field(self):
        malformed = dict(self.records[0])
        del malformed["subcase"]
        with self.assertRaisesRegex(ValueError, "subcase"):
            validate_hans_prediction(malformed)

    def test_prediction_schema_rejects_non_finite_or_out_of_range_probability(self):
        for probability in (float("nan"), float("inf"), -0.1, 1.1):
            malformed = dict(self.records[0], entailment_probability=probability)
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(ValueError, "entailment_probability"):
                    validate_hans_prediction(malformed)

    def test_aggregate_rejects_duplicate_pair_id(self):
        with self.assertRaisesRegex(ValueError, "duplicate HANS prediction pair_id"):
            aggregate_hans_predictions([self.records[0], dict(self.records[0])])

    def test_aggregate_rejects_empty_records(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            aggregate_hans_predictions([])


if __name__ == "__main__":
    unittest.main()
