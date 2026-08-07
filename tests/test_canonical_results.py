import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.results import FINAL_EVALUATION_KEYS, validate_final_metric_schema


class CanonicalResultSchemaContractTest(unittest.TestCase):
    def _metrics(self):
        return {
            "mnli": {"mnli_accuracy": 0.8},
            "hans": {
                "hans_overall": 0.7,
                "hans_entailment": 0.8,
                "hans_non_entailment": 0.6,
                "heuristic_breakdown": {},
                "subcase_breakdown": {},
            },
            "esnli": {"esnli_accuracy": 0.7},
            "anli": {"anli_accuracy": 0.4},
            "snli_hard": {"snli_hard_accuracy": 0.5},
            "wanli": {"wanli_accuracy": None},
        }

    def test_final_schema_requires_the_same_evaluation_battery_for_every_method(self):
        metrics = self._metrics()

        validated = validate_final_metric_schema(metrics)

        self.assertEqual(tuple(FINAL_EVALUATION_KEYS), tuple(validated))
        self.assertIsNone(validated["wanli"]["wanli_accuracy"])

    def test_missing_wanli_is_rejected_instead_of_silently_drifting_schema(self):
        metrics = self._metrics()
        del metrics["wanli"]

        with self.assertRaisesRegex(ValueError, "wanli"):
            validate_final_metric_schema(metrics)


if __name__ == "__main__":
    unittest.main()
