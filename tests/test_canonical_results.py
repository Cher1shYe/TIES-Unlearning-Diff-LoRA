import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.results import (
    FINAL_EVALUATION_KEYS,
    attach_final_metrics,
    validate_final_metric_schema,
)


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

    def test_standard_and_staged_documents_share_one_final_metrics_location(self):
        final = self._metrics()

        standard = attach_final_metrics({"method": "standard_lora"}, final)
        staged = attach_final_metrics({"method": "full_sr", "phase3": final}, final)

        self.assertEqual(standard["final"], staged["final"])
        self.assertEqual(tuple(FINAL_EVALUATION_KEYS), tuple(standard["final"]))


if __name__ == "__main__":
    unittest.main()
