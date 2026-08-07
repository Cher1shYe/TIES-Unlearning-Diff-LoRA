import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.data import deterministic_cap_records, stable_record_id


HANS_ROWS = [
    {
        "pairID": f"{label[:1]}-{heuristic[:2]}-{index}",
        "gold_label": label,
        "heuristic": heuristic,
        "subcase": f"{heuristic}-case",
        "sentence1": f"premise-{label}-{heuristic}-{index}",
        "sentence2": f"hypothesis-{label}-{heuristic}-{index}",
    }
    for label in ("entailment", "non-entailment")
    for heuristic in ("lexical_overlap", "subsequence", "constituent")
    for index in range(4)
]


class DeterministicEvaluationSelectionTest(unittest.TestCase):
    def test_hans_cap_is_order_independent_and_covers_label_heuristic_strata(self):
        selected_a, ids_a = deterministic_cap_records(
            HANS_ROWS, 12, 42, ("gold_label", "heuristic", "subcase")
        )
        selected_b, ids_b = deterministic_cap_records(
            list(reversed(HANS_ROWS)), 12, 42, ("gold_label", "heuristic", "subcase")
        )

        self.assertEqual(ids_a, ids_b)
        self.assertEqual(len(selected_a), 12)
        self.assertEqual({row["gold_label"] for row in selected_a}, {"entailment", "non-entailment"})
        self.assertEqual(
            {row["heuristic"] for row in selected_a},
            {"lexical_overlap", "subsequence", "constituent"},
        )

    def test_preferred_source_id_beats_record_field_order(self):
        first = {"pairID": "row-7", "hypothesis": "h", "premise": "p"}
        second = {"premise": "p", "hypothesis": "h", "pairID": "row-7"}

        self.assertEqual(stable_record_id(first, ("pairID",)), "row-7")
        self.assertEqual(stable_record_id(first, ("pairID",)), stable_record_id(second, ("pairID",)))

    def test_unstratified_cap_has_seeded_order_and_returns_copied_rows(self):
        rows = [{"idx": index, "premise": f"p-{index}"} for index in range(10)]

        selected, ids = deterministic_cap_records(rows, 4, 42)

        self.assertEqual(ids, [stable_record_id(row) for row in selected])
        self.assertEqual(selected, deterministic_cap_records(list(reversed(rows)), 4, 42)[0])
        self.assertTrue(all(selected_row is not source_row for selected_row in selected for source_row in rows if selected_row == source_row))


if __name__ == "__main__":
    unittest.main()
