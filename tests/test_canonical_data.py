import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.data import (
    dataset_row_ids,
    sample_dataset,
    split_hans_records,
    validate_hans_disjointness,
)


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def shuffle(self, seed):
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return FakeDataset(rows)

    def select(self, indices):
        return FakeDataset(self.rows[index] for index in indices)


class ReverseRng:
    def permutation(self, size):
        return list(reversed(range(size)))


def hans_record(pair_id, label, heuristic, subcase):
    return {
        "pairID": pair_id,
        "gold_label": label,
        "heuristic": heuristic,
        "subcase": subcase,
        "sentence1": f"premise-{pair_id}",
        "sentence2": f"hypothesis-{pair_id}",
    }


class CanonicalDataContractTest(unittest.TestCase):
    def test_training_seed_cannot_change_fixed_data_ids(self):
        source = FakeDataset({"idx": index} for index in range(20))

        seed_42_ids = dataset_row_ids(sample_dataset(source, 8, seed=42))
        seed_777_ids = dataset_row_ids(sample_dataset(source, 8, seed=42))

        self.assertEqual(seed_42_ids, seed_777_ids)
        self.assertEqual(8, len(seed_42_ids))

    def test_split_is_stratified_sorted_and_resets_rng_per_stratum(self):
        records = []
        records.extend(
            hans_record(f"e-{index}", "entailment", "lexical_overlap", "ln_subject/object_swap")
            for index in [3, 1, 4, 0, 2]
        )
        records.extend(
            hans_record(f"n-{index}", "non-entailment", "subsequence", "sn_NP/S")
            for index in [2, 4, 1, 3, 0]
        )
        records.extend(
            hans_record(f"s-{index}", "entailment", "constituent", "ce_after_since_clause")
            for index in [2, 0, 3, 1]
        )
        rng_calls = []

        def rng_factory(seed):
            rng_calls.append(seed)
            return ReverseRng()

        split = split_hans_records(records, seed=42, rng_factory=rng_factory)

        self.assertEqual(["e-4", "n-4"], split.dev_pair_ids)
        self.assertEqual(
            [
                "s-0", "s-1", "s-2", "s-3",
                "e-3", "e-2", "e-1", "e-0",
                "n-3", "n-2", "n-1", "n-0",
            ],
            split.build_pair_ids,
        )
        self.assertEqual([42, 42], rng_calls)
        self.assertEqual(1, len(split.small_strata))
        self.assertEqual(4, split.small_strata[0]["count"])
        self.assertEqual([], split.small_strata[0]["dev_pair_ids"])
        self.assertFalse(set(split.build_pair_ids) & set(split.dev_pair_ids))

    def test_split_checksum_is_order_independent_for_input_rows(self):
        records = [
            hans_record(f"p-{index}", "entailment", "lexical_overlap", "case")
            for index in range(10)
        ]
        first = split_hans_records(records, seed=42, rng_factory=lambda _seed: ReverseRng())
        second = split_hans_records(list(reversed(records)), seed=42, rng_factory=lambda _seed: ReverseRng())

        self.assertEqual(first.checksum, second.checksum)
        self.assertEqual(first.manifest(), second.manifest())

    def test_split_rejects_duplicate_pair_id(self):
        duplicate = hans_record("same", "entailment", "lexical_overlap", "case")
        with self.assertRaisesRegex(ValueError, "duplicate HANS pair ID"):
            split_hans_records([duplicate, dict(duplicate)], rng_factory=lambda _seed: ReverseRng())

    def test_split_rejects_missing_stratum_field(self):
        record = hans_record("p-1", "entailment", "lexical_overlap", "case")
        del record["subcase"]
        with self.assertRaisesRegex(ValueError, "subcase"):
            split_hans_records([record], rng_factory=lambda _seed: ReverseRng())

    def test_build_dev_and_evaluation_must_be_disjoint(self):
        validate_hans_disjointness(["b-1"], ["d-1"], ["e-1"])

        with self.assertRaisesRegex(ValueError, "build/evaluation"):
            validate_hans_disjointness(["shared"], ["d-1"], ["shared"])
        with self.assertRaisesRegex(ValueError, "dev/evaluation"):
            validate_hans_disjointness(["b-1"], ["shared"], ["shared"])


if __name__ == "__main__":
    unittest.main()
