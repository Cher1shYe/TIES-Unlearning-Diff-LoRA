from hashlib import sha256
import json
import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import canonical.data as canonical_data
from canonical.data import (
    build_hans_content_integrity_manifest,
    dataset_row_ids,
    qualify_hans_pair_id,
    sample_dataset,
    split_hans_records,
    validate_hans_content_integrity,
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
    def test_official_hans_semantic_anchor_values_are_source_controlled(self):
        anchors = canonical_data.HANS_OFFICIAL_ANCHORS_V1
        self.assertEqual("hans_official_semantic_anchors_v1", anchors["schema_version"])
        self.assertEqual(
            "f2d240a1709481a8c37c0721104697469383e9ad49ed22496f9265633c9f129a",
            anchors["split_checksum"],
        )
        self.assertEqual(
            "afa0aea6a159eb3b4f68077da8a665e1c277d47815d01398633af5cfe8e53b51",
            anchors["selection_384"]["selected_source_pair_ids_sha256"],
        )
        self.assertEqual(
            "d755522b3f3e492d3543400f5fe07fd2ba354f62e89525f7170e3432cc178b96",
            anchors["selection_384"]["source_to_artifact_mapping_sha256"],
        )

    def test_hans_pair_id_is_qualified_by_physical_source_partition(self):
        self.assertEqual("hans_train::ex0", qualify_hans_pair_id("ex0", "train"))
        self.assertEqual(
            "hans_evaluation::ex0",
            qualify_hans_pair_id("ex0", "evaluation"),
        )

    def test_hans_pair_id_qualification_is_idempotent_and_rejects_wrong_source(self):
        self.assertEqual(
            "hans_train::ex0",
            qualify_hans_pair_id("hans_train::ex0", "train"),
        )
        with self.assertRaisesRegex(ValueError, "physical source partition"):
            qualify_hans_pair_id("hans_evaluation::ex0", "train")

    def test_hans_pair_id_rejects_noncanonical_official_suffix_grammar(self):
        invalid_raw = (
            "",
            "row-1",
            "ex",
            "ex00",
            "ex01",
            "ex+1",
            "ex-1",
            "ex1.0",
            "ex1::extra",
        )
        for value in invalid_raw:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "canonical source-local"):
                    qualify_hans_pair_id(value, "train")
        for value in (
            "hans_train::hans_train::ex1",
            "hans_evaluation::hans_train::ex1",
            "hans_train::ex01",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    qualify_hans_pair_id(value, "train")

    def test_hans_content_integrity_manifest_binds_ordered_ids_without_pair_id_content(self):
        build = hans_record("ex0", "entailment", "lexical_overlap", "build-case")
        build["canonical_pair_id"] = "hans_train::ex0"
        dev = hans_record("ex1", "non-entailment", "subsequence", "dev-case")
        dev["canonical_pair_id"] = "hans_train::ex1"
        evaluation = hans_record(
            "ex0", "non-entailment", "constituent", "evaluation-case"
        )
        evaluation["canonical_pair_id"] = "hans_evaluation::ex0"
        records = {"build": [build], "dev": [dev], "evaluation": [evaluation]}
        ids = {
            "build": ["hans_train::ex0"],
            "dev": ["hans_train::ex1"],
            "evaluation": ["hans_evaluation::ex0"],
        }

        evidence = build_hans_content_integrity_manifest(records, ids)

        self.assertEqual("hans_content_integrity_v1", evidence["schema_version"])
        self.assertEqual(
            ["gold_label", "premise", "hypothesis", "heuristic", "subcase"],
            evidence["fields"],
        )
        self.assertTrue(evidence["excludes_pair_id"])
        self.assertEqual(
            {"build_dev": 0, "build_evaluation": 0, "dev_evaluation": 0},
            evidence["overlap_counts"],
        )
        build_entry = evidence["partitions"]["build"]
        self.assertEqual(1, build_entry["count"])
        self.assertEqual(0, build_entry["duplicate_content_count"])
        self.assertEqual(1, len(build_entry["content_sha256"]))
        expected_joint = sha256(
            json.dumps(
                [["hans_train::ex0", build_entry["content_sha256"][0]]],
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(expected_joint, build_entry["source_id_content_joint_checksum"])

        changed_pair_ids = {
            name: [dict(row, pairID="ex999") for row in rows]
            for name, rows in records.items()
        }
        changed_evidence = build_hans_content_integrity_manifest(changed_pair_ids, ids)
        self.assertEqual(
            evidence["partitions"]["build"]["content_sha256"],
            changed_evidence["partitions"]["build"]["content_sha256"],
        )

    def test_hans_content_manifest_rejects_record_identity_not_equal_to_parallel_id(self):
        build = hans_record("ex0", "entailment", "lexical_overlap", "build-case")
        build["canonical_pair_id"] = "hans_train::ex0"
        dev = hans_record("ex1", "non-entailment", "subsequence", "dev-case")
        dev["canonical_pair_id"] = "hans_train::ex1"
        evaluation = hans_record("ex0", "non-entailment", "constituent", "eval-case")
        evaluation["canonical_pair_id"] = "hans_evaluation::ex0"

        with self.assertRaisesRegex(ValueError, "canonical_pair_id.*parallel"):
            build_hans_content_integrity_manifest(
                {"build": [build], "dev": [dev], "evaluation": [evaluation]},
                {
                    "build": ["hans_train::ex99"],
                    "dev": ["hans_train::ex1"],
                    "evaluation": ["hans_evaluation::ex0"],
                },
            )

    def test_split_integrity_uses_only_raw_ids_including_small_strata(self):
        records = []
        for index in range(4):
            record = hans_record(
                f"ex{index}", "entailment", "constituent", "small-case"
            )
            record["canonical_pair_id"] = f"hans_train::ex{index}"
            records.append(record)

        split = split_hans_records(records, seed=42)
        self.assertEqual(
            [f"ex{index}" for index in range(4)],
            split.small_strata[0]["build_pair_ids"],
        )
        integrity = canonical_data.build_hans_split_integrity(split)

        self.assertEqual([f"ex{index}" for index in range(4)], integrity["build_source_pair_ids"])
        self.assertEqual([], integrity["dev_source_pair_ids"])
        self.assertEqual(
            [f"ex{index}" for index in range(4)],
            integrity["small_strata"][0]["build_source_pair_ids"],
        )
        self.assertNotIn("hans_train::", json.dumps(integrity["small_strata"]))

    def test_selection_integrity_binds_raw_ranking_to_parallel_qualified_ids(self):
        selected = [
            {"pairID": "ex9", "canonical_pair_id": "hans_evaluation::ex9"},
            {"pairID": "ex2", "canonical_pair_id": "hans_evaluation::ex2"},
        ]

        integrity = canonical_data.build_hans_selection_integrity(
            selected,
            ["hans_evaluation::ex9", "hans_evaluation::ex2"],
            limit=2,
            seed=42,
        )

        self.assertEqual("source_local_pair_id", integrity["ranking_key"])
        self.assertEqual(["ex9", "ex2"], integrity["selected_source_pair_ids"])
        self.assertEqual(2, integrity["cap"])
        self.assertEqual(2, integrity["selected_count"])

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
        validate_hans_disjointness(
            ["hans_train::b-1"],
            ["hans_train::d-1"],
            ["hans_evaluation::e-1"],
        )

        with self.assertRaisesRegex(ValueError, "build/evaluation"):
            validate_hans_disjointness(["shared"], ["d-1"], ["shared"])
        with self.assertRaisesRegex(ValueError, "dev/evaluation"):
            validate_hans_disjointness(["b-1"], ["shared"], ["shared"])

    def test_build_and_dev_reject_same_physical_train_identity(self):
        with self.assertRaisesRegex(ValueError, "build/dev"):
            validate_hans_disjointness(
                ["hans_train::ex0"],
                ["hans_train::ex0"],
                ["hans_evaluation::ex0"],
            )

    def test_hans_content_overlap_ignores_source_qualified_pair_id(self):
        train = hans_record(
            "hans_train::ex0", "entailment", "lexical_overlap", "case"
        )
        evaluation = dict(train, pairID="hans_evaluation::ex0")

        with self.assertRaisesRegex(ValueError, "content.*build/evaluation"):
            validate_hans_content_integrity([train], [], [evaluation])

    def test_same_local_id_with_different_content_in_official_files_passes(self):
        train = hans_record(
            "hans_train::ex0", "entailment", "lexical_overlap", "case"
        )
        evaluation = hans_record(
            "hans_evaluation::ex0", "non-entailment", "subsequence", "other-case"
        )

        validate_hans_content_integrity([train], [], [evaluation])

    def test_duplicate_hans_content_within_partition_is_rejected(self):
        first = hans_record(
            "hans_train::ex0", "entailment", "lexical_overlap", "case"
        )
        duplicate = dict(first, pairID="hans_train::ex1")

        with self.assertRaisesRegex(ValueError, "duplicate HANS build content"):
            validate_hans_content_integrity([first, duplicate], [], [])


if __name__ == "__main__":
    unittest.main()
