import json
import importlib
import sys
import tempfile
import types
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.access_audit import (
    append_access_event,
    record_dataset_access,
    record_final_evaluation_start,
)
from canonical.artifacts import read_jsonl, sha256_file, write_json, write_jsonl
from canonical.backend import RealCanonicalBackend
from canonical.data import (
    deterministic_cap_records,
    select_hans_evaluation_records,
    stable_record_id,
)
from canonical.data_manifest import dataset_identity_entry
from canonical.runner import CheckpointRef, run_core
from configs.config import TrainConfig


HANS_ROWS = [
    {
        "pairID": f"ex{global_index}",
        "canonical_pair_id": f"hans_evaluation::ex{global_index}",
        "gold_label": label,
        "heuristic": heuristic,
        "subcase": f"{heuristic}-case",
        "sentence1": f"premise-{label}-{heuristic}-{index}",
        "sentence2": f"hypothesis-{label}-{heuristic}-{index}",
    }
    for global_index, (label, heuristic, index) in enumerate(
        (label, heuristic, index)
        for label in ("entailment", "non-entailment")
        for heuristic in ("lexical_overlap", "subsequence", "constituent")
        for index in range(4)
    )
]


MANIFEST_ROWS = [
    {"uid": f"fixture-{index}", "premise": f"premise-{index}", "hypothesis": f"hypothesis-{index}"}
    for index in range(4)
]


class DeterministicEvaluationSelectionTest(unittest.TestCase):
    def test_hans_384_cap_preserves_exact_prefixed_raw_key_membership(self):
        rows = [
            {
                "pairID": f"ex{index}",
                "canonical_pair_id": f"hans_evaluation::ex{index}",
                "gold_label": ("entailment", "non-entailment")[index % 2],
                "heuristic": ("lexical_overlap", "subsequence", "constituent")[index % 3],
                "subcase": f"case-{index % 6}",
            }
            for index in range(30_000)
        ]
        pre_fix_rows, pre_fix_local_ids = deterministic_cap_records(
            rows,
            384,
            42,
            ("gold_label", "heuristic", "subcase"),
            preferred_fields=("pairID",),
        )

        selected, artifact_ids = select_hans_evaluation_records(rows, 384, 42)

        self.assertEqual(384, len(selected))
        self.assertEqual(pre_fix_local_ids, [row["pairID"] for row in selected])
        self.assertEqual(
            [row["canonical_pair_id"] for row in pre_fix_rows],
            artifact_ids,
        )
        qualified_key_rows, _ = deterministic_cap_records(
            rows,
            384,
            42,
            ("gold_label", "heuristic", "subcase"),
            preferred_fields=("canonical_pair_id",),
        )
        self.assertNotEqual(
            pre_fix_local_ids,
            [row["pairID"] for row in qualified_key_rows],
        )

    def test_hans_cap_is_order_independent_and_covers_label_heuristic_strata(self):
        selected_a, ids_a = deterministic_cap_records(
            HANS_ROWS,
            12,
            42,
            ("gold_label", "heuristic", "subcase"),
            preferred_fields=("pairID",),
        )
        selected_b, ids_b = deterministic_cap_records(
            list(reversed(HANS_ROWS)),
            12,
            42,
            ("gold_label", "heuristic", "subcase"),
            preferred_fields=("pairID",),
        )

        self.assertEqual(ids_a, ids_b)
        self.assertEqual(ids_a, [row["pairID"] for row in selected_a])
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


class DataIdentityManifestTest(unittest.TestCase):
    def test_identity_entry_separates_full_and_smoke_selected_membership(self):
        entry = dataset_identity_entry(
            MANIFEST_ROWS,
            source="fixture",
            split="test",
            preferred_id_fields=("uid",),
            selected_limit=2,
            seed=42,
        )

        self.assertEqual(4, entry["full_count"])
        self.assertEqual(2, entry["selected_count"])
        self.assertEqual(["fixture-0", "fixture-1", "fixture-2", "fixture-3"], entry["full_ids"])
        self.assertTrue(set(entry["selected_ids"]).issubset(set(entry["full_ids"])))
        self.assertEqual(64, len(entry["full_ids_sha256"]))
        self.assertEqual(64, len(entry["selected_ids_sha256"]))
        self.assertEqual("preferred_field_or_content_sha256", entry["id_strategy"])
        self.assertEqual("fixture", entry["source"])
        self.assertEqual("test", entry["split"])
        self.assertEqual(42, entry["selection_seed"])
        self.assertEqual(2, entry["selected_limit"])

    def test_identity_entry_rejects_empty_duplicate_and_out_of_range_membership(self):
        kwargs = {
            "source": "fixture",
            "split": "test",
            "preferred_id_fields": ("uid",),
            "seed": 42,
        }
        with self.assertRaisesRegex(ValueError, "empty"):
            dataset_identity_entry([], selected_limit=None, **kwargs)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            dataset_identity_entry([{"uid": "same"}, {"uid": "same"}], selected_limit=None, **kwargs)
        with self.assertRaisesRegex(ValueError, "selected"):
            dataset_identity_entry(MANIFEST_ROWS, selected_limit=5, **kwargs)

    def test_identity_entry_accepts_actual_selection_smaller_than_a_cap(self):
        entry = dataset_identity_entry(
            MANIFEST_ROWS,
            source="fixture",
            split="test",
            preferred_id_fields=("uid",),
            selected_limit=5,
            selected_records=MANIFEST_ROWS,
            seed=42,
        )

        self.assertEqual(4, entry["selected_count"])

    def test_identity_entry_requires_exact_explicit_selected_membership(self):
        kwargs = {
            "source": "fixture",
            "split": "test",
            "preferred_id_fields": ("uid",),
            "seed": 42,
        }
        cases = (
            ("empty", 2, []),
            ("duplicate", 2, [MANIFEST_ROWS[0], MANIFEST_ROWS[0]]),
            ("full dataset", None, MANIFEST_ROWS[:2]),
            ("selected", 2, MANIFEST_ROWS[:1]),
            ("not present", 2, [{"uid": "outside"}, MANIFEST_ROWS[0]]),
        )

        for message, selected_limit, selected_records in cases:
            with self.subTest(message=message, selected_limit=selected_limit):
                with self.assertRaisesRegex(ValueError, message):
                    dataset_identity_entry(
                        MANIFEST_ROWS,
                        selected_limit=selected_limit,
                        selected_records=selected_records,
                        **kwargs,
                    )

    def test_identity_entry_checksums_use_utf8_canonical_json_for_full_and_selected_ids(self):
        rows = [
            {"uid": "café-α"},
            {"uid": "東京-β"},
            {"uid": "München-γ"},
        ]
        selected = [rows[2], rows[0]]
        entry = dataset_identity_entry(
            rows,
            source="fixture",
            split="test",
            preferred_id_fields=("uid",),
            selected_limit=2,
            selected_records=selected,
            seed=42,
        )

        def checksum(ids):
            payload = json.dumps(
                ids,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return sha256(payload).hexdigest()

        self.assertEqual(["café-α", "東京-β", "München-γ"], entry["full_ids"])
        self.assertEqual(["München-γ", "café-α"], entry["selected_ids"])
        self.assertEqual(checksum(entry["full_ids"]), entry["full_ids_sha256"])
        self.assertEqual(checksum(entry["selected_ids"]), entry["selected_ids_sha256"])

    def test_ood_raw_loader_interfaces_are_public_without_network_access(self):
        dataloader = StructuredDataAccessAuditTest._dataloader_module_with_dependency_stubs()

        for name in (
            "load_esnli_raw",
            "load_anli_raw",
            "load_snli_hard_raw",
            "load_wanli_raw",
        ):
            self.assertTrue(callable(getattr(dataloader, name)))

    def test_backend_writes_v4_hans_integrities_and_bound_audit_without_network(self):
        class FixtureRows:
            def __init__(self, rows):
                self.rows = [dict(row) for row in rows]

            def __len__(self):
                return len(self.rows)

            def __iter__(self):
                return iter(self.rows)

            def __getitem__(self, index):
                return self.rows[index]

            def shuffle(self, seed):
                del seed
                return self

            def select(self, indexes):
                return FixtureRows([self.rows[index] for index in indexes])

        mnli = {
            "train": FixtureRows([{"idx": f"train-{index}"} for index in range(4)]),
            "validation_matched": FixtureRows([{"idx": f"validation-{index}"} for index in range(4)]),
        }
        hans_manifest = {
            "build_pair_ids": ["hans_train::ex0"],
            "dev_pair_ids": ["hans_train::ex1"],
            "evaluation_pair_ids": [
                "hans_evaluation::ex0",
                "hans_evaluation::ex1",
                "hans_evaluation::ex2",
                "hans_evaluation::ex3",
            ],
            "build_records": [{
                "pairID": "ex0",
                "canonical_pair_id": "hans_train::ex0",
                "gold_label": "entailment",
                "heuristic": "lexical_overlap",
                "subcase": "train-build-case",
                "sentence1": "train-build-premise",
                "sentence2": "train-build-hypothesis",
                "sentence1_binary_parse": "( train-build-premise )",
                "sentence2_binary_parse": "( train-build-hypothesis )",
                "sentence1_parse": "(ROOT ( train-build-premise ))",
                "sentence2_parse": "(ROOT ( train-build-hypothesis ))",
                "template": "temp_train_build",
            }],
            "dev_records": [{
                "pairID": "ex1",
                "canonical_pair_id": "hans_train::ex1",
                "gold_label": "non-entailment",
                "heuristic": "subsequence",
                "subcase": "train-dev-case",
                "sentence1": "train-dev-premise",
                "sentence2": "train-dev-hypothesis",
                "sentence1_binary_parse": "( train-dev-premise )",
                "sentence2_binary_parse": "( train-dev-hypothesis )",
                "sentence1_parse": "(ROOT ( train-dev-premise ))",
                "sentence2_parse": "(ROOT ( train-dev-hypothesis ))",
                "template": "temp_train_dev",
            }],
            "evaluation_records": [
                {
                    "pairID": f"ex{index - 1}",
                    "canonical_pair_id": f"hans_evaluation::ex{index - 1}",
                    "gold_label": label,
                    "heuristic": heuristic,
                    "subcase": f"{heuristic}-case",
                    "sentence1": f"evaluation-premise-{index}",
                    "sentence2": f"evaluation-hypothesis-{index}",
                    "sentence1_binary_parse": f"( evaluation-premise-{index} )",
                    "sentence2_binary_parse": f"( evaluation-hypothesis-{index} )",
                    "sentence1_parse": f"(ROOT ( evaluation-premise-{index} ))",
                    "sentence2_parse": f"(ROOT ( evaluation-hypothesis-{index} ))",
                    "template": f"temp_evaluation_{index}",
                }
                for index, (label, heuristic) in enumerate(
                    (
                        ("entailment", "lexical_overlap"),
                        ("non-entailment", "lexical_overlap"),
                        ("entailment", "subsequence"),
                        ("non-entailment", "constituent"),
                    ),
                    start=1,
                )
            ],
        }
        split_payload = {
            "schema_version": "hans_split_v1",
            "hans_split_seed": 42,
            "build_pair_ids": ["ex0"],
            "dev_pair_ids": ["ex1"],
            "small_strata": [],
        }
        hans_manifest["split_integrity"] = {
            "schema_version": "hans_split_integrity_v1",
            "seed": 42,
            "split_algorithm": "source_local_id_sort_numpy_default_rng_per_stratum_v1",
            "checksum_algorithm": "sha256_canonical_json_utf8_v1",
            "build_count": 1,
            "dev_count": 1,
            "build_source_pair_ids": ["ex0"],
            "dev_source_pair_ids": ["ex1"],
            "small_strata": [],
            "split_checksum": sha256(
                json.dumps(
                    split_payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        raw_ood = FixtureRows(MANIFEST_ROWS)
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *_args, **_kwargs: mnli
        fake_dataloader = types.ModuleType("data.dataloader")
        fake_dataloader.make_hans_split_manifest = lambda _cfg: hans_manifest
        fake_dataloader.load_esnli_raw = lambda: raw_ood
        fake_dataloader.load_anli_raw = lambda: raw_ood
        fake_dataloader.load_snli_hard_raw = lambda: raw_ood
        fake_dataloader.load_wanli_raw = lambda: raw_ood

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {"datasets": fake_datasets, "data.dataloader": fake_dataloader}
        ):
            backend = RealCanonicalBackend(
                TrainConfig(
                    output_dir=str(Path(tmp) / "unused"),
                    mnli_train_size=2,
                    mnli_val_size=2,
                    hans_eval_size=2,
                    esnli_eval_size=1,
                    anli_eval_size=1,
                    snli_hard_eval_size=1,
                    wanli_eval_size=1,
                )
            )
            output = Path(tmp) / "out"
            backend.initialize_manifests(output, Path(tmp) / "protocol.md")

            manifest = json.loads((output / "manifests" / "data_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("canonical_data_manifest_v4", manifest["schema_version"])
            self.assertEqual("stage2_smoke", manifest["scope"])
            self.assertEqual({"train", "validation_matched"}, set(manifest["mnli"]))
            self.assertEqual(
                {
                    "build", "dev", "evaluation", "split_integrity",
                    "content_integrity", "selection_integrity", "source_integrity",
                },
                set(manifest["hans"]),
            )
            self.assertEqual(
                "hans_source_integrity_v1",
                manifest["hans"]["source_integrity"]["schema_version"],
            )
            self.assertEqual(
                {"train", "evaluation"},
                set(manifest["hans"]["source_integrity"]["sources"]),
            )
            self.assertEqual(
                2,
                manifest["hans"]["source_integrity"]["sources"]["train"]["count"],
            )
            self.assertEqual(
                4,
                manifest["hans"]["source_integrity"]["sources"]["evaluation"]["count"],
            )
            self.assertEqual({"esnli", "anli", "snli_hard", "wanli"}, set(manifest["ood"]))
            self.assertEqual(4, manifest["mnli"]["train"]["full_count"])
            self.assertEqual(2, manifest["mnli"]["train"]["selected_count"])
            selected_records, expected_hans_ids = select_hans_evaluation_records(
                hans_manifest["evaluation_records"], 2, 42
            )
            self.assertEqual(
                expected_hans_ids,
                manifest["hans"]["evaluation"]["selected_ids"],
            )
            self.assertEqual(
                [row["pairID"] for row in selected_records],
                [identity.removeprefix("hans_evaluation::") for identity in expected_hans_ids],
            )
            self.assertEqual(
                "hans_content_integrity_v1",
                manifest["hans"]["content_integrity"]["schema_version"],
            )
            self.assertEqual(
                "hans_selection_integrity_v1",
                manifest["hans"]["selection_integrity"]["schema_version"],
            )
            events = read_jsonl(output / "manifests" / "data_access.jsonl")
            self.assertEqual("manifest_identity_only", events[0]["purpose"])
            self.assertEqual(
                {
                    "identity_counts", "identity_checksums", "split_integrity_summary",
                    "content_integrity_summary", "selection_integrity_summary",
                },
                set(events[1])
                - {"sequence", "timestamp", "event", "dataset", "split", "purpose"},
            )
            self.assertEqual(
                manifest["hans"]["split_integrity"]["split_checksum"],
                events[1]["split_integrity_summary"]["split_checksum"],
            )


class StructuredDataAccessAuditTest(unittest.TestCase):
    @staticmethod
    def _dataloader_module_with_dependency_stubs():
        numpy_stub = types.ModuleType("numpy")
        torch_stub = types.ModuleType("torch")
        torch_stub.cuda = SimpleNamespace(is_available=lambda: False)
        torch_stub.manual_seed = lambda _seed: None
        torch_utils_stub = types.ModuleType("torch.utils")
        torch_utils_data_stub = types.ModuleType("torch.utils.data")
        torch_utils_data_stub.DataLoader = object
        datasets_stub = types.ModuleType("datasets")

        class Dataset:
            pass

        datasets_stub.Dataset = Dataset
        datasets_stub.Value = object
        datasets_stub.concatenate_datasets = lambda datasets: datasets[0]
        datasets_stub.load_dataset = lambda *_args, **_kwargs: None
        with patch.dict(
            sys.modules,
            {
                "numpy": numpy_stub,
                "torch": torch_stub,
                "torch.utils": torch_utils_stub,
                "torch.utils.data": torch_utils_data_stub,
                "datasets": datasets_stub,
            },
        ):
            sys.modules.pop("data.dataloader", None)
            return importlib.import_module("data.dataloader")

    def test_final_hans_access_follows_final_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data_access.jsonl"

            record_final_evaluation_start(SimpleNamespace(data_access_log=str(path)))
            record_dataset_access(
                SimpleNamespace(data_access_log=str(path)),
                dataset="hans",
                split="evaluation",
                purpose="final",
            )

            events = read_jsonl(path)
            marker_index = next(
                index
                for index, event in enumerate(events)
                if event["event"] == "final_evaluation_start"
            )
            hans_index = next(
                index
                for index, event in enumerate(events)
                if event.get("dataset") == "hans" and event.get("split") == "evaluation"
            )
            self.assertLess(marker_index, hans_index)
            self.assertEqual(list(range(len(events))), [event["sequence"] for event in events])

    def test_append_access_event_rejects_reserved_payload_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data_access.jsonl"
            for reserved_key in ("sequence", "timestamp"):
                with self.assertRaisesRegex(ValueError, "reserved"):
                    append_access_event(
                        path,
                        dataset="hans",
                        split="evaluation",
                        purpose="final",
                        event="dataset_access",
                        **{reserved_key: "override"},
                    )
            self.assertFalse(path.exists())

    def test_hans_evaluation_loader_requires_final_marker_before_access(self):
        dataloader = self._dataloader_module_with_dependency_stubs()

        class DatasetStub:
            def set_format(self, **_kwargs):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data_access.jsonl"
            cfg = SimpleNamespace(data_access_log=str(path), batch_size=1)
            with patch.object(dataloader, "_prepare_hans_base_dataset", return_value=DatasetStub()), patch.object(
                dataloader, "DataLoader", return_value=object()
            ):
                with self.assertRaisesRegex(ValueError, "final_evaluation_start"):
                    dataloader.make_hans_evaluation_loader(cfg, object())
                record_final_evaluation_start(cfg)
                dataloader.make_hans_evaluation_loader(cfg, object())

            self.assertEqual(
                {
                    "event": "dataset_access",
                    "dataset": "hans",
                    "split": "evaluation",
                    "purpose": "final",
                },
                {
                    key: read_jsonl(path)[1][key]
                    for key in ("event", "dataset", "split", "purpose")
                },
            )

    def test_manifest_identity_only_hans_access_is_allowed_without_final_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifests" / "data_access.jsonl"
            append_access_event(
                path,
                dataset="hans",
                split="evaluation",
                purpose="manifest_identity_only",
                event="dataset_access",
            )
            self.assertEqual(
                "manifest_identity_only",
                read_jsonl(path)[0]["purpose"],
            )

    def test_raw_hans_loader_qualifies_ids_before_returning_official_rows(self):
        dataloader = self._dataloader_module_with_dependency_stubs()

        class RawDataset:
            def __init__(self, pair_id):
                self.rows = [{"pairID": pair_id}]

            def map(self, callback, batched=False):
                self.rows = [{**row, **callback(dict(row))} for row in self.rows]
                return self

        with patch.object(dataloader, "load_dataset", return_value=RawDataset("ex0")):
            train = dataloader._load_hans_dataset("train")
        with patch.object(dataloader, "load_dataset", return_value=RawDataset("ex0")):
            evaluation = dataloader._load_hans_dataset("eval")

        self.assertEqual("ex0", train.rows[0]["pairID"])
        self.assertEqual("hans_train::ex0", train.rows[0]["canonical_pair_id"])
        self.assertEqual("ex0", evaluation.rows[0]["pairID"])
        self.assertEqual(
            "hans_evaluation::ex0", evaluation.rows[0]["canonical_pair_id"]
        )

    def test_hans_evaluation_loader_passes_pair_id_preference_to_capping(self):
        dataloader = self._dataloader_module_with_dependency_stubs()

        class HansDataset:
            def __init__(self, rows):
                self.rows = rows

            @property
            def column_names(self):
                return list(self.rows[0])

            def map(self, callback, batched=False):
                self.rows = [{**row, **callback(row)} for row in self.rows]
                return self

            def rename_column(self, old, new):
                self.rows = [
                    {new if key == old else key: value for key, value in row.items()}
                    for row in self.rows
                ]
                return self

            def filter(self, predicate):
                self.rows = [row for row in self.rows if predicate(row)]
                return self

        captured = {}

        def stop_after_capping(dataset, limit, seed, strata_fields, preferred_fields):
            captured["args"] = (dataset, limit, seed, strata_fields, preferred_fields)
            raise RuntimeError("capping observed")

        rows = [{
            "pairID": "ex1",
            "canonical_pair_id": "hans_evaluation::ex1",
            "gold_label": "entailment",
            "heuristic": "lexical_overlap",
            "subcase": "case",
            "sentence1": "premise",
            "sentence2": "hypothesis",
        }]
        cfg = SimpleNamespace(hans_eval_size=1, data_seed=42, batch_size=1, data_access_log=None)
        with patch.object(dataloader, "_load_hans_dataset", return_value=HansDataset(rows)), patch.object(
            dataloader, "_cap_final_evaluation_dataset", side_effect=stop_after_capping
        ):
            with self.assertRaisesRegex(RuntimeError, "capping observed"):
                dataloader.make_hans_evaluation_loader(cfg, object())

        self.assertEqual(
            (1, 42, ("gold_label", "heuristic", "subcase"), ("pairID",)),
            captured["args"][1:],
        )
        self.assertEqual(
            "ex1",
            captured["args"][0].rows[0]["pairID"],
        )

    def test_evaluation_cap_order_is_preserved_in_tokenized_prediction_ids(self):
        dataloader = self._dataloader_module_with_dependency_stubs()

        class HansDataset:
            def __init__(self, rows):
                self.rows = [dict(row) for row in rows]

            @property
            def column_names(self):
                return list(self.rows[0])

            def map(self, callback, batched=False):
                if batched:
                    batch = {
                        key: [row[key] for row in self.rows]
                        for key in self.rows[0]
                    }
                    additions = callback(batch)
                    self.rows = [
                        {**row, **{key: values[index] for key, values in additions.items()}}
                        for index, row in enumerate(self.rows)
                    ]
                else:
                    self.rows = [{**row, **callback(dict(row))} for row in self.rows]
                return self

            def rename_column(self, old, new):
                self.rows = [
                    {new if key == old else key: value for key, value in row.items()}
                    for row in self.rows
                ]
                return self

            def filter(self, predicate):
                self.rows = [row for row in self.rows if predicate(row)]
                return self

        rows = [
            {
                "pairID": f"ex{index}",
                "canonical_pair_id": f"hans_evaluation::ex{index}",
                "gold_label": label,
                "heuristic": heuristic,
                "subcase": f"case-{index}",
                "sentence1": f"premise-{index}",
                "sentence2": f"hypothesis-{index}",
            }
            for index, (label, heuristic) in enumerate(
                (("entailment", "lexical_overlap"), ("non-entailment", "subsequence"))
            )
        ]
        cfg = SimpleNamespace(hans_eval_size=2, data_seed=42, max_seq_length=8)

        def known_cap(dataset, *_args, **_kwargs):
            return HansDataset(list(reversed(dataset.rows)))

        def tokenizer(premises, hypotheses, **_kwargs):
            return {
                "input_ids": [[index] for index, _ in enumerate(premises)],
                "attention_mask": [[1] for _ in hypotheses],
            }

        with patch.object(dataloader, "_load_hans_dataset", return_value=HansDataset(rows)), patch.object(
            dataloader, "_cap_final_evaluation_dataset", side_effect=known_cap
        ):
            prepared = dataloader._prepare_hans_base_dataset(
                cfg, tokenizer, split="evaluation"
            )

        self.assertEqual(
            ["hans_evaluation::ex1", "hans_evaluation::ex0"],
            [row["pair_id"] for row in prepared.rows],
        )

    def test_shared_metadata_contains_class_priors_and_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = RealCanonicalBackend(TrainConfig(output_dir=str(root / "unused")))
            captured = {}

            def fake_train(cfg, **_kwargs):
                checkpoint = Path(cfg.checkpoint_dir) / "shared_phase2_checkpoint.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"shared")
                captured["cfg"] = cfg
                return {
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_hash": sha256_file(checkpoint),
                    "class_prior_weights": {0: 0.1, 1: 0.2, 2: 0.3},
                }

            fake_trainer = types.ModuleType("training.trainer")
            fake_trainer.train_ties_unlearn = fake_train
            with patch.dict(sys.modules, {"training.trainer": fake_trainer}):
                shared_dir = root / "seed_42" / "shared_phase2"
                backend.prepare_shared(42, shared_dir)

            metadata = json.loads(
                (shared_dir / "shared_checkpoint_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual("canonical_shared_phase2", metadata["checkpoint_role"])
            self.assertEqual(64, len(metadata["checkpoint_sha256"]))
            self.assertEqual({"0", "1", "2"}, set(metadata["class_prior_weights"]))
            self.assertEqual(str(shared_dir / "data_access.jsonl"), captured["cfg"].data_access_log)

    def test_runner_rejects_success_when_audit_artifacts_are_missing(self):
        class BackendWithoutAudits:
            def initialize_manifests(self, output_dir, _protocol_path):
                manifests = Path(output_dir) / "manifests"
                write_json(manifests / "data_manifest.json", {"schema_version": "data_manifest_v1"})
                write_json(manifests / "environment_manifest.json", {"schema_version": "environment_manifest_v1"})

            def prepare_shared(self, training_seed, shared_dir):
                shared_dir = Path(shared_dir)
                write_json(shared_dir / "config.json", {"training_seed": training_seed})
                checkpoint = shared_dir / "checkpoints" / "shared.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"shared")
                checkpoint_hash = sha256_file(checkpoint)
                write_json(
                    shared_dir / "shared_checkpoint_metadata.json",
                    {
                        "checkpoint_role": "canonical_shared_phase2",
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": checkpoint_hash,
                        "class_prior_weights": {"0": 0.1, "1": 0.2, "2": 0.3},
                    },
                )
                return CheckpointRef(checkpoint, checkpoint_hash)

            def run_standard(self, condition, training_seed, run_dir):
                return self._write_method(condition, training_seed, run_dir)

            def run_branch(self, condition, training_seed, run_dir, _checkpoint):
                return self._write_method(condition, training_seed, run_dir)

            @staticmethod
            def _write_method(condition, training_seed, run_dir):
                run_dir = Path(run_dir)
                write_json(run_dir / "config.json", {"training_seed": training_seed})
                write_json(run_dir / "metrics.json", {})
                write_json(run_dir / "selected_layers.json", {})
                write_jsonl(run_dir / "hans_predictions.jsonl", [])
                return {"final_checkpoint_hash": "a" * 64}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = root / "FROZEN_EXPERIMENT_PROTOCOL.md"
            protocol.write_text("# frozen\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data_access.jsonl"):
                run_core(
                    protocol,
                    root / "output",
                    BackendWithoutAudits(),
                    fresh=True,
                    seeds=(42,),
                    git_metadata={
                        "commit": "f" * 40,
                        "branch": "test",
                        "dirty": False,
                        "status_porcelain": [],
                    },
                    command=["python", "run_canonical.py"],
                )

    def test_runner_rejects_method_success_when_only_method_audit_is_missing(self):
        class BackendWithSharedAuditOnly:
            def initialize_manifests(self, output_dir, _protocol_path):
                manifests = Path(output_dir) / "manifests"
                write_json(manifests / "data_manifest.json", {"schema_version": "data_manifest_v1"})
                write_json(manifests / "environment_manifest.json", {"schema_version": "environment_manifest_v1"})

            def prepare_shared(self, training_seed, shared_dir):
                shared_dir = Path(shared_dir)
                write_json(shared_dir / "config.json", {"training_seed": training_seed})
                checkpoint = shared_dir / "checkpoints" / "shared.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"shared")
                checkpoint_hash = sha256_file(checkpoint)
                write_json(
                    shared_dir / "shared_checkpoint_metadata.json",
                    {
                        "checkpoint_role": "canonical_shared_phase2",
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": checkpoint_hash,
                        "class_prior_weights": {"0": 0.1, "1": 0.2, "2": 0.3},
                    },
                )
                write_jsonl(shared_dir / "data_access.jsonl", [])
                return CheckpointRef(checkpoint, checkpoint_hash)

            def run_standard(self, condition, training_seed, run_dir):
                run_dir = Path(run_dir)
                write_json(run_dir / "config.json", {"training_seed": training_seed})
                write_json(run_dir / "metrics.json", {})
                write_json(run_dir / "selected_layers.json", {})
                write_jsonl(run_dir / "hans_predictions.jsonl", [])
                return {"final_checkpoint_hash": "a" * 64}

            def run_branch(self, *_args):
                raise AssertionError("method execution must stop at standard_lora")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = root / "FROZEN_EXPERIMENT_PROTOCOL.md"
            protocol.write_text("# frozen\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data_access.jsonl"):
                run_core(
                    protocol,
                    root / "output",
                    BackendWithSharedAuditOnly(),
                    fresh=True,
                    seeds=(42,),
                    git_metadata={
                        "commit": "f" * 40,
                        "branch": "test",
                        "dirty": False,
                        "status_porcelain": [],
                    },
                    command=["python", "run_canonical.py"],
                )
            status = json.loads(
                (root / "output" / "seed_42" / "standard_lora" / "status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("failed", status["state"])


if __name__ == "__main__":
    unittest.main()
