import json
import importlib
import sys
import tempfile
import types
import unittest
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
from canonical.data import deterministic_cap_records, stable_record_id
from canonical.runner import CheckpointRef, run_core
from configs.config import TrainConfig


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
            "pairID": "pair-1",
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
