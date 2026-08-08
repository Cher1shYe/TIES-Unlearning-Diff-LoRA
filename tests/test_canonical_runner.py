import json
from hashlib import sha256
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.artifacts import read_jsonl, sha256_file, write_json, write_jsonl
from canonical.backend import RealCanonicalBackend
from canonical.conditions import CONDITIONS
from canonical.runner import CheckpointRef, run_core
from configs.config import TrainConfig


class FakeBackend:
    def __init__(self, fail_prepare=False, fail_method=None):
        self.fail_prepare = fail_prepare
        self.fail_method = fail_method
        self.initialize_calls = 0
        self.prepare_calls = 0
        self.standard_calls = []
        self.branch_calls = []
        self.method_calls = []

    def initialize_manifests(self, output_dir, protocol_path):
        self.initialize_calls += 1
        manifests = Path(output_dir) / "manifests"
        write_json(manifests / "data_manifest.json", {"schema_version": "data_manifest_v1"})
        write_json(manifests / "environment_manifest.json", {"schema_version": "environment_manifest_v1"})

    def prepare_shared(self, training_seed, shared_dir):
        self.prepare_calls += 1
        if self.fail_prepare:
            raise RuntimeError("prepare failed")
        shared_dir = Path(shared_dir)
        write_json(shared_dir / "config.json", {"training_seed": training_seed, "role": "shared"})
        checkpoint = shared_dir / "checkpoints" / "shared.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"shared-{training_seed}".encode("ascii"))
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

    def _write_method_artifacts(self, tag, training_seed, run_dir):
        if self.fail_method == tag:
            raise RuntimeError(f"{tag} failed")
        run_dir = Path(run_dir)
        write_json(run_dir / "config.json", {"method_tag": tag, "training_seed": training_seed})
        write_json(run_dir / "metrics.json", {"hans": {"hans_overall": 0.5}})
        write_json(run_dir / "selected_layers.json", {"shortcut_layers": []})
        write_jsonl(run_dir / "data_access.jsonl", [])
        write_jsonl(
            run_dir / "hans_predictions.jsonl",
            [{
                "pair_id": "p-1",
                "gold_label": "entailment",
                "predicted_label": "entailment",
                "entailment_probability": 0.8,
                "heuristic": "lexical_overlap",
                "subcase": "case",
                "training_seed": training_seed,
                "method_tag": tag,
                "checkpoint_hash": "a" * 64,
            }],
        )
        return {"final_checkpoint_hash": "a" * 64}

    def run_standard(self, condition, training_seed, run_dir):
        self.method_calls.append(condition.tag)
        self.standard_calls.append((condition.tag, training_seed))
        return self._write_method_artifacts(condition.tag, training_seed, run_dir)

    def run_branch(self, condition, training_seed, run_dir, checkpoint):
        self.method_calls.append(condition.tag)
        self.branch_calls.append((condition.tag, training_seed, checkpoint.sha256))
        return self._write_method_artifacts(condition.tag, training_seed, run_dir)


class CanonicalRunnerContractTest(unittest.TestCase):
    def _inputs(self, root):
        protocol = Path(root) / "FROZEN_EXPERIMENT_PROTOCOL.md"
        protocol.write_text("# frozen\n", encoding="utf-8")
        output = Path(root) / "canonical_v1"
        return protocol, output

    def _run(self, protocol, output, backend, **overrides):
        return run_core(
            protocol,
            output,
            backend,
            git_metadata={"commit": "f" * 40, "branch": "test", "dirty": False, "status_porcelain": []},
            command=["python", "run_canonical.py"],
            **overrides,
        )

    def test_one_shared_checkpoint_is_passed_to_all_five_dual_branches(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol, output = self._inputs(tmp)
            backend = FakeBackend()

            self._run(protocol, output, backend, fresh=True, seeds=(42,))

            self.assertEqual(1, backend.initialize_calls)
            self.assertEqual(1, backend.prepare_calls)
            self.assertEqual(1, len(backend.standard_calls))
            self.assertEqual(5, len(backend.branch_calls))
            self.assertEqual(1, len({call[2] for call in backend.branch_calls}))
            for tag in ("standard_lora", "full_sr", "subtraction_only", "reweight_only", "staged_neither", "class_prior_reweight"):
                status = json.loads((output / "seed_42" / tag / "status.json").read_text(encoding="utf-8"))
                self.assertEqual("success", status["state"])
                manifest = json.loads(
                    (output / "seed_42" / tag / "run_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    sha256_file(output / "manifests" / "data_manifest.json"),
                    manifest["data_manifest_sha256"],
                )
                self.assertEqual(
                    sha256_file(output / "manifests" / "environment_manifest.json"),
                    manifest["environment_manifest_sha256"],
                )

    def test_method_order_is_rotated_by_frozen_seed_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol, output = self._inputs(tmp)
            backend = FakeBackend()

            self._run(protocol, output, backend, fresh=True, seeds=(123,))

            self.assertEqual(
                ["full_sr", "subtraction_only", "reweight_only", "staged_neither", "class_prior_reweight", "standard_lora"],
                backend.method_calls,
            )

    def test_run_core_defaults_still_execute_thirty_method_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol, output = self._inputs(tmp)
            backend = FakeBackend()

            result = self._run(protocol, output, backend, fresh=True)

            self.assertEqual(30, len(result["executed"]))
            self.assertEqual(5, backend.prepare_calls)
            self.assertEqual(30, len(backend.method_calls))

    def test_fresh_refuses_non_empty_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol, output = self._inputs(tmp)
            output.mkdir()
            (output / "existing.txt").write_text("do not overwrite", encoding="utf-8")
            backend = FakeBackend()

            with self.assertRaisesRegex(ValueError, "empty output directory"):
                self._run(protocol, output, backend, fresh=True, seeds=(42,))
            self.assertEqual(0, backend.initialize_calls)

    def test_run_core_fresh_rejects_root_commands_provenance_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol, output = self._inputs(tmp)
            output.mkdir()
            write_json(output / "commands.json", {"mode": "primary"})
            backend = FakeBackend()

            with self.assertRaisesRegex(ValueError, "empty output directory"):
                self._run(protocol, output, backend, fresh=True, seeds=(42,))

            self.assertEqual(0, backend.initialize_calls)

    def test_failed_shared_preparation_stops_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol, output = self._inputs(tmp)
            backend = FakeBackend(fail_prepare=True)

            with self.assertRaisesRegex(RuntimeError, "prepare failed"):
                self._run(protocol, output, backend, fresh=True, seeds=(42,))
            self.assertEqual([], backend.method_calls)
            status = json.loads((output / "seed_42" / "shared_phase2" / "status.json").read_text(encoding="utf-8"))
            self.assertEqual("failed", status["state"])

    def test_failed_branch_stops_later_methods_for_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol, output = self._inputs(tmp)
            backend = FakeBackend(fail_method="full_sr")

            with self.assertRaisesRegex(RuntimeError, "full_sr failed"):
                self._run(protocol, output, backend, fresh=True, seeds=(42,))
            self.assertEqual(["standard_lora", "full_sr"], backend.method_calls)
            self.assertFalse((output / "seed_42" / "subtraction_only").exists())

    def test_resume_skips_only_checksum_valid_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol, output = self._inputs(tmp)
            first = FakeBackend()
            self._run(protocol, output, first, fresh=True, seeds=(42,))

            second = FakeBackend()
            self._run(protocol, output, second, fresh=False, seeds=(42,))
            self.assertEqual(0, second.prepare_calls)
            self.assertEqual([], second.method_calls)

            (output / "seed_42" / "standard_lora" / "metrics.json").write_text("{}\n", encoding="utf-8")
            third = FakeBackend()
            self._run(protocol, output, third, fresh=False, seeds=(42,))
            self.assertEqual(["standard_lora"], third.method_calls)
            self.assertEqual(0, third.prepare_calls)


class RealCanonicalBackendContractTest(unittest.TestCase):
    def test_manifest_identity_access_is_logged_before_hans_manifest_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = RealCanonicalBackend(TrainConfig(output_dir=str(root / "unused")))
            datasets_stub = types.ModuleType("datasets")
            datasets_stub.load_dataset = lambda *_args, **_kwargs: {
                "train": [{"idx": 1}],
                "validation_matched": [{"idx": 2}],
            }
            dataloader_stub = types.ModuleType("data.dataloader")

            def fake_hans_manifest(_cfg):
                events = read_jsonl(root / "output" / "manifests" / "data_access.jsonl")
                self.assertEqual(
                    {
                        "event": "dataset_access",
                        "dataset": "hans",
                        "split": "evaluation",
                        "purpose": "manifest_identity_only",
                    },
                    {key: events[0][key] for key in ("event", "dataset", "split", "purpose")},
                )
                return {
                    "build_pair_ids": ["hans_train::ex0"],
                    "dev_pair_ids": ["hans_train::ex1"],
                    "evaluation_pair_ids": ["hans_evaluation::ex0"],
                    "build_records": [{
                        "pairID": "ex0", "canonical_pair_id": "hans_train::ex0",
                        "gold_label": "entailment", "heuristic": "lexical_overlap",
                        "subcase": "build", "sentence1": "build premise", "sentence2": "build hypothesis",
                    }],
                    "dev_records": [{
                        "pairID": "ex1", "canonical_pair_id": "hans_train::ex1",
                        "gold_label": "non-entailment", "heuristic": "subsequence",
                        "subcase": "dev", "sentence1": "dev premise", "sentence2": "dev hypothesis",
                    }],
                    "evaluation_records": [{
                        "pairID": "ex0", "canonical_pair_id": "hans_evaluation::ex0",
                        "gold_label": "non-entailment", "heuristic": "constituent",
                        "subcase": "evaluation", "sentence1": "evaluation premise", "sentence2": "evaluation hypothesis",
                    }],
                    "split_integrity": {
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
                                {
                                    "schema_version": "hans_split_v1",
                                    "hans_split_seed": 42,
                                    "build_pair_ids": ["ex0"],
                                    "dev_pair_ids": ["ex1"],
                                    "small_strata": [],
                                },
                                ensure_ascii=False,
                                allow_nan=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest(),
                    },
                }

            dataloader_stub.make_hans_split_manifest = fake_hans_manifest
            raw_ood = [{"id": "ood-1"}]
            dataloader_stub.load_esnli_raw = lambda: raw_ood
            dataloader_stub.load_anli_raw = lambda: raw_ood
            dataloader_stub.load_snli_hard_raw = lambda: raw_ood
            dataloader_stub.load_wanli_raw = lambda: raw_ood
            with patch.dict(
                sys.modules,
                {"datasets": datasets_stub, "data.dataloader": dataloader_stub},
            ), patch("canonical.backend.collect_environment_metadata", return_value={}):
                backend.initialize_manifests(root / "output", root / "protocol.md")

            self.assertEqual(
                "manifest_identity_only",
                read_jsonl(root / "output" / "manifests" / "data_access.jsonl")[0]["purpose"],
            )

    def test_real_backend_rejects_unclean_hans_split(self):
        with self.assertRaisesRegex(ValueError, "hans_clean_split"):
            RealCanonicalBackend(TrainConfig(hans_clean_split=False))

    def test_shared_and_branch_configs_use_exact_run_directories_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = TrainConfig(output_dir=str(root / "unused"))
            backend = RealCanonicalBackend(base)
            calls = {}

            fake_trainer = types.ModuleType("training.trainer")

            def fake_train(cfg, **kwargs):
                if kwargs.get("stop_after_phase2"):
                    checkpoint = Path(cfg.checkpoint_dir) / "shared_phase2_checkpoint.pt"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.write_bytes(b"shared")
                    calls["prepare"] = (cfg, kwargs)
                    return {
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_hash": sha256_file(checkpoint),
                        "class_prior_weights": {0: 0.1, 1: 0.2, 2: 0.3},
                    }
                calls["branch"] = (cfg, kwargs)
                write_json(Path(cfg.output_dir) / cfg.experiment_name / "metrics.json", {})
                return {
                    "checkpoint_provenance": {"final_checkpoint_hash": "b" * 64}
                }

            fake_trainer.train_ties_unlearn = fake_train
            with patch.dict(sys.modules, {"training.trainer": fake_trainer}):
                shared_dir = root / "seed_42" / "shared_phase2"
                checkpoint = backend.prepare_shared(42, shared_dir)
                run_dir = root / "seed_42" / "full_sr"
                result = backend.run_branch(
                    CONDITIONS["full_sr"], 42, run_dir, checkpoint
                )

            prepare_cfg, prepare_kwargs = calls["prepare"]
            self.assertEqual(str(shared_dir.parent), prepare_cfg.output_dir)
            self.assertEqual(shared_dir.name, prepare_cfg.experiment_name)
            self.assertEqual(str(shared_dir / "checkpoints"), prepare_cfg.checkpoint_dir)
            self.assertEqual(str(shared_dir / "data_access.jsonl"), prepare_cfg.data_access_log)
            self.assertTrue(prepare_kwargs["stop_after_phase2"])

            branch_cfg, branch_kwargs = calls["branch"]
            self.assertEqual(str(run_dir.parent), branch_cfg.output_dir)
            self.assertEqual(run_dir.name, branch_cfg.experiment_name)
            self.assertEqual("n_guided", branch_cfg.phase3_weighting)
            self.assertEqual(str(checkpoint.path), branch_kwargs["shared_checkpoint_path"])
            self.assertEqual(checkpoint.sha256, branch_kwargs["checkpoint_hash"])
            self.assertEqual("b" * 64, result["final_checkpoint_hash"])
            self.assertTrue((shared_dir / "config.json").is_file())
            self.assertTrue((shared_dir / "shared_checkpoint_metadata.json").is_file())
            self.assertTrue((run_dir / "config.json").is_file())

    def test_standard_backend_uses_baseline_with_frozen_training_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = RealCanonicalBackend(TrainConfig(output_dir=str(root / "unused")))
            calls = {}
            fake_baseline = types.ModuleType("training.baseline")

            def fake_baseline_train(cfg, *, method_tag):
                calls["args"] = (cfg, method_tag)
                return {
                    "checkpoint_provenance": {"final_checkpoint_hash": "c" * 64}
                }

            fake_baseline.train_single_lora_baseline = fake_baseline_train
            with patch.dict(sys.modules, {"training.baseline": fake_baseline}):
                run_dir = root / "seed_123" / "standard_lora"
                result = backend.run_standard(
                    CONDITIONS["standard_lora"], 123, run_dir
                )

            cfg, tag = calls["args"]
            self.assertEqual(123, cfg.training_seed)
            self.assertEqual(42, cfg.data_seed)
            self.assertEqual(42, cfg.hans_split_seed)
            self.assertEqual("standard_lora", tag)
            self.assertEqual("c" * 64, result["final_checkpoint_hash"])


if __name__ == "__main__":
    unittest.main()
