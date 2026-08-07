import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.artifacts import sha256_file, write_json, write_jsonl
from canonical.hans import aggregate_hans_predictions
from canonical.stage2_validation import (
    compare_a100_repeat,
    compare_metric_values,
    validate_smoke_root,
)


PRIMARY_CONDITIONS = ("standard_lora", "full_sr", "class_prior_reweight")
HASH_A = "a" * 64
HASH_B = "b" * 64


def _predictions(method, checkpoint_hash):
    return [
        {
            "pair_id": f"{method}-entailment",
            "gold_label": "entailment",
            "predicted_label": "entailment",
            "entailment_probability": 0.9,
            "heuristic": "lexical_overlap",
            "subcase": "entailment_case",
            "training_seed": 42,
            "method_tag": method,
            "checkpoint_hash": checkpoint_hash,
        },
        {
            "pair_id": f"{method}-non-entailment",
            "gold_label": "non-entailment",
            "predicted_label": "non-entailment",
            "entailment_probability": 0.1,
            "heuristic": "subsequence",
            "subcase": "non_entailment_case",
            "training_seed": 42,
            "method_tag": method,
            "checkpoint_hash": checkpoint_hash,
        },
    ]


def _write_success_status(directory, relative_outputs):
    write_json(
        directory / "status.json",
        {
            "schema_version": "canonical_status_v1",
            "state": "success",
            "output_hashes": {
                relative: sha256_file(directory / relative) for relative in relative_outputs
            },
        },
    )


def _rehash_status(directory):
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    _write_success_status(directory, list(status["output_hashes"]))


def _manifest(method, checkpoint=None, *, protocol_hash=HASH_A, data_hash=HASH_A, environment_hash=HASH_B):
    return {
        "schema_version": "canonical_run_manifest_v1",
        "role": "standard_lora" if method == "standard_lora" else "dual_adapter_branch",
        "method_tag": method,
        "data_seed": 42,
        "hans_split_seed": 42,
        "training_seed": 42,
        "protocol_sha256": protocol_hash,
        "data_manifest_sha256": data_hash,
        "environment_manifest_sha256": environment_hash,
        "git": {"commit": "c" * 40, "dirty": False},
        "command": ["python", "run_stage2_smoke.py"],
        "shared_phase2_checkpoint": checkpoint,
        "result": {"final_checkpoint_hash": HASH_B},
    }


def _create_smoke_root(base, *, environment="NVIDIA A100-SXM4-40GB", mnli_accuracy=0.8):
    root = Path(base) / "smoke"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    write_json(manifests / "data_manifest.json", {"schema_version": "data_manifest_v1"})
    write_json(
        manifests / "environment_manifest.json",
        {"schema_version": "environment_manifest_v1", "gpu": environment, "python": "3.11"},
    )
    write_json(
        manifests / "run_matrix.json",
        {
            "schema_version": "stage2_smoke_matrix_v1",
            "training_seeds": [42],
            "condition_orders": {"42": list(PRIMARY_CONDITIONS)},
        },
    )
    data_hash = sha256_file(manifests / "data_manifest.json")
    environment_hash = sha256_file(manifests / "environment_manifest.json")
    write_json(
        root / "commands.json",
        {
            "schema_version": "stage2_smoke_commands_v1",
            "mode": "primary",
            "environment": "colab_a100",
            "argv": ["python", "run_stage2_smoke.py"],
            "expected_condition_tags": list(PRIMARY_CONDITIONS),
            "profile_name": "stage2_smoke_v1",
            "gpu_name": environment,
            "started_at": "2026-08-08T00:00:00+00:00",
        },
    )
    (root / "protocol_snapshot").mkdir()
    (root / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md").write_text(
        "# frozen\n", encoding="utf-8"
    )
    protocol_hash = sha256_file(root / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md")
    (root / "protocol_snapshot" / "protocol_sha256.txt").write_text(protocol_hash + "\n", encoding="utf-8")
    write_jsonl(
        manifests / "data_access.jsonl",
        [{"event": "dataset_access", "dataset": "hans", "split": "evaluation", "purpose": "manifest_identity_only"}],
    )

    seed = root / "seed_42"
    shared = seed / "shared_phase2"
    checkpoint = shared / "checkpoints" / "shared.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"shared checkpoint")
    checkpoint_hash = sha256_file(checkpoint)
    write_json(shared / "config.json", {"training_seed": 42, "output_dir": str(shared)})
    shared_manifest = _manifest(
        None, protocol_hash=protocol_hash, data_hash=data_hash, environment_hash=environment_hash
    )
    shared_manifest["role"] = "shared_phase2"
    shared_manifest["method_tag"] = None
    shared_manifest["result"] = {"checkpoint_sha256": checkpoint_hash}
    write_json(shared / "run_manifest.json", shared_manifest)
    write_json(shared / "shared_checkpoint.json", {"path_relative": "checkpoints/shared.pt", "sha256": checkpoint_hash})
    weights = {"0": 0.1, "1": 0.2, "2": 0.3}
    write_json(
        shared / "shared_checkpoint_metadata.json",
        {"checkpoint_role": "canonical_shared_phase2", "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hash, "class_prior_weights": weights},
    )
    write_jsonl(shared / "data_access.jsonl", [])
    (shared / "stdout.log").write_text("shared\n", encoding="utf-8")
    (shared / "stderr.log").write_text("", encoding="utf-8")
    _write_success_status(
        shared,
        ["config.json", "run_manifest.json", "shared_checkpoint.json", "shared_checkpoint_metadata.json", "data_access.jsonl", "stdout.log", "stderr.log", "checkpoints/shared.pt"],
    )

    shared_ref = {"path": str(checkpoint), "sha256": checkpoint_hash}
    for method in PRIMARY_CONDITIONS:
        run = seed / method
        run.mkdir(parents=True)
        write_json(run / "config.json", {"training_seed": 42, "method_tag": method, "output_dir": str(run)})
        manifest = _manifest(
            method,
            None if method == "standard_lora" else shared_ref,
            protocol_hash=protocol_hash,
            data_hash=data_hash,
            environment_hash=environment_hash,
        )
        write_json(run / "run_manifest.json", manifest)
        rows = _predictions(method, HASH_B)
        metrics = {
            "final": {
                "mnli": {"mnli_accuracy": mnli_accuracy},
                "hans": aggregate_hans_predictions(rows),
                "esnli": {"esnli_accuracy": 0.7},
                "anli": {"anli_accuracy": 0.6},
                "snli_hard": {"snli_hard_accuracy": 0.5},
                "wanli": {"wanli_accuracy": 0.4},
            }
        }
        write_json(run / "metrics.json", metrics)
        write_json(run / "selected_layers.json", {})
        write_jsonl(
            run / "data_access.jsonl",
            [
                {"sequence": 0, "event": "final_evaluation_start", "dataset": "hans", "split": None, "purpose": "boundary"},
                {"sequence": 1, "event": "dataset_access", "dataset": "hans", "split": "evaluation", "purpose": "final"},
            ],
        )
        write_jsonl(run / "hans_predictions.jsonl", rows)
        log = "class-prior reweighting ON: {0: 0.1, 1: 0.2, 2: 0.3}\n" if method == "class_prior_reweight" else "run\n"
        (run / "stdout.log").write_text(log, encoding="utf-8")
        (run / "stderr.log").write_text("", encoding="utf-8")
        _write_success_status(
            run,
            ["config.json", "run_manifest.json", "metrics.json", "hans_predictions.jsonl", "selected_layers.json", "data_access.jsonl", "stdout.log", "stderr.log"],
        )
    return root


class Stage2ValidationTest(unittest.TestCase):
    def test_validator_recomputes_hans_and_matches_metrics_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "nan")
            canonical = Path(tmp) / "canonical_v1"

            report = validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=canonical)

            self.assertEqual("pass", report["checks"]["hans_recomputation"]["state"])
            self.assertEqual("pass", report["checks"]["audit_order"]["state"])

    def test_validator_rejects_evaluation_access_before_final_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "tampered")
            write_jsonl(
                root / "seed_42" / "standard_lora" / "data_access.jsonl",
                [
                    {"event": "dataset_access", "dataset": "hans", "split": "evaluation", "purpose": "final"},
                    {"event": "final_evaluation_start", "dataset": "hans", "split": None, "purpose": "boundary"},
                ],
            )
            _rehash_status(root / "seed_42" / "standard_lora")
            with self.assertRaisesRegex(ValueError, "before final_evaluation_start"):
                validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

    def test_validator_rejects_nan_json_and_tampered_success_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "binding")
            metrics = root / "seed_42" / "standard_lora" / "metrics.json"
            metrics.write_text('{"final": NaN}\n', encoding="utf-8")
            _rehash_status(root / "seed_42" / "standard_lora")
            with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

            root = _create_smoke_root(Path(tmp) / "log")
            (root / "seed_42" / "full_sr" / "stdout.log").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

    def test_validator_rejects_prediction_manifest_binding_and_bad_class_prior_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "binding")
            prediction_path = root / "seed_42" / "full_sr" / "hans_predictions.jsonl"
            rows = json.loads("[" + ",".join(prediction_path.read_text(encoding="utf-8").splitlines()) + "]")
            rows[0]["method_tag"] = "standard_lora"
            write_jsonl(prediction_path, rows)
            _rehash_status(root / "seed_42" / "full_sr")
            with self.assertRaisesRegex(ValueError, "method_tag"):
                validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

            root = _create_smoke_root(Path(tmp) / "log")
            (root / "seed_42" / "class_prior_reweight" / "stdout.log").write_text("class-prior reweighting ON\n", encoding="utf-8")
            _rehash_status(root / "seed_42" / "class_prior_reweight")
            with self.assertRaisesRegex(ValueError, "class-prior"):
                validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

    def test_repeat_tolerance_is_inclusive_at_half_percentage_point(self):
        report = compare_metric_values(0.400, 0.405, tolerance=0.005)
        self.assertTrue(report["within_tolerance"])
        self.assertAlmostEqual(0.005, report["absolute_difference"])

    def test_repeat_comparison_uses_hans_non_entailment_and_requires_frozen_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary = _create_smoke_root(Path(tmp) / "primary", mnli_accuracy=0.80)
            repeat = _create_smoke_root(Path(tmp) / "repeat", mnli_accuracy=0.70)
            report = compare_a100_repeat(primary, repeat)
            self.assertTrue(report["primary_metric"]["within_tolerance"])
            self.assertEqual(0.80, report["mnli_diagnostic"]["primary"])
            changed = json.loads((repeat / "commands.json").read_text(encoding="utf-8"))
            changed["gpu_name"] = "NVIDIA T4"
            write_json(repeat / "commands.json", changed)
            with self.assertRaisesRegex(ValueError, "gpu"):
                compare_a100_repeat(primary, repeat)

    def test_repeat_comparison_rejects_a_manifest_not_bound_to_its_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary = _create_smoke_root(Path(tmp) / "primary")
            repeat = _create_smoke_root(Path(tmp) / "repeat")
            manifest_path = repeat / "seed_42" / "full_sr" / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["protocol_sha256"] = HASH_A
            write_json(manifest_path, manifest)
            _rehash_status(repeat / "seed_42" / "full_sr")

            with self.assertRaisesRegex(ValueError, "does not bind"):
                compare_a100_repeat(primary, repeat)

    def test_cli_writes_reports_without_importing_ml_dependencies_for_help(self):
        help_result = subprocess.run(
            [sys.executable, "validate_stage2_smoke.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, help_result.returncode, help_result.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(tmp)
            result = subprocess.run(
                [sys.executable, "validate_stage2_smoke.py", "--root", str(root), "--conditions", *PRIMARY_CONDITIONS],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue((root / "stage2_validation.json").is_file())
            self.assertTrue((root / "stage2_validation.md").is_file())


if __name__ == "__main__":
    unittest.main()
