import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.smoke import (
    PRIMARY_CONDITIONS,
    REPEAT_CONDITIONS,
    SMOKE_PROFILE_NAME,
    assert_stage2_output_path,
    build_smoke_config,
)
from canonical.artifacts import sha256_file, write_json, write_jsonl
from canonical.runner import CheckpointRef, run_condition_matrix
from configs.config import TrainConfig


CLEAN_GIT = {
    "commit": "f" * 40,
    "branch": "test",
    "dirty": False,
    "status_porcelain": [],
}


class MatrixFakeBackend:
    def __init__(self):
        self.prepared = []
        self.methods = []

    def initialize_manifests(self, output_dir, protocol_path):
        manifests = Path(output_dir) / "manifests"
        write_json(manifests / "data_manifest.json", {"schema_version": "data_manifest_v1"})
        write_json(
            manifests / "environment_manifest.json",
            {"schema_version": "environment_manifest_v1"},
        )

    def prepare_shared(self, training_seed, shared_dir):
        self.prepared.append(training_seed)
        shared_dir = Path(shared_dir)
        write_json(shared_dir / "config.json", {"training_seed": training_seed})
        checkpoint = shared_dir / "checkpoints" / "shared.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"shared-{training_seed}".encode("ascii"))
        checkpoint_hash = sha256_file(checkpoint)
        write_json(
            shared_dir / "shared_checkpoint_metadata.json",
            {"checkpoint_sha256": checkpoint_hash, "class_prior_weights": {"0": 0.1}},
        )
        write_jsonl(shared_dir / "data_access.jsonl", [])
        return CheckpointRef(checkpoint, checkpoint_hash)

    def _write_method_artifacts(self, condition, training_seed, run_dir, checkpoint):
        self.methods.append((training_seed, condition.tag, checkpoint))
        run_dir = Path(run_dir)
        write_json(run_dir / "config.json", {"method_tag": condition.tag})
        write_json(run_dir / "metrics.json", {})
        write_json(run_dir / "selected_layers.json", {})
        write_jsonl(run_dir / "data_access.jsonl", [])
        write_jsonl(run_dir / "hans_predictions.jsonl", [])
        return {"final_checkpoint_hash": "a" * 64}

    def run_standard(self, condition, training_seed, run_dir):
        return self._write_method_artifacts(condition, training_seed, run_dir, None)

    def run_branch(self, condition, training_seed, run_dir, checkpoint):
        return self._write_method_artifacts(condition, training_seed, run_dir, checkpoint)


class Stage2SmokeProfileTest(unittest.TestCase):
    def test_profile_has_exact_frozen_budget_without_changing_core_defaults(self):
        core = TrainConfig()
        smoke = build_smoke_config(Path("out"))

        self.assertEqual((core.mnli_train_size, core.mnli_val_size), (100_000, 5_000))
        self.assertIsNone(core.hans_eval_size)
        self.assertIsNone(core.esnli_eval_size)
        self.assertIsNone(core.anli_eval_size)
        self.assertIsNone(core.snli_hard_eval_size)
        self.assertIsNone(core.wanli_eval_size)
        self.assertIsNone(core.data_access_log)
        self.assertEqual(SMOKE_PROFILE_NAME, "stage2_smoke_v1")
        self.assertEqual((smoke.mnli_train_size, smoke.mnli_val_size), (96, 96))
        self.assertEqual((smoke.batch_size, smoke.max_seq_length), (8, 64))
        self.assertEqual(
            (smoke.phase1_epochs, smoke.phase2_epochs, smoke.phase3_epochs), (1, 1, 1)
        )
        self.assertEqual(smoke.phase2_epoch_batches, 4)
        self.assertEqual(smoke.hans_eval_size, 384)
        self.assertEqual(smoke.esnli_eval_size, 128)
        self.assertEqual(smoke.anli_eval_size, 128)
        self.assertEqual(smoke.snli_hard_eval_size, 128)
        self.assertEqual(smoke.wanli_eval_size, 128)
        self.assertEqual(PRIMARY_CONDITIONS, ("standard_lora", "full_sr", "class_prior_reweight"))
        self.assertEqual(REPEAT_CONDITIONS, ("full_sr",))

    def test_smoke_rejects_any_canonical_v1_path_component(self):
        with self.assertRaisesRegex(ValueError, "canonical_v1"):
            assert_stage2_output_path(Path("ties_results/canonical_v1/run"), Path.cwd())

    def test_smoke_accepts_a_noncanonical_stage2_output_path(self):
        output_dir = assert_stage2_output_path(Path("ties_results/stage2_smoke/run"), Path.cwd())

        self.assertEqual(output_dir, (Path.cwd() / "ties_results/stage2_smoke/run").resolve())

    def test_stage2_primary_executes_only_three_methods_and_one_shared_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = root / "FROZEN_EXPERIMENT_PROTOCOL.md"
            protocol.write_text("# frozen\n", encoding="utf-8")
            output = root / "stage2_smoke"
            backend = MatrixFakeBackend()

            result = run_condition_matrix(
                protocol,
                output,
                backend,
                fresh=True,
                seeds=(42,),
                condition_tags=PRIMARY_CONDITIONS,
                matrix_schema_version="stage2_smoke_matrix_v1",
                git_metadata=CLEAN_GIT,
            )

            self.assertEqual(backend.prepared, [42])
            self.assertEqual(
                [tag for _, tag, _ in backend.methods], list(PRIMARY_CONDITIONS)
            )
            self.assertEqual(3, len(result["executed"]))
            matrix = json.loads((output / "manifests" / "run_matrix.json").read_text(encoding="utf-8"))
            self.assertEqual("stage2_smoke_matrix_v1", matrix["schema_version"])
            self.assertEqual(list(PRIMARY_CONDITIONS), matrix["condition_orders"]["42"])

    def test_stage2_smoke_help_does_not_need_ml_dependencies(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_stage2_smoke.py"), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("--environment", result.stdout)


if __name__ == "__main__":
    unittest.main()
