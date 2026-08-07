import sys
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
from configs.config import TrainConfig


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


if __name__ == "__main__":
    unittest.main()
