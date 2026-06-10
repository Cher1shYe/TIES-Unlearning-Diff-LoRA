import math
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

training_pkg = types.ModuleType("training")
training_pkg.__path__ = []
trainer_module = types.ModuleType("training.trainer")
trainer_module.train_ties_unlearn = lambda _cfg: {}
sys.modules.setdefault("training", training_pkg)
sys.modules["training.trainer"] = trainer_module

import run_sensitivity as rs
import finish_sensitivity as fs


def _metric_block(value):
    return {
        "mnli": {"mnli_accuracy": value},
        "esnli": {"esnli_accuracy": value + 0.01},
        "anli": {"anli_accuracy": value + 0.02},
        "snli_hard": {"snli_hard_accuracy": value + 0.03},
        "wanli": {"wanli_accuracy": value + 0.04},
        "hans": {
            "hans_overall": value + 0.05,
            "hans_entailment": value + 0.06,
            "hans_non_entailment": value + 0.07,
        },
    }


class RunSensitivityRankControlsTest(unittest.TestCase):
    def _base_cfg(self):
        return rs._make_base_cfg(small=True, output_dir=tempfile.mkdtemp())

    def test_default_run_list_includes_rank_controls_with_branch_eval(self):
        runs, _ = rs._build_run_list(
            self._base_cfg(),
            rs.PARAM_GRID,
            only=[],
            skip_rank_controls=False,
        )

        by_tag = {run.tag: run for run in runs}
        self.assertIn("anchor_default", by_tag)
        self.assertTrue(by_tag["anchor_default"].overrides["record_branch_only_metrics"])

        rank_controls = [run for run in runs if run.group == "rank_control"]
        self.assertEqual(
            {"equal_rank_low", "equal_rank_high", "reversed_rank_default"},
            {run.value for run in rank_controls},
        )
        self.assertTrue(
            all(run.overrides["record_branch_only_metrics"] for run in rank_controls)
        )

    def test_only_specific_parameter_excludes_rank_controls(self):
        runs, _ = rs._build_run_list(
            self._base_cfg(),
            rs.PARAM_GRID,
            only=["pos_rank"],
            skip_rank_controls=False,
        )

        self.assertFalse(any(run.group == "rank_control" for run in runs))
        self.assertTrue(all(run.param in {"anchor", "pos_rank"} for run in runs))
        anchor = next(run for run in runs if run.param == "anchor")
        self.assertFalse(anchor.overrides.get("record_branch_only_metrics", False))

    def test_extract_metrics_flattens_final_branch_only_evaluations(self):
        metrics = {
            "phase3": _metric_block(0.10),
            "branch_only": {
                "p_only": _metric_block(0.20),
                "n_only": _metric_block(0.30),
            },
        }

        extracted = rs._extract_metrics(metrics)

        self.assertEqual("final_branch_only", extracted["branch_eval_source"])
        self.assertTrue(math.isclose(extracted["mnli_accuracy"], 0.10))
        self.assertTrue(math.isclose(extracted["p_branch_mnli_accuracy"], 0.20))
        self.assertTrue(math.isclose(extracted["p_branch_hans_non_entailment"], 0.27))
        self.assertTrue(math.isclose(extracted["n_branch_mnli_accuracy"], 0.30))
        self.assertTrue(math.isclose(extracted["n_branch_hans_non_entailment"], 0.37))

    def test_finish_sensitivity_accepts_rank_controls_only_resume(self):
        args = fs._parse_args([
            "--assemble-only",
            "--only",
            "rank_controls",
            "--output-dir",
            "unused",
        ])

        self.assertEqual(["rank_controls"], args.only)


if __name__ == "__main__":
    unittest.main()
