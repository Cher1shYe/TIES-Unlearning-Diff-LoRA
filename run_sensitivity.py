"""One-At-a-Time (OAT) sensitivity analysis for TIES-Unlearning Diff-LoRA.

Directly responds to reviewer item #10:
    "Sensitivity Analysis: Study key hyperparameters systematically.
     Suggest: analyze sensitivity to r_P, r_N, alpha, beta, trimming ratio,
     the MNLI/HANS mixture ratio, the number of selected layers, the
     negative-branch learning-rate multiplier, and the choice of target modules."

For each parameter we keep all other hyperparameters at their TrainConfig defaults
and sweep that single parameter across a small grid. The default value (the
"anchor" run) appears in every grid, so we de-duplicate it and run it only once.

Each run writes trainer metrics to <output_dir>/<tag>/metrics.json; the loop pulls
Phase-3 MNLI / e-SNLI / HANS metrics back out. After all runs, sensitivity_summary.json
is written (one record per (param, value), with the anchor replicated into each sweep
so plot_sensitivity.py can draw it as a fixed reference line).

Usage:
    python run_sensitivity.py                       # full scale
    python run_sensitivity.py --small               # short runs (Colab-friendly)
    python run_sensitivity.py --only pos_rank neg_rank target_modules
    python run_sensitivity.py --output-dir ./sens_out
"""
import argparse
import json
import os
import sys
import traceback
from copy import deepcopy
from dataclasses import asdict, fields
from typing import Any, Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import TrainConfig
from training.trainer import train_ties_unlearn


# Every parameter named in reviewer item #10. target_modules is categorical (a tuple);
# all others are numeric.
PARAM_GRID: Dict[str, List[Any]] = {
    "pos_rank":              [8, 16, 32],
    "neg_rank":              [2, 4, 8],
    "alpha":                 [1.0, 1.25, 2.0],
    "beta":                  [0.3, 0.5, 0.7],
    "trim_ratio":            [0.1, 0.2, 0.3],
    "phase2_mnli_mix_ratio": [0.0, 0.1, 0.2],
    "layer_selection_topk":  [2, 4, 6],
    "neg_lr_mult":           [1.0, 2.0, 3.0],
    "target_modules":        [("query", "value"),
                              ("query", "key", "value"),
                              ("query", "key", "value", "output.dense")],
}


def _value_label(value: Any) -> str:
    """Filesystem-friendly label for a parameter value (handles tuple target_modules)."""
    if isinstance(value, (tuple, list)):
        return "-".join(str(v) for v in value)
    return str(value)


def _apply_small_overrides(cfg: TrainConfig) -> TrainConfig:
    """Same shrinkage scheme as run_baselines.py --small, so the suites stay comparable."""
    cfg.mnli_train_size = 20_000
    cfg.mnli_val_size = 2_000
    cfg.phase1_epochs = 2
    cfg.phase2_epochs = 1
    cfg.phase3_epochs = 2
    cfg.phase2_epoch_batches = 625
    cfg.knn_ref_mnli = 800
    cfg.knn_ref_hans_entail = 400
    cfg.knn_ref_hans_non_entail = 400
    cfg.knn_query_mnli = 200
    cfg.knn_query_hans_entail = 300
    cfg.knn_query_hans_non_entail = 300
    return cfg


def _make_base_cfg(small: bool, output_dir: str) -> TrainConfig:
    cfg = TrainConfig(
        run_baseline=False, run_jtt=False,
        save_checkpoints=False, save_checkpoints_per_phase=False,
        output_dir=output_dir,
    )
    if small:
        cfg = _apply_small_overrides(cfg)
    return cfg


def _build_run_list(base_cfg: TrainConfig,
                    grid: Dict[str, List[Any]],
                    only: List[str]) -> Tuple[List[Tuple[str, Any]], Tuple[str, Any]]:
    """Return (unique_runs, anchor) where unique_runs is the list of (param, value)
    that need a training run. The anchor (all defaults) appears exactly once."""
    base_dict = asdict(base_cfg)
    variant_runs: List[Tuple[str, Any]] = []
    for p, values in grid.items():
        if only and p not in only:
            continue
        default_val = base_dict[p]
        for v in values:
            if v == default_val:
                continue  # collapsed into the single anchor run
            variant_runs.append((p, v))
    anchor = ("anchor", "default")
    return [anchor] + variant_runs, anchor


def _extract_metrics(metrics: Dict) -> Dict[str, float]:
    p3 = metrics.get("phase3", {})
    mnli, hans, esnli = p3.get("mnli", {}), p3.get("hans", {}), p3.get("esnli", {})
    return {
        "mnli_accuracy":       float(mnli.get("mnli_accuracy", float("nan"))),
        "esnli_accuracy":      float(esnli.get("esnli_accuracy", float("nan"))),
        "hans_overall":        float(hans.get("hans_overall", float("nan"))),
        "hans_entailment":     float(hans.get("hans_entailment", float("nan"))),
        "hans_non_entailment": float(hans.get("hans_non_entailment", float("nan"))),
    }


def _run_one(base_cfg: TrainConfig, param: str, value: Any, output_dir: str) -> Dict:
    cfg = deepcopy(base_cfg)

    if param != "anchor":
        if not any(f.name == param for f in fields(cfg)):
            raise ValueError(f"Unknown TrainConfig field: {param}")
        setattr(cfg, param, value)
        # kl_topk_candidates must be >= layer_selection_topk (TrainConfig invariant).
        if param == "layer_selection_topk" and value > cfg.kl_topk_candidates:
            cfg.kl_topk_candidates = value

    tag = f"{param}_{_value_label(value)}"
    cfg.experiment_name = tag
    cfg.output_dir = output_dir

    print("\n" + "=" * 70 + f"\n[Sensitivity] {tag}\n" + "=" * 70)

    record = {"param": param, "value": value, "value_label": _value_label(value), "tag": tag}
    try:
        metrics = train_ties_unlearn(cfg)
        record.update(_extract_metrics(metrics))
        record["status"] = "ok"
    except Exception as e:
        record["status"] = "error"
        record["error"] = repr(e)
        record["traceback"] = traceback.format_exc()
        print(f"[Sensitivity] ERROR on {tag}: {e}")
        metric_file = os.path.join(output_dir, tag, "metrics.json")
        if os.path.exists(metric_file):
            with open(metric_file, "r", encoding="utf-8") as f:
                record.update(_extract_metrics(json.load(f)))
    return record


def _expand_anchor_for_plotting(records: List[Dict],
                                grid: Dict[str, List[Any]],
                                base_cfg: TrainConfig,
                                only: List[str]) -> List[Dict]:
    """The anchor was trained once; emit one virtual record per sweep so each
    parameter's chart has a point at its default value."""
    anchor = next((r for r in records if r["param"] == "anchor"), None)
    if anchor is None:
        return records

    base_dict = asdict(base_cfg)
    expanded = [r for r in records if r["param"] != "anchor"]
    for p, values in grid.items():
        if only and p not in only:
            continue
        default_val = base_dict[p]
        if default_val in values:
            virt = dict(anchor)
            virt["param"] = p
            virt["value"] = default_val
            virt["value_label"] = _value_label(default_val)
            virt["tag"] = f"{p}_{_value_label(default_val)}"
            virt["from_anchor"] = True
            expanded.append(virt)
    return expanded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./sensitivity_results")
    parser.add_argument("--small", action="store_true",
                        help="Use a reduced training budget (Colab-friendly).")
    parser.add_argument("--only", nargs="+", default=[],
                        choices=list(PARAM_GRID.keys()),
                        help="Restrict the sweep to these parameters.")
    args = parser.parse_args()

    base_cfg = _make_base_cfg(small=args.small, output_dir=args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    runs, _ = _build_run_list(base_cfg, PARAM_GRID, args.only)
    print(f"\n[Sensitivity] output_dir = {args.output_dir}")
    print(f"[Sensitivity] small mode  = {args.small}")
    print(f"[Sensitivity] only        = {args.only or 'all %d params' % len(PARAM_GRID)}")
    print(f"[Sensitivity] {len(runs)} unique runs queued (1 anchor + {len(runs)-1} variants)")

    all_records: List[Dict] = []
    for i, (param, value) in enumerate(runs, 1):
        print(f"\n>>> Run {i}/{len(runs)}: {param}={value}")
        all_records.append(_run_one(base_cfg, param, value, args.output_dir))
        # Save incrementally so a crash doesn't lose finished runs.
        with open(os.path.join(args.output_dir, "sensitivity_summary.json"), "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2, ensure_ascii=False)

    expanded = _expand_anchor_for_plotting(all_records, PARAM_GRID, base_cfg, args.only)
    with open(os.path.join(args.output_dir, "sensitivity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(expanded, f, indent=2, ensure_ascii=False)

    print(f"\n[Sensitivity] Done. {len(all_records)} runs executed, "
          f"{len(expanded)} records written.")
    print("[Sensitivity] Next: `python plot_sensitivity.py` to draw the curves.")


if __name__ == "__main__":
    main()
