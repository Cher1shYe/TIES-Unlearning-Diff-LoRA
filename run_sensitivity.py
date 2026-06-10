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
    python run_sensitivity.py --only rank_controls  # equal/reversed ranks + branch-only eval
    python run_sensitivity.py --skip-rank-controls  # original OAT grid only
    python run_sensitivity.py --output-dir ./sens_out
"""
import argparse
import json
import os
import sys
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import TrainConfig


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


@dataclass(frozen=True)
class RunSpec:
    """Concrete run description for an OAT sweep or a multi-field control."""

    param: str
    value: Any
    tag: str
    overrides: Dict[str, Any]
    group: str = "sensitivity"
    control_type: str = ""


RANK_CONTROLS: List[Dict[str, Any]] = [
    {
        "value": "default_differential",
        "control_type": "rank_differential_default",
        "overrides": {},
    },
    {
        "value": "equal_rank_low",
        "control_type": "equal_rank",
        "overrides": {"pos_rank": 4, "neg_rank": 4},
    },
    {
        "value": "equal_rank_high",
        "control_type": "equal_rank",
        "overrides": {"pos_rank": 16, "neg_rank": 16},
    },
    {
        "value": "reversed_rank_default",
        "control_type": "reversed_rank",
        "overrides": {"pos_rank": 4, "neg_rank": 16},
    },
]


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
                    only: List[str],
                    skip_rank_controls: bool = False) -> Tuple[List[RunSpec], RunSpec]:
    """Return (unique_runs, anchor), with the anchor trained exactly once."""
    base_dict = asdict(base_cfg)
    include_rank_controls = (not skip_rank_controls) and (not only or "rank_controls" in only)
    anchor_overrides: Dict[str, Any] = {}
    if include_rank_controls:
        anchor_overrides["record_branch_only_metrics"] = True

    anchor = RunSpec(
        param="anchor",
        value="default",
        tag="anchor_default",
        overrides=anchor_overrides,
        group="anchor",
    )
    runs: List[RunSpec] = [anchor]

    for p, values in grid.items():
        if only and p not in only:
            continue
        default_val = base_dict[p]
        for v in values:
            if v == default_val:
                continue  # collapsed into the single anchor run
            runs.append(
                RunSpec(
                    param=p,
                    value=v,
                    tag=f"{p}_{_value_label(v)}",
                    overrides={p: v},
                )
            )

    if include_rank_controls:
        for control in RANK_CONTROLS:
            overrides = dict(control["overrides"])
            if not overrides:
                continue  # represented by the shared anchor in the expanded summary
            overrides["record_branch_only_metrics"] = True
            runs.append(
                RunSpec(
                    param="rank_control",
                    value=control["value"],
                    tag=f"rank_control_{control['value']}",
                    overrides=overrides,
                    group="rank_control",
                    control_type=control["control_type"],
                )
            )

    return runs, anchor


def _rank_relation(pos_rank: int, neg_rank: int) -> str:
    if pos_rank == neg_rank:
        return "equal"
    if pos_rank < neg_rank:
        return "reversed"
    return "rank_differential"


def _rank_control_metadata(base_cfg: TrainConfig,
                           overrides: Dict[str, Any],
                           control_type: str) -> Dict[str, Any]:
    pos_rank = int(overrides.get("pos_rank", base_cfg.pos_rank))
    neg_rank = int(overrides.get("neg_rank", base_cfg.neg_rank))
    return {
        "control_type": control_type,
        "pos_rank": pos_rank,
        "neg_rank": neg_rank,
        "rank_relation": _rank_relation(pos_rank, neg_rank),
    }


def _record_for_run(run: RunSpec, base_cfg: TrainConfig) -> Dict:
    record = {
        "param": run.param,
        "value": run.value,
        "value_label": _value_label(run.value),
        "tag": run.tag,
        "group": run.group,
    }
    if run.group == "rank_control":
        record.update(_rank_control_metadata(base_cfg, run.overrides, run.control_type))
    return record


def _extract_metrics(metrics: Dict) -> Dict[str, float]:
    def flatten(block: Dict, prefix: str = "") -> Dict[str, float]:
        mnli, hans, esnli = block.get("mnli", {}), block.get("hans", {}), block.get("esnli", {})
        anli, snli_hard = block.get("anli", {}), block.get("snli_hard", {})
        wanli = block.get("wanli", {})
        return {
            f"{prefix}mnli_accuracy":       float(mnli.get("mnli_accuracy", float("nan"))),
            f"{prefix}esnli_accuracy":      float(esnli.get("esnli_accuracy", float("nan"))),
            f"{prefix}anli_accuracy":       float(anli.get("anli_accuracy", float("nan"))),
            f"{prefix}snli_hard_accuracy":  float(snli_hard.get("snli_hard_accuracy", float("nan"))),
            f"{prefix}wanli_accuracy":      float(wanli.get("wanli_accuracy", float("nan"))),
            f"{prefix}hans_overall":        float(hans.get("hans_overall", float("nan"))),
            f"{prefix}hans_entailment":     float(hans.get("hans_entailment", float("nan"))),
            f"{prefix}hans_non_entailment": float(hans.get("hans_non_entailment", float("nan"))),
        }

    out = flatten(metrics.get("phase3", {}))
    branch = metrics.get("branch_only", {})
    if branch:
        out.update(flatten(branch.get("p_only", {}), "p_branch_"))
        out.update(flatten(branch.get("n_only", {}), "n_branch_"))
        out["branch_eval_source"] = "final_branch_only"
    elif metrics.get("phase1") or metrics.get("phase2"):
        out.update(flatten(metrics.get("phase1", {}), "p_branch_"))
        out.update(flatten(metrics.get("phase2", {}), "n_branch_"))
        out["branch_eval_source"] = "phase1_phase2"
    return out


def _apply_overrides(cfg: TrainConfig, overrides: Dict[str, Any]) -> TrainConfig:
    valid_fields = {f.name for f in fields(cfg)}
    for key, val in overrides.items():
        if key not in valid_fields:
            raise ValueError(f"Unknown TrainConfig field: {key}")
        setattr(cfg, key, val)
    if cfg.layer_selection_topk > cfg.kl_topk_candidates:
        cfg.kl_topk_candidates = cfg.layer_selection_topk
    return cfg


def _run_one(base_cfg: TrainConfig, run: RunSpec, output_dir: str) -> Dict:
    cfg = _apply_overrides(deepcopy(base_cfg), run.overrides)
    cfg.experiment_name = run.tag
    cfg.output_dir = output_dir

    print("\n" + "=" * 70 + f"\n[Sensitivity] {run.tag}\n" + "=" * 70)

    record = _record_for_run(run, base_cfg)
    try:
        from training.trainer import train_ties_unlearn

        metrics = train_ties_unlearn(cfg)
        record.update(_extract_metrics(metrics))
        record["status"] = "ok"
    except Exception as e:
        record["status"] = "error"
        record["error"] = repr(e)
        record["traceback"] = traceback.format_exc()
        print(f"[Sensitivity] ERROR on {run.tag}: {e}")
        metric_file = os.path.join(output_dir, run.tag, "metrics.json")
        if os.path.exists(metric_file):
            with open(metric_file, "r", encoding="utf-8") as f:
                record.update(_extract_metrics(json.load(f)))
    return record


def _fmt_pct(x: float) -> str:
    return "n/a" if x != x else f"{x * 100:.2f}%"


def _write_rank_control_table(records: List[Dict], output_dir: str) -> None:
    rows = [
        r for r in records
        if r.get("group") == "rank_control" and r.get("status", "ok") == "ok"
    ]
    if not rows:
        return

    order = {c["value"]: i for i, c in enumerate(RANK_CONTROLS)}
    rows.sort(key=lambda r: order.get(r.get("value"), 999))

    lines = [
        "# Rank-control sensitivity",
        "",
        "| Control | r_P | r_N | Relation | Merged MNLI | Merged HANS non-ent | "
        "P-only MNLI | P-only HANS non-ent | N-only MNLI | N-only HANS non-ent |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        mark = " *(default)*" if r.get("from_anchor") else ""
        lines.append(
            f"| {r.get('value_label', r['value'])}{mark} "
            f"| {r.get('pos_rank', '')} | {r.get('neg_rank', '')} "
            f"| {r.get('rank_relation', '')} "
            f"| {_fmt_pct(r.get('mnli_accuracy', float('nan')))} "
            f"| {_fmt_pct(r.get('hans_non_entailment', float('nan')))} "
            f"| {_fmt_pct(r.get('p_branch_mnli_accuracy', float('nan')))} "
            f"| {_fmt_pct(r.get('p_branch_hans_non_entailment', float('nan')))} "
            f"| {_fmt_pct(r.get('n_branch_mnli_accuracy', float('nan')))} "
            f"| {_fmt_pct(r.get('n_branch_hans_non_entailment', float('nan')))} |"
        )

    with open(os.path.join(output_dir, "rank_control_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _expand_anchor_for_plotting(records: List[Dict],
                                grid: Dict[str, List[Any]],
                                base_cfg: TrainConfig,
                                only: List[str],
                                skip_rank_controls: bool = False) -> List[Dict]:
    """Replicate the shared anchor into each OAT group and the default rank control."""
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
            virt["group"] = "sensitivity"
            virt["from_anchor"] = True
            expanded.append(virt)

    include_rank_controls = (not skip_rank_controls) and (not only or "rank_controls" in only)
    if include_rank_controls:
        default_control = RANK_CONTROLS[0]
        virt = dict(anchor)
        virt["param"] = "rank_control"
        virt["value"] = default_control["value"]
        virt["value_label"] = default_control["value"]
        virt["tag"] = f"rank_control_{default_control['value']}"
        virt["group"] = "rank_control"
        virt["from_anchor"] = True
        virt.update(_rank_control_metadata(base_cfg, {}, default_control["control_type"]))
        expanded.append(virt)

    return expanded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./sensitivity_results")
    parser.add_argument("--small", action="store_true",
                        help="Use a reduced training budget (Colab-friendly).")
    parser.add_argument("--only", nargs="+", default=[],
                        choices=list(PARAM_GRID.keys()) + ["rank_controls"],
                        help="Restrict the sweep to these parameters/control families.")
    parser.add_argument("--skip-rank-controls", action="store_true",
                        help="Run only the original OAT grid, without equal/reversed rank controls.")
    args = parser.parse_args()

    base_cfg = _make_base_cfg(small=args.small, output_dir=args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    runs, _ = _build_run_list(
        base_cfg,
        PARAM_GRID,
        args.only,
        skip_rank_controls=args.skip_rank_controls,
    )
    print(f"\n[Sensitivity] output_dir = {args.output_dir}")
    print(f"[Sensitivity] small mode  = {args.small}")
    print(f"[Sensitivity] only        = {args.only or 'all %d params' % len(PARAM_GRID)}")
    print(f"[Sensitivity] rank ctrls  = {not args.skip_rank_controls and (not args.only or 'rank_controls' in args.only)}")
    print(f"[Sensitivity] {len(runs)} unique runs queued (1 anchor + {len(runs)-1} variants)")

    all_records: List[Dict] = []
    for i, run in enumerate(runs, 1):
        print(f"\n>>> Run {i}/{len(runs)}: {run.tag}")
        all_records.append(_run_one(base_cfg, run, args.output_dir))
        # Save incrementally so a crash doesn't lose finished runs.
        with open(os.path.join(args.output_dir, "sensitivity_summary.json"), "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2, ensure_ascii=False)

    expanded = _expand_anchor_for_plotting(
        all_records,
        PARAM_GRID,
        base_cfg,
        args.only,
        skip_rank_controls=args.skip_rank_controls,
    )
    with open(os.path.join(args.output_dir, "sensitivity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(expanded, f, indent=2, ensure_ascii=False)
    _write_rank_control_table(expanded, args.output_dir)

    print(f"\n[Sensitivity] Done. {len(all_records)} runs executed, "
          f"{len(expanded)} records written.")
    print("[Sensitivity] Next: `python plot_sensitivity.py` to draw the curves.")


if __name__ == "__main__":
    main()
