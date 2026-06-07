"""Baseline experiment driver for TIES-Unlearning Diff-LoRA.

Addresses reviewer item #7 ("compare against stronger and more relevant debiasing
methods"). Runs each baseline, collects MNLI / HANS / e-SNLI metrics, and writes a
unified comparison table.

Methods:
  1. standard_lora  : plain single-path LoRA                       (training/baseline.py)
  2. jtt            : Just Train Twice                             (training/train_jtt.py)
  3. poe            : Product-of-Experts w/ hypothesis-only bias   (training/train_poe.py)
  4. zfilter        : data-centric bias-aligned example filtering  (training/train_zfilter.py)
  5. negmerge       : global sign-consensus merge (NegMerge-style) (training/train_negmerge.py)
  6. naive_subtract : GLOBAL naive task-vector subtraction, no masks, no Phase-3 FT
                      (merge_mode="naive"). NOTE: the previous version of this script
                      set no_ties_ablation=True, which actually produced a P-only model
                      (no subtraction at all) — that bug is fixed here.
  7. ties_full      : the full TIES-Unlearning Diff-LoRA pipeline  (training/trainer.py)

Usage:
    python run_baselines.py                     # full scale
    python run_baselines.py --small             # short run, sanity-check / Colab
    python run_baselines.py --skip jtt poe
    python run_baselines.py --only ties_full negmerge
"""
import argparse
import json
import os
import sys
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import TrainConfig
from training.baseline import train_single_lora_baseline
from training.train_jtt import train_jtt_baseline
from training.train_poe import train_poe_baseline
from training.train_zfilter import train_zfilter_baseline
from training.train_negmerge import train_negmerge_baseline
from training.trainer import train_ties_unlearn


METHOD_TAGS = [
    "standard_lora", "jtt", "poe", "zfilter", "negmerge", "naive_subtract", "ties_full",
]


def _apply_small_overrides(cfg: TrainConfig) -> TrainConfig:
    """Shrink training to be feasible on a free Colab GPU while keeping the pipeline
    behaviour representative (relative phase ratios preserved)."""
    cfg.mnli_train_size = 20_000
    cfg.mnli_val_size = 2_000
    cfg.phase1_epochs = 2
    cfg.phase2_epochs = 1
    cfg.phase3_epochs = 2
    cfg.phase2_epoch_batches = 625
    cfg.bias_model_epochs = 1
    cfg.knn_ref_mnli = 800
    cfg.knn_ref_hans_entail = 400
    cfg.knn_ref_hans_non_entail = 400
    cfg.knn_query_mnli = 200
    cfg.knn_query_hans_entail = 300
    cfg.knn_query_hans_non_entail = 300
    return cfg


def _make_base_cfg(small: bool, output_dir: str, hans_clean_split: bool = True) -> TrainConfig:
    cfg = TrainConfig(
        run_baseline=False,
        run_jtt=False,
        save_checkpoints=False,
        save_checkpoints_per_phase=False,
        output_dir=output_dir,
        hans_clean_split=hans_clean_split,
    )
    if small:
        cfg = _apply_small_overrides(cfg)
    return cfg


def _normalize_metrics(method: str, raw: Dict) -> Dict:
    """Flatten the TIES (phase3) vs flat baseline metric shapes into one schema."""
    if "phase3" in raw:
        mnli = raw["phase3"]["mnli"]
        hans = raw["phase3"]["hans"]
        esnli = raw["phase3"].get("esnli", {})
        anli = raw["phase3"].get("anli", {})
        snli_hard = raw["phase3"].get("snli_hard", {})
        wanli = raw["phase3"].get("wanli", {})
    else:
        mnli = raw["mnli"]
        hans = raw["hans"]
        esnli = raw.get("esnli", {})
        anli = raw.get("anli", {})
        snli_hard = raw.get("snli_hard", {})
        wanli = raw.get("wanli", {})
    return {
        "method": method,
        "mnli_accuracy": float(mnli["mnli_accuracy"]),
        "esnli_accuracy": float(esnli.get("esnli_accuracy", float("nan"))),
        "anli_accuracy": float(anli.get("anli_accuracy", float("nan"))),
        "snli_hard_accuracy": float(snli_hard.get("snli_hard_accuracy", float("nan"))),
        "wanli_accuracy": float(wanli.get("wanli_accuracy", float("nan"))),
        "hans_overall": float(hans["hans_overall"]),
        "hans_entailment": float(hans["hans_entailment"]),
        "hans_non_entailment": float(hans["hans_non_entailment"]),
        "heuristic_breakdown": hans.get("heuristic_breakdown", {}),
    }


def run_standard_lora(base_cfg: TrainConfig) -> Dict:
    cfg = deepcopy(base_cfg); cfg.experiment_name = "standard_lora"
    print("\n" + "#" * 70 + "\n# Standard LoRA (no N-branch, no subtraction)\n" + "#" * 70)
    return _normalize_metrics("Standard LoRA", train_single_lora_baseline(cfg))


def run_jtt(base_cfg: TrainConfig) -> Dict:
    cfg = deepcopy(base_cfg); cfg.experiment_name = "jtt"
    print("\n" + "#" * 70 + "\n# JTT (Just Train Twice, upweight=4)\n" + "#" * 70)
    return _normalize_metrics("JTT", train_jtt_baseline(cfg))


def run_poe(base_cfg: TrainConfig) -> Dict:
    cfg = deepcopy(base_cfg); cfg.experiment_name = "poe"
    print("\n" + "#" * 70 + "\n# PoE (Product-of-Experts, hypothesis-only bias)\n" + "#" * 70)
    return _normalize_metrics("PoE (bias-model)", train_poe_baseline(cfg))


def run_zfilter(base_cfg: TrainConfig) -> Dict:
    cfg = deepcopy(base_cfg); cfg.experiment_name = "zfilter"
    print("\n" + "#" * 70 + "\n# z-filtering (data-centric debiasing)\n" + "#" * 70)
    return _normalize_metrics("z-filtering", train_zfilter_baseline(cfg))


def run_negmerge(base_cfg: TrainConfig) -> Dict:
    cfg = deepcopy(base_cfg); cfg.experiment_name = "negmerge"
    print("\n" + "#" * 70 + "\n# NegMerge (global sign-consensus merge)\n" + "#" * 70)
    return _normalize_metrics("NegMerge", train_negmerge_baseline(cfg))


def run_naive_subtract(base_cfg: TrainConfig) -> Dict:
    """TRUE naive task-vector subtraction: global αΔP − βΔN, no trim/sign masks,
    no Phase-3 fine-tuning. Directly addresses reviewer item #7's
    "naive task-vector subtraction without TIES masks"."""
    cfg = deepcopy(base_cfg)
    cfg.experiment_name = "naive_subtract"
    cfg.merge_mode = "naive"
    cfg.enable_layerwise_analysis = False  # global (every layer)
    cfg.no_ties_ablation = False
    cfg.phase3_epochs = 0                  # pure task arithmetic, no debias FT
    print("\n" + "#" * 70 + "\n# Naive task-vector subtraction (no masks, global)\n" + "#" * 70)
    return _normalize_metrics("Naive Subtraction (no TIES)", train_ties_unlearn(cfg))


def run_ties_full(base_cfg: TrainConfig) -> Dict:
    cfg = deepcopy(base_cfg)
    cfg.experiment_name = "ties_full"
    cfg.merge_mode = "full"
    cfg.enable_layerwise_analysis = True
    cfg.no_ties_ablation = False
    print("\n" + "#" * 70 + "\n# TIES-Unlearning Diff-LoRA (full pipeline)\n" + "#" * 70)
    return _normalize_metrics("TIES-Unlearning Diff-LoRA", train_ties_unlearn(cfg))


RUNNERS = {
    "standard_lora": run_standard_lora,
    "jtt": run_jtt,
    "poe": run_poe,
    "zfilter": run_zfilter,
    "negmerge": run_negmerge,
    "naive_subtract": run_naive_subtract,
    "ties_full": run_ties_full,
}


def _fmt_pct(x: float) -> str:
    return "  n/a  " if x != x else f"{x * 100:>6.2f}%"  # x != x ⇒ NaN


def _print_table(rows: List[Dict]):
    header = (f"{'Method':<30s} {'MNLI':>8s} {'ESNLI':>8s} {'ANLI':>8s} {'SNLI-h':>8s} "
              f"{'WANLI':>8s} {'HANS':>8s} {'H-ent':>8s} {'H-nent':>8s}")
    width = 110
    print("\n" + "=" * width)
    print("FINAL COMPARISON")
    print("=" * width)
    print(header)
    print("-" * width)
    for r in rows:
        print(f"{r['method']:<30s} "
              f"{_fmt_pct(r['mnli_accuracy'])} "
              f"{_fmt_pct(r['esnli_accuracy'])} "
              f"{_fmt_pct(r.get('anli_accuracy', float('nan')))} "
              f"{_fmt_pct(r.get('snli_hard_accuracy', float('nan')))} "
              f"{_fmt_pct(r.get('wanli_accuracy', float('nan')))} "
              f"{_fmt_pct(r['hans_overall'])} "
              f"{_fmt_pct(r['hans_entailment'])} "
              f"{_fmt_pct(r['hans_non_entailment'])}")
    print("=" * width)


def _write_markdown(rows: List[Dict], path: str):
    lines = [
        "# Baseline comparison",
        "",
        "| Method | MNLI | e-SNLI | ANLI | SNLI-hard | WANLI | HANS overall | HANS entailment | HANS non-entailment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['mnli_accuracy']*100:.2f}% | {r['esnli_accuracy']*100:.2f}% "
            f"| {r.get('anli_accuracy', float('nan'))*100:.2f}% "
            f"| {r.get('snli_hard_accuracy', float('nan'))*100:.2f}% "
            f"| {r.get('wanli_accuracy', float('nan'))*100:.2f}% "
            f"| {r['hans_overall']*100:.2f}% | {r['hans_entailment']*100:.2f}% "
            f"| {r['hans_non_entailment']*100:.2f}% |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./baseline_results")
    parser.add_argument("--small", action="store_true",
                        help="Use a reduced training budget (Colab-friendly).")
    parser.add_argument("--only", nargs="+", choices=METHOD_TAGS, default=None,
                        help="Run only the specified methods.")
    parser.add_argument("--skip", nargs="+", choices=METHOD_TAGS, default=[],
                        help="Skip the specified methods.")
    parser.add_argument("--leaky-hans", action="store_true",
                        help="Reproduce the original leaky behaviour: train the negative "
                             "branch and run layer localization on the HANS *evaluation* "
                             "set (default trains/localizes on the disjoint HANS train set).")
    args = parser.parse_args()

    methods_to_run = args.only if args.only else METHOD_TAGS
    methods_to_run = [m for m in methods_to_run if m not in args.skip]

    base_cfg = _make_base_cfg(small=args.small, output_dir=args.output_dir,
                              hans_clean_split=not args.leaky_hans)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[run_baselines] output_dir = {args.output_dir}")
    print(f"[run_baselines] small mode  = {args.small}")
    print(f"[run_baselines] HANS split  = {'EVAL (leaky)' if args.leaky_hans else 'TRAIN/EVAL disjoint (clean)'}")
    print(f"[run_baselines] methods     = {methods_to_run}")

    rows: List[Dict] = []
    for tag in methods_to_run:
        try:
            rows.append(RUNNERS[tag](base_cfg))
        except Exception as e:
            print(f"\n[run_baselines] ERROR in {tag}: {e}")
            rows.append({
                "method": tag, "mnli_accuracy": float("nan"), "esnli_accuracy": float("nan"),
                "anli_accuracy": float("nan"), "snli_hard_accuracy": float("nan"),
                "hans_overall": float("nan"), "hans_entailment": float("nan"),
                "hans_non_entailment": float("nan"), "error": str(e),
            })
        # Save incrementally so a crash mid-suite doesn't lose finished runs.
        with open(os.path.join(args.output_dir, "comparison.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    _print_table(rows)
    _write_markdown(rows, os.path.join(args.output_dir, "comparison.md"))
    print(f"\n[run_baselines] Wrote {args.output_dir}/comparison.json and comparison.md")


if __name__ == "__main__":
    main()
