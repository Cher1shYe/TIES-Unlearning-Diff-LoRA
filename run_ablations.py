"""Component ablation driver for TIES-Unlearning Diff-LoRA.

Addresses reviewer item #8 ("isolate the contribution of each method component").
Every variant runs the full Phase-1/2 pipeline and differs only in the Phase-3
merge / layer-selection configuration, so each row isolates one design choice.

Two ablation families:

  Merge masks (layer localization kept ON, Phase-3 FT kept ON):
    * full        : sign-consensus + magnitude trimming   (the proposed method)
    * naive_mask  : no masks            (αΔP − βΔN)
    * sign_only   : sign-consensus mask only
    * trim_only   : magnitude-trimming mask only
    * no_phase3   : full masks but NO Phase-3 debiasing fine-tuning

  Layer localization (merge mask kept FULL, Phase-3 FT kept ON):
    * global      : subtract on every layer
    * random      : subtract on `layer_selection_topk` random layers
    * kl_only     : layer selection by KL divergence only      (knn_mode="off")
    * knn_only    : layer selection by kNN signals only         (kl_weight=0)
    * full        : KL + kNN hybrid (same as the proposed method, shared row)

  Lower bound:
    * no_subtraction : P-only everywhere (no_ties_ablation=True)

Usage:
    python run_ablations.py
    python run_ablations.py --small
    python run_ablations.py --only full sign_only knn_only
"""
import argparse
import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import TrainConfig
from training.trainer import train_ties_unlearn


# tag -> dict of TrainConfig overrides applied on top of the base config.
ABLATIONS: Dict[str, Dict[str, Any]] = {
    # --- merge-mask family (localization on, phase-3 on) ---
    # The proposed full method = sign+trim merge + layer localization + Phase-3 debias FT
    # WITH N-reweighting (Fix A, now the default). Explicit here for documentation.
    "full":            {"merge_mode": "full", "enable_layerwise_analysis": True,
                        "phase3_debias_reweight": True, "phase3_reweight_gamma": 2.0},
    "naive_mask":      {"merge_mode": "naive", "enable_layerwise_analysis": True},
    "sign_only":       {"merge_mode": "sign_only", "enable_layerwise_analysis": True},
    "trim_only":       {"merge_mode": "trim_only", "enable_layerwise_analysis": True},
    "no_phase3":       {"merge_mode": "full", "enable_layerwise_analysis": True, "phase3_epochs": 0},
    # Ablate Fix (A): full method but WITHOUT N-reweighting in Phase-3 (the old behaviour,
    # where debias FT re-injects the shortcut). This is the key row showing A's contribution.
    "no_reweight":     {"merge_mode": "full", "enable_layerwise_analysis": True,
                        "phase3_debias_reweight": False},
    # Alternative fix (B), kept for comparison against (A): freeze the subtracted layers' P
    # in Phase-3 instead of reweighting. Reweight explicitly OFF to isolate the freeze effect.
    "full_lockP":      {"merge_mode": "full", "enable_layerwise_analysis": True,
                        "phase3_freeze_subtracted_p": True, "phase3_debias_reweight": False},
    # --- layer-localization family (full mask, phase-3 on) ---
    "global":          {"merge_mode": "full", "enable_layerwise_analysis": False},
    "random":          {"merge_mode": "full", "random_layer_selection": True},
    "kl_only":         {"merge_mode": "full", "enable_layerwise_analysis": True, "knn_mode": "off"},
    "knn_only":        {"merge_mode": "full", "enable_layerwise_analysis": True,
                        "knn_mode": "pd_and_early_wrong", "kl_weight": 0.0,
                        "kl_topk_candidates": 999},
    # --- lower bound ---
    "no_subtraction":  {"no_ties_ablation": True},
}


def _apply_small_overrides(cfg: TrainConfig) -> TrainConfig:
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


def _normalize(tag: str, raw: Dict) -> Dict:
    p3 = raw.get("phase3", {})
    mnli, hans, esnli = p3.get("mnli", {}), p3.get("hans", {}), p3.get("esnli", {})
    anli, snli_hard = p3.get("anli", {}), p3.get("snli_hard", {})
    wanli = p3.get("wanli", {})
    selected = raw.get("phase2_5", {}).get("shortcut_layers", None)
    return {
        "ablation": tag,
        "mnli_accuracy": float(mnli.get("mnli_accuracy", float("nan"))),
        "esnli_accuracy": float(esnli.get("esnli_accuracy", float("nan"))),
        "anli_accuracy": float(anli.get("anli_accuracy", float("nan"))),
        "snli_hard_accuracy": float(snli_hard.get("snli_hard_accuracy", float("nan"))),
        "wanli_accuracy": float(wanli.get("wanli_accuracy", float("nan"))),
        "hans_overall": float(hans.get("hans_overall", float("nan"))),
        "hans_entailment": float(hans.get("hans_entailment", float("nan"))),
        "hans_non_entailment": float(hans.get("hans_non_entailment", float("nan"))),
        "selected_layers": selected,
    }


def _run_one(base_cfg: TrainConfig, tag: str, overrides: Dict[str, Any]) -> Dict:
    cfg = deepcopy(base_cfg)
    for key, val in overrides.items():
        setattr(cfg, key, val)
    # knn_only needs kl_topk_candidates >= layer_selection_topk (post_init invariant).
    if cfg.kl_topk_candidates < cfg.layer_selection_topk:
        cfg.kl_topk_candidates = cfg.layer_selection_topk
    cfg.experiment_name = f"ablation_{tag}"
    print("\n" + "#" * 72 + f"\n# Ablation: {tag}  ({overrides})\n" + "#" * 72)
    return _normalize(tag, train_ties_unlearn(cfg))


def _fmt_pct(x: float) -> str:
    return "  n/a  " if x != x else f"{x * 100:>6.2f}%"


def _print_table(rows: List[Dict]):
    print("\n" + "=" * 106)
    print("ABLATION COMPARISON")
    print("=" * 106)
    print(f"{'Ablation':<18s} {'MNLI':>8s} {'ESNLI':>8s} {'ANLI':>8s} {'SNLI-h':>8s} "
          f"{'WANLI':>8s} {'HANS':>8s} {'H-ent':>8s} {'H-nent':>8s}")
    print("-" * 106)
    for r in rows:
        print(f"{r['ablation']:<18s} "
              f"{_fmt_pct(r['mnli_accuracy'])} {_fmt_pct(r['esnli_accuracy'])} "
              f"{_fmt_pct(r['anli_accuracy'])} {_fmt_pct(r['snli_hard_accuracy'])} "
              f"{_fmt_pct(r.get('wanli_accuracy', float('nan')))} "
              f"{_fmt_pct(r['hans_overall'])} {_fmt_pct(r['hans_entailment'])} "
              f"{_fmt_pct(r['hans_non_entailment'])}")
    print("=" * 106)


def _write_markdown(rows: List[Dict], path: str):
    lines = [
        "# Component ablation",
        "",
        "| Ablation | MNLI | e-SNLI | ANLI | SNLI-hard | WANLI | HANS overall | HANS entailment | HANS non-entailment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['ablation']} | {r['mnli_accuracy']*100:.2f}% | {r['esnli_accuracy']*100:.2f}% "
            f"| {r['anli_accuracy']*100:.2f}% | {r['snli_hard_accuracy']*100:.2f}% "
            f"| {r.get('wanli_accuracy', float('nan'))*100:.2f}% "
            f"| {r['hans_overall']*100:.2f}% | {r['hans_entailment']*100:.2f}% "
            f"| {r['hans_non_entailment']*100:.2f}% |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./ablation_results")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--only", nargs="+", choices=list(ABLATIONS.keys()), default=None)
    parser.add_argument("--skip", nargs="+", choices=list(ABLATIONS.keys()), default=[])
    args = parser.parse_args()

    tags = args.only if args.only else list(ABLATIONS.keys())
    tags = [t for t in tags if t not in args.skip]

    base_cfg = _make_base_cfg(small=args.small, output_dir=args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n[run_ablations] output_dir = {args.output_dir}")
    print(f"[run_ablations] small mode  = {args.small}")
    print(f"[run_ablations] ablations   = {tags}")

    rows: List[Dict] = []
    for tag in tags:
        try:
            rows.append(_run_one(base_cfg, tag, ABLATIONS[tag]))
        except Exception as e:
            print(f"\n[run_ablations] ERROR in {tag}: {e}")
            rows.append({
                "ablation": tag, "mnli_accuracy": float("nan"), "esnli_accuracy": float("nan"),
                "anli_accuracy": float("nan"), "snli_hard_accuracy": float("nan"),
                "wanli_accuracy": float("nan"),
                "hans_overall": float("nan"), "hans_entailment": float("nan"),
                "hans_non_entailment": float("nan"), "selected_layers": None, "error": str(e),
            })
        with open(os.path.join(args.output_dir, "ablation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    _print_table(rows)
    _write_markdown(rows, os.path.join(args.output_dir, "ablation_summary.md"))
    print(f"\n[run_ablations] Wrote {args.output_dir}/ablation_summary.json and .md")


if __name__ == "__main__":
    main()
