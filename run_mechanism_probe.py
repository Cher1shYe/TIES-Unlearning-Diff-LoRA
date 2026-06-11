"""Mechanism probe: does the masked N-subtraction move HANS *at all*, and at
which (merge_mode, layer set, beta) does it start to?

Motivation (2026-06-11): on A100 with paired seeds, full vs no_subtraction is
identical on every metric, pre- and post-FT (new_multiseed seeds 42/123,
delta <= 0.65pt H-non). The earlier +10.9pt delta came from the control having
been run on an L4. Before spending any more multiseed compute, this script
answers the prior question: is there ANY merge configuration whose pre-FT
effect is distinguishable from the p_only control?

Because TIESUnlearnLoRALinear computes the merge in forward() (never baked
into weights), one phase-1+2(+2.5) training run supports an arbitrary sweep of
post-hoc merge settings at pure eval cost:

  1. Train phases 1-2.5 once via train_ties_unlearn(phase3_epochs=0); this
     also writes the usual metrics.json (shortcut_layers, post_merge_pre_ft).
  2. Reload the phase-2 checkpoint into a fresh model and, for each condition
     (merge_mode x {selected, global layers} x beta), evaluate MNLI + HANS.

All conditions share the exact same trained weights, so differences between
rows are the mechanism and nothing else (no seed, no GPU, no FT confounds).

Usage:
    python run_mechanism_probe.py --seed 42                  # full grid
    python run_mechanism_probe.py --small --betas 0.5 1.0 2.0
    python run_mechanism_probe.py --seed 42 --modes full naive sign_only trim_only
"""
import argparse
import json
import os
import sys
from copy import deepcopy
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.config import TrainConfig, LoRAConfig
from data.dataloader import set_seed, make_mnli_loaders, make_hans_loader
from models.surgery import (
    inject_ties_unlearn_lora, get_ties_modules_by_layer, configure_ties_layers,
)
from models.ties_lora import TIESUnlearnLoRALinear, set_model_forward_mode
from training.evaluate import eval_mnli, eval_hans
from training.trainer import train_ties_unlearn
from run_baselines import _make_base_cfg, _apply_small_overrides  # noqa: F401 (small applied via _make_base_cfg)


def _probe_cfg(args) -> TrainConfig:
    cfg = _make_base_cfg(small=args.small, output_dir=args.output_dir)
    cfg.seed = args.seed
    cfg.experiment_name = f"probe_seed{args.seed}"
    cfg.merge_mode = "full"
    cfg.enable_layerwise_analysis = True
    cfg.phase3_epochs = 0                 # no FT: we only need phases 1-2.5
    cfg.save_checkpoints = True
    cfg.save_checkpoints_per_phase = True
    cfg.checkpoint_dir = None             # checkpoints under <run_dir>/checkpoints/
    return cfg


def _run_dir(cfg: TrainConfig) -> str:
    return os.path.join(cfg.output_dir, cfg.experiment_name)


def _phase2_ckpt_path(cfg: TrainConfig) -> str:
    return os.path.join(_run_dir(cfg), "checkpoints",
                        f"phase2_checkpoint_epoch{cfg.phase2_epochs}.pt")


def _ensure_trained(cfg: TrainConfig) -> Dict:
    """Run phases 1-2.5 if needed; return the run's metrics dict."""
    ckpt = _phase2_ckpt_path(cfg)
    metrics_path = os.path.join(_run_dir(cfg), "metrics.json")
    if os.path.exists(metrics_path) and os.path.exists(ckpt):
        m = json.load(open(metrics_path, encoding="utf-8"))
        if m.get("phase2_5", {}).get("shortcut_layers"):
            print(f"[probe] Reusing existing run: {metrics_path}")
            return m
    if os.path.exists(ckpt):
        # Phase 1-2 done but 2.5 metrics missing: resume to redo analysis only.
        print(f"[probe] Resuming from {ckpt} to regenerate phase-2.5 analysis")
        return train_ties_unlearn(cfg, resume_from_checkpoint_path=ckpt)
    print("[probe] No checkpoint found; training phases 1-2.5 now")
    return train_ties_unlearn(cfg)


def _build_model_from_ckpt(cfg: TrainConfig, device) -> torch.nn.Module:
    """Reconstruct the model exactly as training/trainer.py does, then load the
    phase-2 checkpoint weights (P, N, head all trained)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels,
    )
    lora_cfg = LoRAConfig(
        pos_rank=cfg.pos_rank, neg_rank=cfg.neg_rank,
        alpha=cfg.alpha, beta=cfg.beta,
        lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        trim_ratio=cfg.trim_ratio, merge_mode=cfg.merge_mode,
    )
    model, _ = inject_ties_unlearn_lora(model, list(cfg.target_modules), lora_cfg)
    ckpt = torch.load(_phase2_ckpt_path(cfg), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    set_model_forward_mode(model, "eval")
    return model


def _set_beta(model: torch.nn.Module, beta: float):
    for module in model.modules():
        if isinstance(module, TIESUnlearnLoRALinear):
            module.beta = beta


def _set_merge_mode(model: torch.nn.Module, mode: str):
    for module in model.modules():
        if isinstance(module, TIESUnlearnLoRALinear):
            module.merge_mode = mode


@torch.no_grad()
def _eval_condition(model, val_loader, hans_loader, device) -> Dict:
    mnli = eval_mnli(model, val_loader, device)
    hans = eval_hans(model, hans_loader, device)
    return {
        "mnli_accuracy": mnli["mnli_accuracy"],
        "hans_overall": hans["hans_overall"],
        "hans_entailment": hans["hans_entailment"],
        "hans_non_entailment": hans["hans_non_entailment"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./mechanism_probe_results")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--modes", nargs="+", default=["full", "naive"],
                        choices=["full", "naive", "sign_only", "trim_only"])
    args = parser.parse_args()

    cfg = _probe_cfg(args)
    os.makedirs(args.output_dir, exist_ok=True)

    run_metrics = _ensure_trained(cfg)
    selected = run_metrics["phase2_5"]["shortcut_layers"]
    print(f"[probe] phase-2.5 selected layers: {selected}")

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    _, val_loader = make_mnli_loaders(cfg, tok)
    hans_loader = make_hans_loader(cfg, tok)

    model = _build_model_from_ckpt(cfg, device)
    all_layers = list(get_ties_modules_by_layer(model).keys())
    layer_sets = {"selected": selected, "global": all_layers}

    rows: List[Dict] = []

    def record(name: str, res: Dict):
        rows.append({"condition": name, **res})
        print(f"[probe] {name:<34s} MNLI={res['mnli_accuracy']*100:6.2f}  "
              f"HANS={res['hans_overall']*100:6.2f}  "
              f"H-ent={res['hans_entailment']*100:6.2f}  "
              f"H-non={res['hans_non_entailment']*100:6.2f}")

    # Control: alpha*dP only (== no_subtraction's merged model).
    configure_ties_layers(model, [])
    record("p_only(control)", _eval_condition(model, val_loader, hans_loader, device))
    control = rows[0]

    for mode in args.modes:
        _set_merge_mode(model, mode)
        for layers_name, layers in layer_sets.items():
            configure_ties_layers(model, layers)
            for beta in args.betas:
                _set_beta(model, beta)
                record(f"{mode}/{layers_name}/b{beta:g}",
                       _eval_condition(model, val_loader, hans_loader, device))

    for r in rows:
        r["d_hnon_vs_control"] = r["hans_non_entailment"] - control["hans_non_entailment"]
        r["d_mnli_vs_control"] = r["mnli_accuracy"] - control["mnli_accuracy"]

    out_json = os.path.join(args.output_dir, f"probe_seed{args.seed}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "selected_layers": selected, "rows": rows},
                  f, indent=2, ensure_ascii=False)

    lines = [
        f"# Mechanism probe (seed {args.seed}, pre-FT, shared phase-2 weights)",
        "",
        f"Selected layers: {', '.join(selected)}",
        "",
        "| Condition | MNLI | HANS | H-ent | H-non | ΔH-non | ΔMNLI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['condition']} | {r['mnli_accuracy']*100:.2f} | {r['hans_overall']*100:.2f} "
            f"| {r['hans_entailment']*100:.2f} | {r['hans_non_entailment']*100:.2f} "
            f"| {r['d_hnon_vs_control']*100:+.2f} | {r['d_mnli_vs_control']*100:+.2f} |"
        )
    out_md = os.path.join(args.output_dir, f"probe_seed{args.seed}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[probe] Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
