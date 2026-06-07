"""Multi-seed robustness driver for TIES-Unlearning Diff-LoRA.

Directly responds to reviewer item #9:
    "Robustness: Report multi-seed results and statistical variation.
     Suggest: report results over multiple random seeds with standard deviations
     or confidence intervals. The method contains several sensitive components,
     including rank choices, shortcut-biased training, layer selection, and the
     final debiasing stage."

For each (method, seed) it runs the corresponding baseline/method from
run_baselines.py with that seed, then aggregates per method into mean ± std
across seeds. Per-seed outputs are isolated under <output_dir>/seed_<s>/.

Usage:
    python run_multiseed.py                                  # default methods, 3 seeds
    python run_multiseed.py --small --seeds 42 123 2024
    python run_multiseed.py --methods ties_full standard_lora negmerge poe
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from copy import deepcopy

from run_baselines import RUNNERS, METHOD_TAGS, _make_base_cfg, _fmt_pct, _normalize_metrics
from run_ablations import ABLATIONS
from training.trainer import train_ties_unlearn

METRIC_KEYS = [
    "mnli_accuracy", "esnli_accuracy", "anli_accuracy", "snli_hard_accuracy",
    "wanli_accuracy",
    "hans_overall", "hans_entailment", "hans_non_entailment",
]

# Methods you can run across seeds: every baseline tag PLUS every ablation tag
# (so we can also get mean±std for no_reweight / no_subtraction / full_lockP, etc.).
ALL_TAGS = METHOD_TAGS + [t for t in ABLATIONS if t not in METHOD_TAGS]


def _run_tagged(tag: str, base_cfg):
    """Dispatch a tag to either a run_baselines runner or a run_ablations config,
    returning a normalized metrics dict (method + metric keys)."""
    if tag in RUNNERS:
        return RUNNERS[tag](base_cfg)
    if tag in ABLATIONS:
        cfg = deepcopy(base_cfg)
        for k, v in ABLATIONS[tag].items():
            setattr(cfg, k, v)
        # knn_only etc. need kl_topk_candidates >= layer_selection_topk (post_init invariant).
        if cfg.kl_topk_candidates < cfg.layer_selection_topk:
            cfg.kl_topk_candidates = cfg.layer_selection_topk
        cfg.experiment_name = f"ablation_{tag}"
        return _normalize_metrics(tag, train_ties_unlearn(cfg))
    raise ValueError(f"unknown method/ablation tag: {tag!r}")


def _aggregate(records: List[Dict]) -> List[Dict]:
    """Group per-(method,seed) records by method and compute mean / std per metric."""
    per_tag: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        per_tag[rec["tag"]].append(rec)

    agg: List[Dict] = []
    for tag, recs in per_tag.items():
        row = {
            "tag": tag,
            "method": recs[0].get("method", tag),
            "n_seeds": len(recs),
            "seeds": [r["seed"] for r in recs],
        }
        for k in METRIC_KEYS:
            vals = np.array([r.get(k, float("nan")) for r in recs], dtype=float)
            vals = vals[~np.isnan(vals)]
            row[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
            # Sample std (ddof=1); 0.0 when only one successful seed.
            row[f"{k}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        agg.append(row)
    return agg


def _fmt_mean_std(mean: float, std: float) -> str:
    if mean != mean:  # NaN
        return "     n/a     "
    return f"{mean*100:5.2f} ± {std*100:4.2f}"


def _print_table(agg: List[Dict]):
    print("\n" + "=" * 144)
    print("MULTI-SEED COMPARISON  (mean ± std over seeds, %)")
    print("=" * 144)
    print(f"{'Method':<26s} {'n':>2s}  {'MNLI':>13s}  {'e-SNLI':>13s}  {'ANLI':>13s}  "
          f"{'SNLI-h':>13s}  {'WANLI':>13s}  {'HANS':>13s}  {'H-ent':>13s}  {'H-nent':>13s}")
    print("-" * 144)
    for r in agg:
        print(f"{r['method']:<26s} {r['n_seeds']:>2d}  "
              f"{_fmt_mean_std(r['mnli_accuracy_mean'], r['mnli_accuracy_std'])}  "
              f"{_fmt_mean_std(r['esnli_accuracy_mean'], r['esnli_accuracy_std'])}  "
              f"{_fmt_mean_std(r['anli_accuracy_mean'], r['anli_accuracy_std'])}  "
              f"{_fmt_mean_std(r['snli_hard_accuracy_mean'], r['snli_hard_accuracy_std'])}  "
              f"{_fmt_mean_std(r['wanli_accuracy_mean'], r['wanli_accuracy_std'])}  "
              f"{_fmt_mean_std(r['hans_overall_mean'], r['hans_overall_std'])}  "
              f"{_fmt_mean_std(r['hans_entailment_mean'], r['hans_entailment_std'])}  "
              f"{_fmt_mean_std(r['hans_non_entailment_mean'], r['hans_non_entailment_std'])}")
    print("=" * 144)


def _write_markdown(agg: List[Dict], path: str):
    lines = [
        "# Multi-seed results (mean ± std over seeds)",
        "",
        "| Method | seeds | MNLI | e-SNLI | ANLI | SNLI-hard | WANLI | HANS overall | HANS entailment | HANS non-entailment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def cell(r, k):
        m, s = r[f"{k}_mean"], r[f"{k}_std"]
        return "n/a" if m != m else f"{m*100:.2f} ± {s*100:.2f}"

    for r in agg:
        lines.append(
            f"| {r['method']} | {r['n_seeds']} "
            f"| {cell(r, 'mnli_accuracy')} | {cell(r, 'esnli_accuracy')} "
            f"| {cell(r, 'anli_accuracy')} | {cell(r, 'snli_hard_accuracy')} "
            f"| {cell(r, 'wanli_accuracy')} "
            f"| {cell(r, 'hans_overall')} | {cell(r, 'hans_entailment')} "
            f"| {cell(r, 'hans_non_entailment')} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./multiseed_results")
    parser.add_argument("--small", action="store_true",
                        help="Use a reduced training budget (Colab-friendly).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024],
                        help="Random seeds to run each method with.")
    parser.add_argument("--methods", nargs="+", choices=ALL_TAGS, metavar="TAG",
                        default=["standard_lora", "jtt", "ties_full", "no_reweight", "no_subtraction"],
                        help="Methods to run across seeds: run_baselines tags AND run_ablations "
                             "tags (e.g. ties_full no_reweight no_subtraction full_lockP).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n[run_multiseed] output_dir = {args.output_dir}")
    print(f"[run_multiseed] small mode  = {args.small}")
    print(f"[run_multiseed] seeds       = {args.seeds}")
    print(f"[run_multiseed] methods     = {args.methods}")

    records: List[Dict] = []
    for seed in args.seeds:
        seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
        for tag in args.methods:
            base_cfg = _make_base_cfg(small=args.small, output_dir=seed_dir)
            base_cfg.seed = seed
            print("\n" + "*" * 72 + f"\n* seed={seed}  method={tag}\n" + "*" * 72)
            try:
                metrics = _run_tagged(tag, base_cfg)   # normalized dict (method + metric keys)
                rec = {"tag": tag, "seed": seed, "method": metrics.get("method", tag)}
                for k in METRIC_KEYS:
                    rec[k] = float(metrics.get(k, float("nan")))
            except Exception as e:
                print(f"[run_multiseed] ERROR seed={seed} method={tag}: {e}")
                rec = {"tag": tag, "seed": seed, "method": tag, "error": str(e)}
                for k in METRIC_KEYS:
                    rec[k] = float("nan")
            records.append(rec)

            # Save incrementally (raw per-run records + current aggregate).
            agg = _aggregate(records)
            with open(os.path.join(args.output_dir, "multiseed_runs.json"), "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            with open(os.path.join(args.output_dir, "multiseed_summary.json"), "w", encoding="utf-8") as f:
                json.dump(agg, f, indent=2, ensure_ascii=False)

    agg = _aggregate(records)
    _print_table(agg)
    _write_markdown(agg, os.path.join(args.output_dir, "multiseed_summary.md"))
    print(f"\n[run_multiseed] Wrote {args.output_dir}/multiseed_summary.{{json,md}} "
          f"and multiseed_runs.json")


if __name__ == "__main__":
    main()
