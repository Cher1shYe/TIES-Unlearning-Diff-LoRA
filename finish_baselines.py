"""Rebuild run_baselines.py's comparison.{json,md} from per-method metrics.json.

Use this when comparison.json/.md is incomplete -- e.g. the baselines were run in
separate sessions and each invocation overwrote comparison.json with only the
methods it ran, so the final file holds just the last method. It does NOT touch
run_baselines.py and never retrains anything: each method's
<output_dir>/<dir>/metrics.json (written by the trainer) is the source of truth.

For every method tag in run_baselines.METHOD_TAGS it loads that run's metrics.json,
normalizes it with run_baselines._normalize_metrics (so ANLI / SNLI-hard and the
HANS heuristic breakdown are all carried), and rewrites comparison.json + comparison.md
in the canonical method order with the canonical display names.

Usage:
    python finish_baselines.py --output-dir ./ties_results/baseline_results
    python finish_baselines.py --output-dir ./baseline_results --only ties_full negmerge
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_baselines import (
    METHOD_TAGS, _normalize_metrics, _write_markdown, _print_table,
)

# Canonical display name per tag -- must match the strings run_baselines.py passes
# to _normalize_metrics so the rebuilt table matches a fresh full run.
TAG_TO_NAME: Dict[str, str] = {
    "standard_lora":  "Standard LoRA",
    "jtt":            "JTT",
    "poe":            "PoE (bias-model)",
    "zfilter":        "z-filtering",
    "negmerge":       "NegMerge",
    "naive_subtract": "Naive Subtraction (no TIES)",
    "ties_full":      "TIES-Unlearning Diff-LoRA",
}

# Candidate output sub-directory names per tag. The underlying train_* functions
# sometimes set their own experiment_name (e.g. "baseline_single_lora") that
# differs from the tag run_baselines.py uses, so we try a few known aliases.
TAG_TO_DIRS: Dict[str, List[str]] = {
    "standard_lora":  ["standard_lora", "baseline_single_lora"],
    "jtt":            ["jtt", "jtt_baseline"],
    "poe":            ["poe", "poe_baseline"],
    "zfilter":        ["zfilter", "zfilter_baseline"],
    "negmerge":       ["negmerge", "negmerge_baseline"],
    "naive_subtract": ["naive_subtract"],
    "ties_full":      ["ties_full"],
}


def _find_metrics(output_dir: str, tag: str) -> Optional[str]:
    for d in TAG_TO_DIRS.get(tag, [tag]):
        path = os.path.join(output_dir, d, "metrics.json")
        if os.path.exists(path):
            return path
    return None


def _load(output_dir: str, tag: str) -> Optional[Dict]:
    path = _find_metrics(output_dir, tag)
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[finish] {tag}: metrics.json unreadable ({e}); skipping.")
        return None
    rec = _normalize_metrics(TAG_TO_NAME.get(tag, tag), raw)
    rec["_source"] = os.path.relpath(path, output_dir)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="./baseline_results")
    ap.add_argument("--only", nargs="+", choices=METHOD_TAGS, default=None,
                    help="Only include these methods.")
    ap.add_argument("--skip", nargs="+", choices=METHOD_TAGS, default=[],
                    help="Exclude these methods.")
    args = ap.parse_args()

    tags = args.only if args.only else METHOD_TAGS
    tags = [t for t in tags if t not in args.skip]

    print(f"[finish] output_dir = {args.output_dir}")
    print(f"[finish] methods    = {tags}")

    rows: List[Dict] = []
    found, missing = [], []
    for tag in tags:
        rec = _load(args.output_dir, tag)
        if rec is None:
            missing.append(tag)
            continue
        found.append(tag)
        rows.append(rec)

    if not rows:
        print("[finish] No metrics.json found under "
              f"{args.output_dir}. Nothing to write.")
        return

    json_path = os.path.join(args.output_dir, "comparison.json")
    md_path = os.path.join(args.output_dir, "comparison.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    _write_markdown(rows, md_path)
    _print_table(rows)

    print("\n" + "=" * 60)
    print(f"[finish] rebuilt from {len(found)} methods: {found}")
    if missing:
        print(f"[finish] MISSING (no metrics.json): {missing}")
    print(f"[finish] wrote {json_path}")
    print(f"[finish] wrote {md_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
