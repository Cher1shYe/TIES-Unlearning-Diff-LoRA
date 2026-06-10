"""Finish an interrupted run_sensitivity.py sweep and (re)build its summary JSON.

Use this when a sweep died partway (e.g. a Colab disconnect). It does NOT touch
run_sensitivity.py. For every (param, value) the sweep should contain, it:

  * loads <output_dir>/<tag>/metrics.json if that run already finished, or
  * trains only the runs whose metrics.json is missing,

then writes a complete <output_dir>/sensitivity_summary.json (with the anchor
expanded into each parameter group) that plot_sensitivity.py can consume.

The per-run metrics.json files written by the trainer are the source of truth, so
already-finished runs are never retrained -- zero wasted compute.

Usage:
    # original sweep was full scale:
    python finish_sensitivity.py --output-dir ./sensitivity_results
    # original sweep used --small (you MUST match it so the new runs are comparable):
    python finish_sensitivity.py --output-dir ./sensitivity_results --small
    # only rebuild the JSON from what's already on disk, train nothing:
    python finish_sensitivity.py --output-dir ./sensitivity_results --assemble-only
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_sensitivity import (
    PARAM_GRID, RunSpec, _build_run_list, _extract_metrics, _make_base_cfg,
    _record_for_run, _run_one, _expand_anchor_for_plotting,
    _write_rank_control_table,
)


def _load_cached(output_dir: str, run: RunSpec, base_cfg) -> Optional[Dict]:
    """Return a summary record from an existing <tag>/metrics.json, or None if it
    is missing / unreadable (then the run still needs training)."""
    path = os.path.join(output_dir, run.tag, "metrics.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"[finish] {run.tag}: metrics.json unreadable ({e}); will re-train.")
        return None
    record = _record_for_run(run, base_cfg)
    record["status"] = "ok"
    record.update(_extract_metrics(metrics))
    return record


def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="./sensitivity_results")
    ap.add_argument("--small", action="store_true",
                    help="Match a --small sweep when training the missing runs.")
    ap.add_argument("--assemble-only", action="store_true",
                    help="Rebuild the summary JSON from existing metrics.json; train nothing.")
    ap.add_argument("--only", nargs="+", default=[],
                    choices=list(PARAM_GRID.keys()) + ["rank_controls"],
                    help="Restrict finish/resume to these parameters/control families.")
    ap.add_argument("--skip-rank-controls", action="store_true",
                    help="Rebuild/finish only the original OAT grid.")
    return ap.parse_args(argv)


def main():
    args = _parse_args()

    base_cfg = _make_base_cfg(small=args.small, output_dir=args.output_dir)
    runs, _ = _build_run_list(
        base_cfg,
        PARAM_GRID,
        only=args.only,
        skip_rank_controls=args.skip_rank_controls,
    )
    print(f"[finish] output_dir = {args.output_dir}")
    print(f"[finish] small      = {args.small}  (must match the original sweep)")
    print(f"[finish] expecting {len(runs)} runs total")

    records: List[Dict] = []
    cached: List[str] = []
    trained: List[str] = []
    missing: List[str] = []

    for run in runs:
        rec = _load_cached(args.output_dir, run, base_cfg)
        if rec is not None:
            cached.append(rec["tag"])
            records.append(rec)
        elif args.assemble_only:
            missing.append(run.tag)
        else:
            print(f"\n[finish] MISSING -> training {run.tag}")
            rec = _run_one(base_cfg, run, args.output_dir)
            trained.append(rec["tag"])
            records.append(rec)

    expanded = _expand_anchor_for_plotting(
        records,
        PARAM_GRID,
        base_cfg,
        only=args.only,
        skip_rank_controls=args.skip_rank_controls,
    )
    out_path = os.path.join(args.output_dir, "sensitivity_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(expanded, f, indent=2, ensure_ascii=False)
    _write_rank_control_table(expanded, args.output_dir)

    print("\n" + "=" * 60)
    print(f"[finish] cached  : {len(cached)} runs (loaded from disk, not retrained)")
    print(f"[finish] trained : {len(trained)} runs {trained}")
    if missing:
        print(f"[finish] MISSING : {len(missing)} runs (assemble-only, not in JSON) {missing}")
    print(f"[finish] wrote {out_path}: {len(expanded)} records")
    if args.only == ["rank_controls"]:
        print(f"[finish] next: python plot_mr4_rank_controls.py --results-dir {args.output_dir}")
    else:
        print(f"[finish] next: python plot_sensitivity.py --output-dir {args.output_dir}")
        if (not args.skip_rank_controls) and (not args.only or "rank_controls" in args.only):
            print(f"[finish] next MR.4 figure: python plot_mr4_rank_controls.py --results-dir {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
