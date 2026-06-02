"""Plot the OAT sensitivity sweep produced by run_sensitivity.py.

Reads sensitivity_results/sensitivity_summary.json (one record per (param, value))
and writes one PNG per swept parameter showing:
    - MNLI accuracy            (in-distribution utility, left y-axis)
    - e-SNLI accuracy          (extra utility/OOD, left y-axis)
    - HANS non-entailment acc  (robustness, right y-axis -- the metric the paper
                                actually moves; HANS-overall is dominated by the
                                entailment side which most methods ace.)
    - HANS overall             (right y-axis, dashed reference)
A vertical dotted line marks the anchor (default-config) value. Numeric parameters
get a numeric x-axis; the categorical `target_modules` sweep gets string ticks.
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _group_by_param(records: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        if r.get("status", "ok") != "ok":
            continue
        if r["param"] == "anchor":   # the bare anchor is expanded into each param group
            continue
        grouped[r["param"]].append(r)
    for p in grouped:
        # Sort by the string label so both numeric and categorical sweeps order stably.
        grouped[p].sort(key=lambda x: x.get("value_label", str(x["value"])))
    return grouped


def _is_numeric_param(rows: List[Dict]) -> bool:
    return all(isinstance(r["value"], (int, float)) and not isinstance(r["value"], bool)
               for r in rows)


def _plot_one(param: str, rows: List[Dict], out_path: str):
    numeric = _is_numeric_param(rows)
    if numeric:
        rows = sorted(rows, key=lambda r: r["value"])
        xs = [r["value"] for r in rows]
    else:
        xs = list(range(len(rows)))
    labels = [r.get("value_label", str(r["value"])) for r in rows]

    mnli = [r["mnli_accuracy"] * 100 for r in rows]
    esnli = [r["esnli_accuracy"] * 100 for r in rows]
    hans_nent = [r["hans_non_entailment"] * 100 for r in rows]
    hans_overall = [r["hans_overall"] * 100 for r in rows]

    fig, ax1 = plt.subplots(figsize=(6.8, 4.4))
    c_mnli, c_esnli, c_nent, c_all = "#1f77b4", "#2ca02c", "#d62728", "#7f7f7f"

    l1, = ax1.plot(xs, mnli, marker="o", color=c_mnli, label="MNLI acc (utility)")
    l2, = ax1.plot(xs, esnli, marker="D", color=c_esnli, linestyle="--",
                   alpha=0.8, label="e-SNLI acc")
    ax1.set_xlabel(param)
    ax1.set_ylabel("Utility accuracy (%)", color=c_mnli)
    ax1.tick_params(axis="y", labelcolor=c_mnli)
    ax1.grid(True, alpha=0.3)
    if not numeric:
        ax1.set_xticks(xs)
        ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

    ax2 = ax1.twinx()
    l3, = ax2.plot(xs, hans_nent, marker="s", color=c_nent,
                   label="HANS non-ent (robustness)")
    l4, = ax2.plot(xs, hans_overall, marker="^", color=c_all, linestyle="--",
                   alpha=0.7, label="HANS overall")
    ax2.set_ylabel("HANS accuracy (%)", color=c_nent)
    ax2.tick_params(axis="y", labelcolor=c_nent)

    # Mark the anchor (default-value) point if present.
    anchor_rows = [r for r in rows if r.get("from_anchor")]
    if anchor_rows:
        anchor = anchor_rows[0]
        x_anchor = anchor["value"] if numeric else labels.index(anchor.get("value_label", str(anchor["value"])))
        ax1.axvline(x_anchor, color="black", alpha=0.25, linestyle=":", label="default")

    ax1.set_title(f"Sensitivity: {param}")
    fig.legend(handles=[l1, l2, l3, l4], loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_summary_table(grouped: Dict[str, List[Dict]], out_path: str):
    lines = ["# Sensitivity sweep summary",
             "",
             "Each row reports MNLI / e-SNLI / HANS-overall / HANS-non-entailment for one "
             "(parameter, value) configuration. The default-value row in each block "
             "is the shared anchor run.",
             ""]
    for p, rows in grouped.items():
        lines.append(f"## {p}\n")
        lines.append("| value | MNLI | e-SNLI | HANS overall | HANS non-entailment |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in rows:
            mark = "  *(default)*" if r.get("from_anchor") else ""
            lines.append(
                f"| {r.get('value_label', r['value'])}{mark} "
                f"| {r['mnli_accuracy']*100:.2f}% | {r['esnli_accuracy']*100:.2f}% "
                f"| {r['hans_overall']*100:.2f}% | {r['hans_non_entailment']*100:.2f}% |"
            )
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sensitivity_results/sensitivity_summary.json")
    parser.add_argument("--output-dir", default="sensitivity_results")
    args = parser.parse_args()

    records = _load(args.input)
    grouped = _group_by_param(records)
    if not grouped:
        print(f"[plot] No successful runs found in {args.input}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    for p, rows in grouped.items():
        out_path = os.path.join(args.output_dir, f"{p}_sensitivity.png")
        _plot_one(p, rows, out_path)
        print(f"[plot] {p}: {len(rows)} points -> {out_path}")

    table_path = os.path.join(args.output_dir, "sensitivity_table.md")
    _write_summary_table(grouped, table_path)
    print(f"[plot] Wrote summary table: {table_path}")


if __name__ == "__main__":
    main()
