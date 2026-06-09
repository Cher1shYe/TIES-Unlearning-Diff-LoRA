"""Generate the MR.4 publication figure from rank-control results.

The figure answers the reviewer request for equal-rank, reversed-rank, and
branch-only controls. It reads the small-budget rank-control outputs and writes
an editable SVG plus a high-resolution PNG preview.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "ties_results" / "mr4_rank_controls_small"
mpl = None
plt = None
np = None
Rectangle = None

ORDER = [
    "default_differential",
    "equal_rank_low",
    "equal_rank_high",
    "reversed_rank_default",
]

SHORT_LABEL = {
    "default_differential": "Default diff",
    "equal_rank_low": "Equal low",
    "equal_rank_high": "Equal high",
    "reversed_rank_default": "Reversed",
}

RANK_LABEL = {
    "default_differential": "rP16 / rN4",
    "equal_rank_low": "rP4 / rN4",
    "equal_rank_high": "rP16 / rN16",
    "reversed_rank_default": "rP4 / rN16",
}

CONTROL_COLORS = {
    "default_differential": "#0F4D92",
    "equal_rank_low": "#42949E",
    "equal_rank_high": "#9A4D8E",
    "reversed_rank_default": "#B64342",
}

P_COLOR = "#0F4D92"
N_COLOR = "#B64342"
NEUTRAL_DARK = "#272727"
NEUTRAL_MID = "#767676"
NEUTRAL_LIGHT = "#D8D8D8"


def load_plot_dependencies() -> None:
    global mpl, plt, np, Rectangle
    if plt is not None:
        return

    try:
        import matplotlib
        import numpy as numpy_module
        from matplotlib.patches import Rectangle as RectangleClass
    except ModuleNotFoundError as exc:
        missing = exc.name or "plotting dependency"
        raise SystemExit(
            f"Missing plotting dependency: {missing}. "
            "Install project requirements with `pip install -r requirements.txt`."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib as mpl_module
    import matplotlib.pyplot as plt_module

    mpl = mpl_module
    plt = plt_module
    np = numpy_module
    Rectangle = RectangleClass


def percent(value: float) -> float:
    return float(value) * 100.0


def metric(record: Dict, key: str) -> float:
    return percent(record[key])


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metrics_path_for(record: Dict, results_dir: Path) -> Path:
    if record["value"] == "default_differential" or record.get("from_anchor"):
        return results_dir / "anchor_default" / "metrics.json"
    return results_dir / record["tag"] / "metrics.json"


def load_rank_records(results_dir: Path) -> Tuple[List[Dict], Dict[str, Dict]]:
    summary_path = results_dir / "sensitivity_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing sensitivity summary: {summary_path}")

    raw_records = load_json(summary_path)
    by_value = {
        row["value"]: row
        for row in raw_records
        if row.get("group") == "rank_control" and row.get("status", "ok") == "ok"
    }

    missing = [value for value in ORDER if value not in by_value]
    if missing:
        raise ValueError(f"Missing rank-control records: {missing}")

    records = [by_value[value] for value in ORDER]
    metrics_by_value: Dict[str, Dict] = {}
    for record in records:
        path = metrics_path_for(record, results_dir)
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics for {record['value']}: {path}")
        metrics_by_value[record["value"]] = load_json(path)
    return records, metrics_by_value


def layer_number(layer_tag: str) -> int:
    return int(layer_tag.rsplit(".", 1)[-1])


def setup_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=NEUTRAL_DARK,
    )


def write_source_tables(records: List[Dict], metrics_by_value: Dict[str, Dict], out_dir: Path) -> None:
    metrics_path = out_dir / "mr4_rank_control_figure_source_metrics.csv"
    layer_path = out_dir / "mr4_rank_control_figure_source_layers.csv"

    metric_fields = [
        "control",
        "short_label",
        "relation",
        "pos_rank",
        "neg_rank",
        "merged_mnli",
        "merged_esnli",
        "merged_anli",
        "merged_snli_hard",
        "merged_wanli",
        "merged_hans_entailment",
        "merged_hans_non_entailment",
        "p_branch_mnli",
        "p_branch_hans_entailment",
        "p_branch_hans_non_entailment",
        "n_branch_mnli",
        "n_branch_hans_entailment",
        "n_branch_hans_non_entailment",
        "branch_eval_source",
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "control": record["value"],
                    "short_label": SHORT_LABEL[record["value"]],
                    "relation": record["rank_relation"],
                    "pos_rank": record["pos_rank"],
                    "neg_rank": record["neg_rank"],
                    "merged_mnli": metric(record, "mnli_accuracy"),
                    "merged_esnli": metric(record, "esnli_accuracy"),
                    "merged_anli": metric(record, "anli_accuracy"),
                    "merged_snli_hard": metric(record, "snli_hard_accuracy"),
                    "merged_wanli": metric(record, "wanli_accuracy"),
                    "merged_hans_entailment": metric(record, "hans_entailment"),
                    "merged_hans_non_entailment": metric(record, "hans_non_entailment"),
                    "p_branch_mnli": metric(record, "p_branch_mnli_accuracy"),
                    "p_branch_hans_entailment": metric(record, "p_branch_hans_entailment"),
                    "p_branch_hans_non_entailment": metric(record, "p_branch_hans_non_entailment"),
                    "n_branch_mnli": metric(record, "n_branch_mnli_accuracy"),
                    "n_branch_hans_entailment": metric(record, "n_branch_hans_entailment"),
                    "n_branch_hans_non_entailment": metric(record, "n_branch_hans_non_entailment"),
                    "branch_eval_source": record.get("branch_eval_source", ""),
                }
            )

    layer_fields = ["control", "layer", "score", "selected_for_subtraction"]
    with layer_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=layer_fields)
        writer.writeheader()
        for record in records:
            value = record["value"]
            selected = set(metrics_by_value[value]["phase2_5"].get("shortcut_layers", []))
            for row in metrics_by_value[value]["phase2_5"].get("final_layer_scores", []):
                writer.writerow(
                    {
                        "control": value,
                        "layer": row["layer_tag"],
                        "score": row["score"],
                        "selected_for_subtraction": row["layer_tag"] in selected,
                    }
                )


def panel_rank_design(ax, records: List[Dict]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 27.3)
    ax.set_ylim(0.1, len(records) + 0.9)
    ax.set_title("Rank-control design", loc="left", fontsize=8, pad=4)

    ax.text(9.7, len(records) + 0.52, "P branch", color=P_COLOR, fontsize=6.5, ha="left")
    ax.text(16.7, len(records) + 0.52, "N branch", color=N_COLOR, fontsize=6.5, ha="left")
    ax.text(23.0, len(records) + 0.52, "relation", color=NEUTRAL_MID, fontsize=6.5, ha="left")

    for row_idx, record in enumerate(records):
        y = len(records) - row_idx
        value = record["value"]
        ax.text(0.1, y, SHORT_LABEL[value], ha="left", va="center", fontsize=7)
        ax.text(5.85, y, RANK_LABEL[value], ha="left", va="center", fontsize=6.5, color=NEUTRAL_MID)

        p_rank = int(record["pos_rank"])
        n_rank = int(record["neg_rank"])
        ax.add_patch(Rectangle((9.7, y + 0.07), p_rank / 2.3, 0.18, color=P_COLOR, alpha=0.90))
        ax.add_patch(Rectangle((16.7, y - 0.25), n_rank / 2.3, 0.18, color=N_COLOR, alpha=0.88))
        ax.text(9.7 + p_rank / 2.3 + 0.15, y + 0.16, str(p_rank), fontsize=6.2, va="center")
        ax.text(16.7 + n_rank / 2.3 + 0.15, y - 0.16, str(n_rank), fontsize=6.2, va="center")
        relation = record["rank_relation"].replace("_", " ")
        ax.text(23.0, y, relation, fontsize=6.4, color=CONTROL_COLORS[value], va="center")

    ax.plot([9.7, 22.6], [0.55, 0.55], color=NEUTRAL_LIGHT, lw=0.7)
    ax.text(16.15, 0.25, "rank capacity, scaled to max 16", ha="center", fontsize=6.1, color=NEUTRAL_MID)
    add_panel_label(ax, "a")


def panel_tradeoff(ax, records: List[Dict]) -> None:
    ax.set_title("Merged model trade-off", loc="left", fontsize=8, pad=4)
    for record in records:
        value = record["value"]
        x = metric(record, "mnli_accuracy")
        y = metric(record, "hans_non_entailment")
        ax.scatter(
            x,
            y,
            s=58,
            color=CONTROL_COLORS[value],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        offsets = {
            "default_differential": (0.07, 0.8),
            "equal_rank_low": (0.08, 0.6),
            "equal_rank_high": (0.08, 0.45),
            "reversed_rank_default": (0.08, -0.9),
        }
        dx, dy = offsets[value]
        ax.text(x + dx, y + dy, SHORT_LABEL[value], fontsize=6.4, color=CONTROL_COLORS[value])

    ax.grid(True, color="#E6E6E6", lw=0.6)
    ax.set_xlabel("MNLI accuracy (%)")
    ax.set_ylabel("HANS non-ent accuracy (%)")
    ax.set_xlim(79.4, 83.0)
    ax.set_ylim(2.0, 23.0)
    add_panel_label(ax, "b")


def annotate_matrix(ax, matrix: np.ndarray, fmt: str = "{:.1f}") -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            color = "white" if value > 68 else NEUTRAL_DARK
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=5.7, color=color)


def panel_branch_matrix(ax, records: List[Dict]) -> None:
    columns = ["P\nMNLI", "P\nHANS-N", "N\nMNLI", "N\nHANS-E", "N\nHANS-N"]
    matrix = np.array(
        [
            [
                metric(r, "p_branch_mnli_accuracy"),
                metric(r, "p_branch_hans_non_entailment"),
                metric(r, "n_branch_mnli_accuracy"),
                metric(r, "n_branch_hans_entailment"),
                metric(r, "n_branch_hans_non_entailment"),
            ]
            for r in records
        ]
    )

    cmap = mpl.colormaps["Blues"].copy()
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    annotate_matrix(ax, matrix)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, fontsize=6.1)
    ax.set_yticks(range(len(records)))
    ax.set_yticklabels([SHORT_LABEL[r["value"]] for r in records], fontsize=6.3)
    ax.tick_params(length=0)
    ax.set_title("Branch-only evaluations", loc="left", fontsize=8, pad=4)
    ax.axvline(1.5, color="white", lw=1.2)
    ax.text(0.5, -0.92, "P-only", ha="center", fontsize=6.2, color=P_COLOR)
    ax.text(3.0, -0.92, "N-only", ha="center", fontsize=6.2, color=N_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("accuracy (%)", fontsize=6.3)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    add_panel_label(ax, "c")


def panel_delta_heatmap(ax, records: List[Dict]) -> None:
    baseline = records[0]
    rows = records[1:]
    columns = [
        ("MNLI", "mnli_accuracy"),
        ("e-SNLI", "esnli_accuracy"),
        ("ANLI", "anli_accuracy"),
        ("SNLI-hard", "snli_hard_accuracy"),
        ("WANLI", "wanli_accuracy"),
        ("HANS-N", "hans_non_entailment"),
    ]
    matrix = np.array(
        [
            [metric(record, key) - metric(baseline, key) for _, key in columns]
            for record in rows
        ]
    )
    vmax = max(2.0, float(np.nanmax(np.abs(matrix))))
    im = ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if abs(value) > vmax * 0.55 else NEUTRAL_DARK
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=5.7, color=color)

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([label for label, _ in columns], rotation=30, ha="right", fontsize=6.1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([SHORT_LABEL[r["value"]] for r in rows], fontsize=6.3)
    ax.tick_params(length=0)
    ax.set_title("Control deltas vs default", loc="left", fontsize=8, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("delta (pp)", fontsize=6.3)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    add_panel_label(ax, "d")


def panel_layer_scores(ax, records: List[Dict], metrics_by_value: Dict[str, Dict]) -> None:
    all_layers = sorted(
        {
            layer_number(row["layer_tag"])
            for value in ORDER
            for row in metrics_by_value[value]["phase2_5"].get("final_layer_scores", [])
        },
        reverse=True,
    )
    layer_to_idx = {layer: idx for idx, layer in enumerate(all_layers)}
    matrix = np.full((len(all_layers), len(records)), np.nan)
    selected_cells: List[Tuple[int, int]] = []

    for col, record in enumerate(records):
        value = record["value"]
        selected = set(metrics_by_value[value]["phase2_5"].get("shortcut_layers", []))
        for row in metrics_by_value[value]["phase2_5"].get("final_layer_scores", []):
            layer = layer_number(row["layer_tag"])
            matrix[layer_to_idx[layer], col] = float(row["score"])
            if row["layer_tag"] in selected:
                selected_cells.append((layer_to_idx[layer], col))

    cmap = mpl.colormaps["YlGnBu"].copy()
    cmap.set_bad("#F1F1F1")
    im = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for i, j in selected_cells:
        ax.scatter(j, i, s=34, marker="o", facecolors="none", edgecolors=NEUTRAL_DARK, linewidths=1.0)

    ax.set_xticks(range(len(records)))
    ax.set_xticklabels([SHORT_LABEL[r["value"]] for r in records], fontsize=6.5)
    ax.set_yticks(range(len(all_layers)))
    ax.set_yticklabels([f"L{layer}" for layer in all_layers], fontsize=6.3)
    ax.tick_params(length=0)
    ax.set_title("Layer-localization score under each rank setting", loc="left", fontsize=8, pad=4)
    ax.set_xlabel("rank-control condition")
    ax.set_ylabel("encoder layer")
    ax.text(
        0.99,
        1.04,
        "circle = selected layer",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=NEUTRAL_MID,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    cbar.set_label("composite score", fontsize=6.3)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    add_panel_label(ax, "e")


def make_figure(records: List[Dict], metrics_by_value: Dict[str, Dict], out_dir: Path, dpi: int) -> None:
    load_plot_dependencies()
    setup_style()

    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.02, 1.34],
        width_ratios=[1.0, 1.0],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    panel_rank_design(ax_a, records)
    panel_tradeoff(ax_b, records)
    panel_branch_matrix(ax_c, records)
    panel_delta_heatmap(ax_d, records)
    panel_layer_scores(ax_e, records, metrics_by_value)

    fig.text(
        0.08,
        0.018,
        "Small-budget rank-control runs; seed 42; clean HANS train/eval split. Values are final accuracies; HANS-N = HANS non-entailment.",
        fontsize=6.1,
        color=NEUTRAL_MID,
    )
    fig.subplots_adjust(left=0.08, right=0.94, top=0.965, bottom=0.065, hspace=0.72, wspace=0.42)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "mr4_rank_control_figure"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    results_dir = args.results_dir.resolve()
    out_dir = (args.output_dir or (results_dir / "figure")).resolve()

    records, metrics_by_value = load_rank_records(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_source_tables(records, metrics_by_value, out_dir)
    make_figure(records, metrics_by_value, out_dir, args.dpi)

    print(f"[mr4] SVG: {out_dir / 'mr4_rank_control_figure.svg'}")
    print(f"[mr4] PNG: {out_dir / 'mr4_rank_control_figure.png'}")
    print(f"[mr4] source metrics: {out_dir / 'mr4_rank_control_figure_source_metrics.csv'}")
    print(f"[mr4] source layers: {out_dir / 'mr4_rank_control_figure_source_layers.csv'}")


if __name__ == "__main__":
    main()
