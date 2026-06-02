from typing import Dict, Optional

def print_comparison(ties_metrics: Dict, baseline_metrics: Optional[Dict]):
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    def _row(label, m):
        hans = m.get("hans", m.get("phase3", {}).get("hans", {}))
        mnli = m.get("mnli", m.get("phase3", {}).get("mnli", {}))
        esnli = m.get("esnli", m.get("phase3", {}).get("esnli", {}))
        mnli_acc = mnli.get("mnli_accuracy", 0)
        esnli_acc = esnli.get("esnli_accuracy", 0)
        h_all = hans.get("hans_overall", 0)
        h_ent = hans.get("hans_entailment", 0)
        h_nent = hans.get("hans_non_entailment", 0)
        print(f"  {label:<30s}  MNLI={mnli_acc:.4f}  ESNLI={esnli_acc:.4f}  "
              f"HANS={h_all:.4f} (ent={h_ent:.4f} nent={h_nent:.4f})")
        heur = hans.get("heuristic_breakdown", {})
        if heur:
            for k, v in heur.items():
                print(f"    {k}: {v:.4f}")

    _row("TIES-Unlearn Diff-LoRA", ties_metrics)
    if baseline_metrics:
        _row("Single LoRA Baseline", baseline_metrics)

    # Compute deltas
    t_hans = ties_metrics.get("phase3", {}).get("hans", {})
    if baseline_metrics:
        b_hans = baseline_metrics.get("hans", {})
        delta = t_hans.get("hans_overall", 0) - b_hans.get("hans_overall", 0)
        print(f"\n  HANS overall delta: {delta:+.4f}")

    print("=" * 70)
