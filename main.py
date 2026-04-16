import os
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs.config import TrainConfig
from training.trainer import train_ties_unlearn
from training.baseline import train_single_lora_baseline
from training.train_jtt import train_jtt_baseline
from utils.logger import print_comparison

def main():
    cfg = TrainConfig()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║    TIES-Unlearning Diff-LoRA  (TIES + Unlearning merge)    ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Model:      {cfg.model_name:<46s}║")
    print(f"║  P rank:     {cfg.pos_rank:<46d}║")
    print(f"║  N rank:     {cfg.neg_rank:<46d}║")
    print(f"║  alpha:      {cfg.alpha:<46.2f}║")
    print(f"║  beta:       {cfg.beta:<46.2f}║")
    print(f"║  trim_ratio: {cfg.trim_ratio:<46.2f}║")
    print(f"║  knn_mode:   {cfg.knn_mode:<46s}║")
    print(f"║  KL cand.:   {cfg.kl_topk_candidates:<46d}║")
    print(f"║  shortcuts:  {cfg.layer_selection_topk:<46d}║")
    print(f"║  Phases:     {cfg.phase1_epochs}+{cfg.phase2_epochs}+{cfg.phase3_epochs}"
          f" = {cfg.phase1_epochs+cfg.phase2_epochs+cfg.phase3_epochs} epochs"
          f"{'':<33s}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if cfg.run_jtt:
        print("====== Running JTT Baseline (3-0-5 configuration) ======")
        ties_metrics = train_jtt_baseline(cfg)
    else:
        print("====== Running TIES-Unlearning ======")
        ties_metrics = train_ties_unlearn(cfg)
    
    # Run baseline
    baseline_metrics = None
    if cfg.run_baseline:
        baseline_metrics = train_single_lora_baseline(cfg)

    # Compare
    print_comparison(ties_metrics, baseline_metrics)

    # Save combined summary
    summary = {
        "ties_unlearn": ties_metrics,
        "baseline": baseline_metrics,
    }
    summary_path = os.path.join(cfg.output_dir, "final_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {cfg.output_dir}/")


if __name__ == "__main__":
    main()