"""NegMerge-style sign-consensus merging baseline.

Reviewer item #7: "sign-consensual unlearning or merging methods such as NegMerge".

NegMerge (Kim et al., 2024) negates a task vector while keeping only the
sign-consensual components. We instantiate a *single-source* sign-consensus
variant inside our own dual-path framework so it is directly comparable to the
full method:

    * Phase 1 / Phase 2 train the P and N branches exactly as usual.
    * The merge is GLOBAL (every layer), uses the sign-consensus mask only
      (no magnitude trimming, no KL/kNN layer localization), and there is NO
      Phase-3 debiasing fine-tuning — it is a pure task-arithmetic merge.

This isolates "sign-consensus negation" as a baseline; the full TIES-Unlearning
method differs by adding magnitude trimming, layer localization, and a short
debiasing fine-tune.

NOTE: the original NegMerge builds consensus across *multiple* task vectors. Here
we use a single shortcut vector (one N branch), so this is an adaptation rather
than a verbatim reimplementation; see the discussion in run_baselines.py.
"""
from copy import deepcopy

from configs.config import TrainConfig
from training.trainer import train_ties_unlearn


def train_negmerge_baseline(cfg: TrainConfig):
    print("\n[NegMerge Baseline] global sign-consensus merge, no trim, no Phase-3 FT")
    nm_cfg = deepcopy(cfg)
    nm_cfg.merge_mode = "sign_only"          # sign-consensus mask only
    nm_cfg.trim_ratio = 1.0                  # disable magnitude trimming (no-op for sign_only)
    nm_cfg.enable_layerwise_analysis = False  # global merge over all layers
    nm_cfg.random_layer_selection = False
    nm_cfg.no_ties_ablation = False
    nm_cfg.phase3_epochs = 0                 # pure merge, no debiasing fine-tuning
    nm_cfg.run_jtt = False
    if not nm_cfg.experiment_name or nm_cfg.experiment_name == cfg.experiment_name:
        nm_cfg.experiment_name = "negmerge"
    metrics = train_ties_unlearn(nm_cfg)
    metrics["method"] = "NegMerge (sign-consensus, global)"
    return metrics
