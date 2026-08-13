import os
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class LoRAConfig:
    pos_rank: int = 16
    neg_rank: int = 4
    alpha: float = 1.00
    beta: float = 0.5
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    trim_ratio: float = 0.2
    # full | naive | sign_only | trim_only | p_only  (see TIESUnlearnLoRALinear)
    merge_mode: str = "full"

@dataclass
class TrainConfig:
    # --- jtt_setup ---
    run_jtt: bool = True             # 默认为 False，用来跑 TIES_unlearn, 需要跑JTT的时候设置为True
    jtt_phase1_epochs: int = 3        # 对标你的 Phase 1
    jtt_upweight_factor: int = 4      # JTT 经典的错题放大倍数

    # --- model ---
    model_name: str = "roberta-base"
    num_labels: int = 3

    # --- data ---
    max_seq_length: int = 128
    mnli_train_size: int = 100_000
    mnli_val_size: int = 5_000
    # When True, the negative branch (Phase 2) and the kNN layer localization
    # (Phase 2.5) draw their HANS examples from the HANS *train* split, while
    # evaluation uses the disjoint HANS *evaluation* split. This removes the
    # train/eval leakage flagged in review (HANS used for both design and test).
    # Set False to reproduce the original single-file (leaky) behaviour.
    hans_clean_split: bool = True
    data_seed: int = 42
    hans_split_seed: int = 42
    # Optional evaluation caps are inactive for canonical runs.  The isolated
    # Stage-2 smoke profile sets them explicitly before constructing loaders.
    hans_eval_size: Optional[int] = None
    esnli_eval_size: Optional[int] = None
    anli_eval_size: Optional[int] = None
    snli_hard_eval_size: Optional[int] = None
    wanli_eval_size: Optional[int] = None
    data_access_log: Optional[str] = None

    # --- optimiser ---
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.5
    fp16: bool = False
    training_seed: int = 42

    # --- LoRA ---
    target_modules: Tuple[str, ...] = ("query", "value")
    pos_rank: int = 16
    neg_rank: int = 4
    alpha: float = 1.25
    beta: float = 0.5
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    trim_ratio: float = 0.2

    # --- layer-wise shortcut analysis ---
    enable_layerwise_analysis: bool = True
    kl_batches: int = 8
    kl_topk_candidates: int = 8
    layer_selection_topk: int = 4
    knn_mode: str = "pd_and_early_wrong"
    knn_k: int = 10
    knn_stability_window: int = 2
    knn_batch_size: int = 64
    knn_ref_mnli: int = 2000
    knn_ref_hans_entail: int = 1000
    knn_ref_hans_non_entail: int = 1000
    knn_query_mnli: int = 500
    knn_query_hans_entail: int = 750
    knn_query_hans_non_entail: int = 750
    kl_weight: float = 0.5
    knn_weight: float = 0.5
    pd_weight: float = 0.25
    early_wrong_weight: float = 0.25
    no_ties_ablation: bool = False

    # --- TIES merge / ablation controls ---
    # Which mask(s) to apply in the Phase-3 merge. "full" = sign+trim (paper default).
    # Use "naive"/"sign_only"/"trim_only" to isolate each mask, "p_only" to disable
    # subtraction entirely. See models/ties_lora.py:TIESUnlearnLoRALinear.
    merge_mode: str = "full"
    # If True, Phase 2.5 picks `layer_selection_topk` layers at random (seeded) instead
    # of running the KL/kNN analysis — the "randomly selected layers" ablation.
    random_layer_selection: bool = False
    # If True, Phase 3 keeps the P-branch of the *subtracted* (enable_ties) layers FROZEN,
    # so debias fine-tuning cannot rebuild the shortcut direction we just removed; only the
    # head + the non-subtracted layers' P are tuned to recover MNLI. Fix (B) for the
    # "Phase-3 re-injects the shortcut" failure mode. Default False = original behaviour.
    phase3_freeze_subtracted_p: bool = False
    # Fix (A) — NOW PART OF THE DEFAULT METHOD. Re-weight the Phase-3 debias FT loss using
    # the frozen N (shortcut) path. Examples the N path already classifies correctly with
    # high confidence (i.e. solvable by the shortcut) get down-weighted by w=(1-p_N)^gamma,
    # so recovering MNLI cannot re-learn the shortcut from the data. This is what makes the
    # subtraction give a net gain (H-nent 22.3 -> 30.8). Set reweight=False to ablate it.
    phase3_debias_reweight: bool = True
    phase3_reweight_gamma: float = 2.0
    # Canonical modes: none | n_guided | class_prior. None preserves the legacy
    # phase3_debias_reweight flag used by the exploratory drivers.
    phase3_weighting: Optional[str] = None
    # Extra final P-only/N-only evaluation for rank-control sensitivity runs.
    # Disabled by default because it adds evaluation passes to every training run.
    record_branch_only_metrics: bool = False

    # --- debiasing baselines (PoE / z-filtering / bias model) ---
    bias_model_epochs: int = 3       # epochs to train the hypothesis-only bias model
    zfilter_drop_ratio: float = 0.2  # fraction of most bias-aligned train examples to drop
    poe_bias_scale: float = 1.0      # weight on the bias log-probs in the PoE objective

    # --- training phases ---
    phase1_epochs: int = 3      # Learn task        (head + P, N frozen)
    phase2_epochs: int = 2      # Capture shortcut  (N only, P & head frozen)
    phase3_epochs: int = 5      # TIES fine-tuning  (head + P, N frozen)
    # Phase2 数据混合：MNLI 占比 + HANS entailment 占比 (其余)
    phase2_mnli_mix_ratio: float = 0.10
    phase2_epoch_batches: int = 3125
    neg_lr_mult: float = 2.0    # Phase-1 lr multiplier for N
    neg_lr_mult_p3: float = 0.3 # Phase-3 lr multiplier for N (keep small)
    neg_reg_lambda: float = 0.001

    # --- baseline comparison ---
    run_baseline: bool = False   # Also train single-LoRA baseline

    # --- output ---
    output_dir: str = "./ties_unlearn_results"
    save_checkpoints: bool = True
    experiment_name: str = "ties_unlearn_diff_lora"
    save_checkpoints_per_phase: bool = True
    checkpoint_dir: Optional[str] = None

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for name in ("data_seed", "hans_split_seed", "training_seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        valid_weighting_modes = {None, "none", "n_guided", "class_prior"}
        if self.phase3_weighting not in valid_weighting_modes:
            raise ValueError(
                "phase3_weighting must be one of None, 'none', 'n_guided', "
                f"or 'class_prior', got {self.phase3_weighting!r}."
            )
        valid_knn_modes = {"off", "pd", "acc", "pd_and_early_wrong"}
        if self.enable_layerwise_analysis and self.kl_batches < 1:
            raise ValueError("kl_batches must be >= 1 when layer-wise analysis is enabled.")
        if self.kl_topk_candidates < 1:
            raise ValueError("kl_topk_candidates must be >= 1.")
        if self.layer_selection_topk < 0:
            raise ValueError("layer_selection_topk must be >= 0.")
        if self.layer_selection_topk > self.kl_topk_candidates:
            raise ValueError("layer_selection_topk cannot exceed kl_topk_candidates.")
        if self.knn_mode not in valid_knn_modes:
            raise ValueError(f"knn_mode must be one of {sorted(valid_knn_modes)}, got {self.knn_mode!r}.")

        sample_counts = [
            self.knn_ref_mnli,
            self.knn_ref_hans_entail,
            self.knn_ref_hans_non_entail,
            self.knn_query_mnli,
            self.knn_query_hans_entail,
            self.knn_query_hans_non_entail,
        ]
        if any(count < 0 for count in sample_counts):
            raise ValueError("All kNN sample counts must be non-negative.")
        if self.knn_k < 1:
            raise ValueError("knn_k must be >= 1.")
        if self.knn_stability_window < 1:
            raise ValueError("knn_stability_window must be >= 1.")
        if not (0.0 <= self.phase2_mnli_mix_ratio <= 1.0):
            raise ValueError("phase2_mnli_mix_ratio must be in [0, 1].")
        if self.phase2_epoch_batches < 1:
            raise ValueError("phase2_epoch_batches must be >= 1.")

        valid_merge_modes = {"full", "naive", "sign_only", "trim_only", "p_only"}
        if self.merge_mode not in valid_merge_modes:
            raise ValueError(f"merge_mode must be one of {sorted(valid_merge_modes)}, got {self.merge_mode!r}.")
        if not (0.0 <= self.zfilter_drop_ratio < 1.0):
            raise ValueError("zfilter_drop_ratio must be in [0, 1).")
        if self.bias_model_epochs < 1:
            raise ValueError("bias_model_epochs must be >= 1.")

    @property
    def seed(self) -> int:
        """Legacy alias for training randomness only.

        Data selection intentionally never consults this property.
        """
        return self.training_seed

    @seed.setter
    def seed(self, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("seed must be a non-negative integer.")
        self.training_seed = value
