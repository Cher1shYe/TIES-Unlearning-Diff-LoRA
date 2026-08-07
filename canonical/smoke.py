"""Immutable, isolated configuration for Stage-2 smoke validation."""

from pathlib import Path

from configs.config import TrainConfig


SMOKE_PROFILE_NAME = "stage2_smoke_v1"
PRIMARY_CONDITIONS = ("standard_lora", "full_sr", "class_prior_reweight")
REPEAT_CONDITIONS = ("full_sr",)


def build_smoke_config(output_dir: Path) -> TrainConfig:
    """Build the fixed small-budget configuration without changing canonical defaults."""
    return TrainConfig(
        max_seq_length=64,
        mnli_train_size=96,
        mnli_val_size=96,
        batch_size=8,
        fp16=False,
        data_seed=42,
        hans_split_seed=42,
        training_seed=42,
        phase1_epochs=1,
        phase2_epochs=1,
        phase3_epochs=1,
        phase2_epoch_batches=4,
        kl_batches=1,
        kl_topk_candidates=2,
        layer_selection_topk=1,
        knn_k=3,
        knn_ref_mnli=16,
        knn_query_mnli=8,
        knn_ref_hans_entail=8,
        knn_query_hans_entail=4,
        knn_ref_hans_non_entail=8,
        knn_query_hans_non_entail=4,
        hans_eval_size=384,
        esnli_eval_size=128,
        anli_eval_size=128,
        snli_hard_eval_size=128,
        wanli_eval_size=128,
        output_dir=str(output_dir),
    )


def assert_stage2_output_path(output_dir: Path, repo_root: Path) -> Path:
    """Resolve a smoke output path and keep it disjoint from canonical v1 results."""
    resolved = (repo_root / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    if "canonical_v1" in resolved.parts:
        raise ValueError("Stage 2 smoke output must not use canonical_v1")
    return resolved
