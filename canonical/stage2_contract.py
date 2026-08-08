"""Dependency-light immutable runtime contract for Stage-2 transport."""

STAGE2_SEED = 42
PRIMARY_CONDITIONS = ("standard_lora", "full_sr", "class_prior_reweight")
REPEAT_CONDITIONS = ("full_sr",)

PRIMARY_ROOT = "ties_results/stage2_smoke/colab_a100_run1"
REPEAT_ROOT = "ties_results/stage2_smoke/colab_a100_repeat_full_sr"
FREEZE_ROOT = "ties_results/stage2_smoke/freeze_bundle"
EXPECTATIONS_MEMBER = "ties_results/stage2_smoke/source_expectations.json"
EVIDENCE_INVENTORY_MEMBER = "ties_results/stage2_smoke/stage2_evidence_inventory.json"

NOTEBOOK_PATH = "notebooks/stage2_colab_a100_smoke.ipynb"
EVIDENCE_ARCHIVE_NAME = "stage2_a100_evidence.zip"
EVIDENCE_MEMBER_ROOT = "ties_results"
SAFE_EXTRACT_ROOT = "."
MONITOR_PATHS = {
    "primary": "ties_results/.stage2_monitor/colab_a100_run1.events.jsonl",
    "repeat_full_sr": "ties_results/.stage2_monitor/colab_a100_repeat_full_sr.events.jsonl",
}

METHOD_OUTPUTS = (
    "config.json",
    "run_manifest.json",
    "metrics.json",
    "hans_predictions.jsonl",
    "selected_layers.json",
    "data_access.jsonl",
    "stdout.log",
    "stderr.log",
)
SHARED_OUTPUTS = (
    "config.json",
    "run_manifest.json",
    "shared_checkpoint.json",
    "shared_checkpoint_metadata.json",
    "data_access.jsonl",
    "stdout.log",
    "stderr.log",
)

OMITTED_WEIGHT_PATHS = {
    "primary": f"{PRIMARY_ROOT}/seed_{STAGE2_SEED}/shared_phase2/checkpoints/shared.pt",
    "repeat_full_sr": f"{REPEAT_ROOT}/seed_{STAGE2_SEED}/shared_phase2/checkpoints/shared.pt",
}
