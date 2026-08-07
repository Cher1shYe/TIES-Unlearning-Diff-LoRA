"""Lazy adapter from canonical orchestration to the existing training code."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from canonical.artifacts import (
    collect_environment_metadata,
    sha256_file,
    write_json,
)
from canonical.access_audit import append_access_event
from canonical.conditions import CanonicalCondition
from canonical.data import sample_dataset
from canonical.data_manifest import dataset_identity_entry
from canonical.runner import CheckpointRef
from configs.config import TrainConfig

def _peak_gpu_memory_bytes() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        pass
    return None


def _reset_peak_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        pass


def _method_result(metrics: Mapping[str, Any]) -> dict[str, Any]:
    provenance = metrics.get("checkpoint_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Canonical training did not return checkpoint provenance.")
    checkpoint_hash = provenance.get("final_checkpoint_hash")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        raise ValueError("Canonical training did not return a valid final checkpoint hash.")
    return {
        "final_checkpoint_hash": checkpoint_hash,
        "peak_gpu_memory_bytes": _peak_gpu_memory_bytes(),
    }


class RealCanonicalBackend:
    """Run canonical jobs while importing ML dependencies only at call time."""

    def __init__(self, base_config: TrainConfig | None = None):
        self.base_config = deepcopy(base_config or TrainConfig())
        if self.base_config.data_seed != 42 or self.base_config.hans_split_seed != 42:
            raise ValueError("canonical_v1 requires data_seed=42 and hans_split_seed=42")
        if not self.base_config.hans_clean_split:
            raise ValueError("canonical_v1 requires hans_clean_split=True")

    def _config_for_directory(
        self,
        directory: Path,
        training_seed: int,
        condition: CanonicalCondition | None = None,
    ) -> TrainConfig:
        cfg = deepcopy(self.base_config)
        if condition is not None:
            cfg = condition.apply_to_config(cfg)
        cfg.data_seed = 42
        cfg.hans_split_seed = 42
        cfg.training_seed = int(training_seed)
        cfg.output_dir = str(directory.parent)
        cfg.experiment_name = directory.name
        cfg.data_access_log = str(directory / "data_access.jsonl")
        return cfg

    def initialize_manifests(self, output_dir: Path, protocol_path: Path) -> None:
        """Materialize data membership and the actual execution environment."""
        del protocol_path
        output_dir = Path(output_dir)
        manifests = output_dir / "manifests"

        # Deliberately delayed: importing this module and displaying CLI help do
        # not require Datasets, PyTorch, Transformers, NumPy, or network access.
        from datasets import load_dataset
        from data.dataloader import make_hans_split_manifest

        mnli = load_dataset("nyu-mll/glue", "mnli")
        canonical_config = TrainConfig()
        train = sample_dataset(mnli["train"], canonical_config.mnli_train_size, 42)
        validation = sample_dataset(mnli["validation_matched"], canonical_config.mnli_val_size, 42)
        selected_train = sample_dataset(
            mnli["train"], self.base_config.mnli_train_size, self.base_config.data_seed
        )
        selected_validation = sample_dataset(
            mnli["validation_matched"], self.base_config.mnli_val_size, self.base_config.data_seed
        )
        # This remains before any official HANS read.  The summary event below
        # binds the completed identity collection to counts and checksums.
        append_access_event(
            manifests / "data_access.jsonl",
            dataset="hans",
            split="evaluation",
            purpose="manifest_identity_only",
            event="dataset_access",
        )
        hans = make_hans_split_manifest(self.base_config)
        hans_records = {
            "build": [{"pairID": pair_id} for pair_id in hans["build_pair_ids"]],
            "dev": [{"pairID": pair_id} for pair_id in hans["dev_pair_ids"]],
            "evaluation": [
                dict(record)
                for record in hans.get(
                    "evaluation_records",
                    [{"pairID": pair_id} for pair_id in hans["evaluation_pair_ids"]],
                )
            ],
        }
        hans_entries = {
            name: dataset_identity_entry(
                hans_records[name],
                source="tommccoy1/hans",
                split=name,
                preferred_id_fields=("pairID",),
                selected_limit=(self.base_config.hans_eval_size if name == "evaluation" else None),
                seed=self.base_config.data_seed,
                strata_fields=("gold_label", "heuristic", "subcase") if name == "evaluation" else (),
            )
            for name in ("build", "dev", "evaluation")
        }
        from data.dataloader import (
            load_anli_raw,
            load_esnli_raw,
            load_snli_hard_raw,
            load_wanli_raw,
        )

        ood_specs = {
            "esnli": (load_esnli_raw, self.base_config.esnli_eval_size, "e-SNLI", ("pairID", "uid", "id", "idx")),
            "anli": (load_anli_raw, self.base_config.anli_eval_size, "facebook/anli", ("pairID", "uid", "id", "idx")),
            "snli_hard": (load_snli_hard_raw, self.base_config.snli_hard_eval_size, "snli_1.0_test_hard", ("pairID", "uid", "id", "idx")),
            "wanli": (load_wanli_raw, self.base_config.wanli_eval_size, "alisawuffles/WANLI", ("pairID", "uid", "id", "idx")),
        }
        ood_entries = {
            name: dataset_identity_entry(
                records,
                source=source,
                split="test",
                preferred_id_fields=preferred_fields,
                selected_limit=limit,
                seed=self.base_config.data_seed,
            )
            for name, (loader, limit, source, preferred_fields) in ood_specs.items()
            for records in (loader(),)
        }
        append_access_event(
            manifests / "data_access.jsonl",
            dataset="hans",
            split="evaluation",
            purpose="manifest_identity_only",
            event="manifest_identity_summary",
            identity_counts={name: entry["full_count"] for name, entry in hans_entries.items()},
            identity_checksums={name: entry["full_ids_sha256"] for name, entry in hans_entries.items()},
        )
        smoke_caps = (
            self.base_config.hans_eval_size,
            self.base_config.esnli_eval_size,
            self.base_config.anli_eval_size,
            self.base_config.snli_hard_eval_size,
            self.base_config.wanli_eval_size,
        )
        data_manifest = {
            "schema_version": "canonical_data_manifest_v2",
            "scope": "stage2_smoke" if any(cap is not None for cap in smoke_caps) else "canonical_v1",
            "data_seed": 42,
            "hans_split_seed": 42,
            "mnli": {
                "train": dataset_identity_entry(
                    train,
                    source="nyu-mll/glue:mnli",
                    split="train",
                    preferred_id_fields=("idx", "row_id", "id", "uid"),
                    selected_limit=self.base_config.mnli_train_size,
                    seed=self.base_config.data_seed,
                    selected_records=selected_train,
                ),
                "validation_matched": dataset_identity_entry(
                    validation,
                    source="nyu-mll/glue:mnli",
                    split="validation_matched",
                    preferred_id_fields=("idx", "row_id", "id", "uid"),
                    selected_limit=self.base_config.mnli_val_size,
                    seed=self.base_config.data_seed,
                    selected_records=selected_validation,
                ),
            },
            "hans": hans_entries,
            "ood": ood_entries,
        }
        write_json(manifests / "data_manifest.json", data_manifest)

        environment = collect_environment_metadata()
        try:
            import torch

            environment["cuda_runtime"] = torch.version.cuda
            if torch.cuda.is_available():
                environment["gpu"] = torch.cuda.get_device_name(0)
        except (ImportError, RuntimeError):
            pass
        try:
            driver = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            environment["cuda_driver"] = driver[0].strip() if driver else None
        except (OSError, subprocess.CalledProcessError):
            pass
        try:
            freeze = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        except (OSError, subprocess.CalledProcessError):
            freeze = None
        environment_manifest = {
            "schema_version": "canonical_environment_manifest_v1",
            **environment,
            "pip_freeze": freeze,
        }
        write_json(manifests / "environment_manifest.json", environment_manifest)

    def prepare_shared(self, training_seed: int, shared_dir: Path) -> CheckpointRef:
        shared_dir = Path(shared_dir)
        cfg = self._config_for_directory(shared_dir, training_seed)
        cfg.checkpoint_dir = str(shared_dir / "checkpoints")
        write_json(shared_dir / "config.json", asdict(cfg))

        from training.trainer import train_ties_unlearn

        _reset_peak_gpu_memory()
        result = train_ties_unlearn(cfg, stop_after_phase2=True)
        checkpoint_path = Path(result["checkpoint_path"])
        checkpoint_hash = result["checkpoint_hash"]
        if not checkpoint_path.is_file():
            raise ValueError(f"Shared training did not create checkpoint: {checkpoint_path}")
        if sha256_file(checkpoint_path) != checkpoint_hash:
            raise ValueError("Shared training returned a checkpoint hash mismatch.")
        class_prior_weights = result.get("class_prior_weights")
        if not isinstance(class_prior_weights, Mapping):
            raise ValueError("Shared training did not return class-prior weights.")
        write_json(
            shared_dir / "shared_checkpoint_metadata.json",
            {
                "checkpoint_role": "canonical_shared_phase2",
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "class_prior_weights": {
                    str(label): float(weight) for label, weight in class_prior_weights.items()
                },
            },
        )
        return CheckpointRef(checkpoint_path, checkpoint_hash)

    def run_standard(
        self, condition: CanonicalCondition, training_seed: int, run_dir: Path
    ) -> Mapping[str, Any]:
        run_dir = Path(run_dir)
        cfg = self._config_for_directory(run_dir, training_seed, condition)
        write_json(run_dir / "config.json", asdict(cfg))

        from training.baseline import train_single_lora_baseline

        _reset_peak_gpu_memory()
        metrics = train_single_lora_baseline(cfg, method_tag=condition.tag)
        return _method_result(metrics)

    def run_branch(
        self,
        condition: CanonicalCondition,
        training_seed: int,
        run_dir: Path,
        checkpoint: CheckpointRef,
    ) -> Mapping[str, Any]:
        run_dir = Path(run_dir)
        cfg = self._config_for_directory(run_dir, training_seed, condition)
        write_json(run_dir / "config.json", asdict(cfg))

        from training.trainer import train_ties_unlearn

        _reset_peak_gpu_memory()
        metrics = train_ties_unlearn(
            cfg,
            shared_checkpoint_path=str(checkpoint.path),
            method_tag=condition.tag,
            checkpoint_hash=checkpoint.sha256,
        )
        return _method_result(metrics)
