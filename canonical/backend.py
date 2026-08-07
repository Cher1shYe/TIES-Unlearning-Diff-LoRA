"""Lazy adapter from canonical orchestration to the existing training code."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from canonical.artifacts import (
    collect_environment_metadata,
    json_ready,
    sha256_file,
    write_json,
)
from canonical.conditions import CanonicalCondition
from canonical.data import dataset_row_ids, sample_dataset
from canonical.runner import CheckpointRef
from configs.config import TrainConfig


def _json_checksum(value: Any) -> str:
    normalized = json_ready(value)
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


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
        train = sample_dataset(
            mnli["train"], self.base_config.mnli_train_size, self.base_config.data_seed
        )
        validation = sample_dataset(
            mnli["validation_matched"],
            self.base_config.mnli_val_size,
            self.base_config.data_seed,
        )
        train_ids = dataset_row_ids(train)
        validation_ids = dataset_row_ids(validation)
        hans = make_hans_split_manifest(self.base_config)
        data_manifest = {
            "schema_version": "canonical_data_manifest_v1",
            "data_seed": 42,
            "hans_split_seed": 42,
            "mnli": {
                "dataset": "nyu-mll/glue",
                "configuration": "mnli",
                "train_split": "train",
                "validation_split": "validation_matched",
                "train_row_ids": train_ids,
                "validation_row_ids": validation_ids,
                "train_row_ids_sha256": _json_checksum(train_ids),
                "validation_row_ids_sha256": _json_checksum(validation_ids),
                "train_count": len(train_ids),
                "validation_count": len(validation_ids),
            },
            "hans": hans,
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
