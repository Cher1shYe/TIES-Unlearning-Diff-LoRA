"""Resumable orchestration for the frozen canonical experiment matrix."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Protocol, Sequence

from canonical.artifacts import collect_git_metadata, sha256_file, write_json
from canonical.conditions import (
    CANONICAL_TRAINING_SEEDS,
    CONDITIONS,
    CanonicalCondition,
    rotated_condition_order,
)


@dataclass(frozen=True)
class CheckpointRef:
    path: Path
    sha256: str

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path))


class CanonicalBackend(Protocol):
    def initialize_manifests(self, output_dir: Path, protocol_path: Path) -> None: ...
    def prepare_shared(self, training_seed: int, shared_dir: Path) -> CheckpointRef: ...
    def run_standard(
        self, condition: CanonicalCondition, training_seed: int, run_dir: Path
    ) -> Mapping[str, Any]: ...
    def run_branch(
        self,
        condition: CanonicalCondition,
        training_seed: int,
        run_dir: Path,
        checkpoint: CheckpointRef,
    ) -> Mapping[str, Any]: ...


_METHOD_OUTPUTS = (
    "config.json",
    "run_manifest.json",
    "metrics.json",
    "hans_predictions.jsonl",
    "selected_layers.json",
    "data_access.jsonl",
    "stdout.log",
    "stderr.log",
)
_SHARED_OUTPUTS = (
    "config.json",
    "run_manifest.json",
    "shared_checkpoint.json",
    "shared_checkpoint_metadata.json",
    "data_access.jsonl",
    "stdout.log",
    "stderr.log",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _assert_clean_git(metadata: Mapping[str, Any]) -> None:
    if metadata.get("dirty"):
        raise ValueError("Canonical execution requires a clean Git working tree.")
    if not metadata.get("commit"):
        raise ValueError("Canonical execution requires a recorded Git commit.")


def _prepare_output_directory(
    output_dir: Path, fresh: bool, *, allowed_existing_entries: Sequence[str] = ()
) -> None:
    if fresh:
        allowed = set(allowed_existing_entries)
        existing = tuple(output_dir.iterdir()) if output_dir.exists() else ()
        if any(entry.name not in allowed for entry in existing):
            raise ValueError("--fresh requires a new or empty output directory")
        output_dir.mkdir(parents=True, exist_ok=True)
        return
    if not output_dir.exists():
        raise ValueError("Canonical output directory does not exist; initialize it with --fresh.")


def _write_or_validate_protocol_snapshot(
    protocol_path: Path, output_dir: Path, fresh: bool
) -> str:
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Frozen protocol not found: {protocol_path}")
    protocol_hash = sha256_file(protocol_path)
    snapshot_dir = output_dir / "protocol_snapshot"
    snapshot_path = snapshot_dir / "FROZEN_EXPERIMENT_PROTOCOL.md"
    hash_path = snapshot_dir / "protocol_sha256.txt"
    if fresh:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(protocol_path, snapshot_path)
        hash_path.write_text(protocol_hash + "\n", encoding="utf-8", newline="\n")
    if not snapshot_path.is_file() or not hash_path.is_file():
        raise ValueError("Canonical protocol snapshot is incomplete.")
    recorded = hash_path.read_text(encoding="utf-8").strip()
    snapshot_hash = sha256_file(snapshot_path)
    if recorded != protocol_hash or snapshot_hash != protocol_hash:
        raise ValueError("Canonical protocol hash does not match the frozen snapshot.")
    return protocol_hash


def _validate_manifests(output_dir: Path) -> dict[str, str]:
    hashes = {}
    for name in ("data_manifest.json", "environment_manifest.json"):
        path = output_dir / "manifests" / name
        if not path.is_file():
            raise ValueError(f"Canonical manifest is missing: {path}")
        _read_json(path)
        hashes[name] = sha256_file(path)
    return hashes


def _write_or_validate_run_matrix(
    output_dir: Path,
    seeds: tuple[int, ...],
    condition_tags: tuple[str, ...],
    matrix_schema_version: str,
    fresh: bool,
) -> None:
    path = output_dir / "manifests" / "run_matrix.json"
    expected = {
        "schema_version": matrix_schema_version,
        "training_seeds": list(seeds),
        "condition_orders": {
            str(seed): [
                tag for tag in rotated_condition_order(seed) if tag in condition_tags
            ]
            for seed in seeds
        },
    }
    if fresh:
        write_json(path, expected)
    elif _read_json(path) != expected:
        raise ValueError("Canonical run matrix differs from the frozen seed/condition order.")


def _validated_condition_tags(condition_tags: Sequence[str]) -> tuple[str, ...]:
    tags = tuple(condition_tags)
    if not tags:
        raise ValueError("Canonical condition matrix requires at least one condition tag.")
    unknown = [tag for tag in tags if tag not in CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown canonical condition tag(s): {unknown}")
    if len(set(tags)) != len(tags):
        raise ValueError("Canonical condition matrix must not repeat condition tags.")
    return tags


def _artifact_hashes(base_dir: Path, relative_paths: Iterable[str]) -> dict[str, str]:
    hashes = {}
    for relative in relative_paths:
        path = base_dir / relative
        if not path.is_file():
            raise ValueError(f"Required canonical artifact is missing: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def _is_checksum_valid_success(run_dir: Path) -> bool:
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        return False
    try:
        status = _read_json(status_path)
        hashes = status["output_hashes"]
        if status.get("state") != "success" or not isinstance(hashes, dict) or not hashes:
            return False
        for relative, expected in hashes.items():
            path = run_dir / relative
            if not path.is_file() or sha256_file(path) != expected:
                return False
        return True
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _base_run_manifest(
    *,
    role: str,
    training_seed: int,
    method_tag: str | None,
    protocol_hash: str,
    git_metadata: Mapping[str, Any],
    manifest_hashes: Mapping[str, str],
    command: Sequence[str],
    shared_checkpoint: CheckpointRef | None,
) -> dict[str, Any]:
    return {
        "schema_version": "canonical_run_manifest_v1",
        "role": role,
        "method_tag": method_tag,
        "data_seed": 42,
        "hans_split_seed": 42,
        "training_seed": training_seed,
        "protocol_sha256": protocol_hash,
        "data_manifest_sha256": manifest_hashes["data_manifest.json"],
        "environment_manifest_sha256": manifest_hashes["environment_manifest.json"],
        "git": dict(git_metadata),
        "command": list(command),
        "shared_phase2_checkpoint": (
            None
            if shared_checkpoint is None
            else {"path": str(shared_checkpoint.path), "sha256": shared_checkpoint.sha256}
        ),
        "started_at": _utc_now(),
        "finished_at": None,
        "result": None,
    }


def _run_with_logs(callback, stdout_path: Path, stderr_path: Path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as stderr_handle:
        with redirect_stdout(stdout_handle), redirect_stderr(stderr_handle):
            return callback()


def _failed_status(started_at: str, started_clock: float, error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": "canonical_status_v1",
        "state": "failed",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "wall_time_seconds": time.monotonic() - started_clock,
        "exit_status": 1,
        "peak_gpu_memory_bytes": None,
        "error_type": type(error).__name__,
        "error": str(error),
        "output_hashes": {},
    }


def _prepare_shared(
    backend: CanonicalBackend,
    training_seed: int,
    shared_dir: Path,
    *,
    protocol_hash: str,
    git_metadata: Mapping[str, Any],
    manifest_hashes: Mapping[str, str],
    command: Sequence[str],
) -> CheckpointRef:
    if _is_checksum_valid_success(shared_dir):
        payload = _read_json(shared_dir / "shared_checkpoint.json")
        checkpoint = CheckpointRef(shared_dir / payload["path_relative"], payload["sha256"])
        if checkpoint.path.is_file() and sha256_file(checkpoint.path) == checkpoint.sha256:
            return checkpoint

    shared_dir.mkdir(parents=True, exist_ok=True)
    manifest = _base_run_manifest(
        role="shared_phase2",
        training_seed=training_seed,
        method_tag=None,
        protocol_hash=protocol_hash,
        git_metadata=git_metadata,
        manifest_hashes=manifest_hashes,
        command=command,
        shared_checkpoint=None,
    )
    write_json(shared_dir / "run_manifest.json", manifest)
    started_at = manifest["started_at"]
    started_clock = time.monotonic()
    write_json(
        shared_dir / "status.json",
        {"schema_version": "canonical_status_v1", "state": "running", "started_at": started_at},
    )
    try:
        checkpoint = _run_with_logs(
            lambda: backend.prepare_shared(training_seed, shared_dir),
            shared_dir / "stdout.log",
            shared_dir / "stderr.log",
        )
        checkpoint = CheckpointRef(checkpoint.path, checkpoint.sha256)
        if not checkpoint.path.is_file():
            raise ValueError(f"Shared checkpoint is missing: {checkpoint.path}")
        actual_hash = sha256_file(checkpoint.path)
        if actual_hash != checkpoint.sha256:
            raise ValueError(
                f"Shared checkpoint hash mismatch: expected {checkpoint.sha256}, got {actual_hash}."
            )
        relative_checkpoint = os.path.relpath(checkpoint.path, shared_dir)
        write_json(
            shared_dir / "shared_checkpoint.json",
            {"path_relative": relative_checkpoint, "sha256": checkpoint.sha256},
        )
        manifest["finished_at"] = _utc_now()
        manifest["result"] = {"checkpoint_sha256": checkpoint.sha256}
        write_json(shared_dir / "run_manifest.json", manifest)
        relative_outputs = list(_SHARED_OUTPUTS) + [relative_checkpoint]
        output_hashes = _artifact_hashes(shared_dir, relative_outputs)
        write_json(
            shared_dir / "status.json",
            {
                "schema_version": "canonical_status_v1",
                "state": "success",
                "started_at": started_at,
                "finished_at": _utc_now(),
                "wall_time_seconds": time.monotonic() - started_clock,
                "exit_status": 0,
                "peak_gpu_memory_bytes": None,
                "error_type": None,
                "error": None,
                "output_hashes": output_hashes,
            },
        )
        return checkpoint
    except BaseException as error:
        with (shared_dir / "stderr.log").open("a", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        write_json(shared_dir / "status.json", _failed_status(started_at, started_clock, error))
        raise


def _execute_method(
    backend: CanonicalBackend,
    condition: CanonicalCondition,
    training_seed: int,
    run_dir: Path,
    *,
    shared_checkpoint: CheckpointRef,
    protocol_hash: str,
    git_metadata: Mapping[str, Any],
    manifest_hashes: Mapping[str, str],
    command: Sequence[str],
) -> bool:
    if _is_checksum_valid_success(run_dir):
        return False
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _base_run_manifest(
        role="standard_lora" if condition.standard_lora else "dual_adapter_branch",
        training_seed=training_seed,
        method_tag=condition.tag,
        protocol_hash=protocol_hash,
        git_metadata=git_metadata,
        manifest_hashes=manifest_hashes,
        command=command,
        shared_checkpoint=None if condition.standard_lora else shared_checkpoint,
    )
    write_json(run_dir / "run_manifest.json", manifest)
    started_at = manifest["started_at"]
    started_clock = time.monotonic()
    write_json(
        run_dir / "status.json",
        {"schema_version": "canonical_status_v1", "state": "running", "started_at": started_at},
    )
    try:
        if condition.standard_lora:
            callback = lambda: backend.run_standard(condition, training_seed, run_dir)
        else:
            callback = lambda: backend.run_branch(
                condition, training_seed, run_dir, shared_checkpoint
            )
        result = _run_with_logs(callback, run_dir / "stdout.log", run_dir / "stderr.log")
        manifest["finished_at"] = _utc_now()
        manifest["result"] = dict(result or {})
        write_json(run_dir / "run_manifest.json", manifest)
        output_hashes = _artifact_hashes(run_dir, _METHOD_OUTPUTS)
        write_json(
            run_dir / "status.json",
            {
                "schema_version": "canonical_status_v1",
                "state": "success",
                "started_at": started_at,
                "finished_at": _utc_now(),
                "wall_time_seconds": time.monotonic() - started_clock,
                "exit_status": 0,
                "peak_gpu_memory_bytes": (result or {}).get("peak_gpu_memory_bytes"),
                "error_type": None,
                "error": None,
                "output_hashes": output_hashes,
            },
        )
        return True
    except BaseException as error:
        with (run_dir / "stderr.log").open("a", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        write_json(run_dir / "status.json", _failed_status(started_at, started_clock, error))
        raise


def run_condition_matrix(
    protocol_path: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    backend: CanonicalBackend,
    *,
    seeds: Sequence[int],
    condition_tags: Sequence[str],
    matrix_schema_version: str,
    fresh: bool = False,
    git_metadata: Mapping[str, Any] | None = None,
    command: Sequence[str] | None = None,
    repo_root: os.PathLike[str] | str = ".",
) -> dict[str, Any]:
    """Execute or resume a validated subset of the frozen condition matrix."""
    protocol_path = Path(protocol_path).resolve()
    output_dir = Path(output_dir).resolve()
    seeds_tuple = tuple(int(seed) for seed in seeds)
    tags_tuple = _validated_condition_tags(condition_tags)
    for seed in seeds_tuple:
        rotated_condition_order(seed)
    git_info = dict(git_metadata or collect_git_metadata(repo_root))
    _assert_clean_git(git_info)
    _prepare_output_directory(
        output_dir,
        fresh,
        allowed_existing_entries=("commands.json",)
        if fresh and (output_dir / "commands.json").is_file()
        else (),
    )
    protocol_hash = _write_or_validate_protocol_snapshot(protocol_path, output_dir, fresh)
    if fresh:
        backend.initialize_manifests(output_dir, protocol_path)
    manifest_hashes = _validate_manifests(output_dir)
    _write_or_validate_run_matrix(
        output_dir, seeds_tuple, tags_tuple, matrix_schema_version, fresh
    )
    invocation = tuple(command or sys.argv)

    executed = []
    skipped = []
    for training_seed in seeds_tuple:
        seed_dir = output_dir / f"seed_{training_seed}"
        shared = _prepare_shared(
            backend,
            training_seed,
            seed_dir / "shared_phase2",
            protocol_hash=protocol_hash,
            git_metadata=git_info,
            manifest_hashes=manifest_hashes,
            command=invocation,
        )
        for tag in rotated_condition_order(training_seed):
            if tag not in tags_tuple:
                continue
            condition = CONDITIONS[tag]
            did_run = _execute_method(
                backend,
                condition,
                training_seed,
                seed_dir / tag,
                shared_checkpoint=shared,
                protocol_hash=protocol_hash,
                git_metadata=git_info,
                manifest_hashes=manifest_hashes,
                command=invocation,
            )
            target = {"training_seed": training_seed, "method_tag": tag}
            (executed if did_run else skipped).append(target)
    return {"protocol_sha256": protocol_hash, "executed": executed, "skipped": skipped}


def run_core(
    protocol_path: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    backend: CanonicalBackend,
    *,
    fresh: bool = False,
    seeds: Sequence[int] = CANONICAL_TRAINING_SEEDS,
    git_metadata: Mapping[str, Any] | None = None,
    command: Sequence[str] | None = None,
    repo_root: os.PathLike[str] | str = ".",
) -> dict[str, Any]:
    """Execute or resume the frozen core matrix, stopping on the first failure."""
    return run_condition_matrix(
        protocol_path,
        output_dir,
        backend,
        seeds=seeds,
        condition_tags=tuple(CONDITIONS),
        matrix_schema_version="canonical_run_matrix_v1",
        fresh=fresh,
        git_metadata=git_metadata,
        command=command,
        repo_root=repo_root,
    )
