"""Build and independently verify the no-weight Stage-2 evidence transport."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any
import zipfile

from canonical.artifacts import sha256_file, write_json
from canonical.freeze import (
    _commands, _manifest_commits, _safe_relative, _strict_environment, _strict_json,
    verify_freeze_bundle,
)
from canonical.stage2_validation import compare_a100_repeat, validate_smoke_root


INVENTORY_MEMBER = "ties_results/stage2_smoke/stage2_evidence_inventory.json"
_PRIMARY = "ties_results/stage2_smoke/colab_a100_run1"
_REPEAT = "ties_results/stage2_smoke/colab_a100_repeat_full_sr"
_FREEZE = "ties_results/stage2_smoke/freeze_bundle"
_EXPECTATIONS = "ties_results/stage2_smoke/source_expectations.json"
_MONITORS = {
    "primary": "ties_results/.stage2_monitor/colab_a100_run1.events.jsonl",
    "repeat_full_sr": "ties_results/.stage2_monitor/colab_a100_repeat_full_sr.events.jsonl",
}
_WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
_FAILED_EVENTS = {"CRASHED", "HARD_TIMEOUT", "FATAL_PATTERN"}
_ALLOWED_EVENTS = {"STARTED", "STATUS_CHECK", "PROGRESS", "STALL_WARNING", "COMPLETED"}
_POLICY = {"check_interval_seconds": 300, "stall_seconds": 3600, "hard_timeout_seconds": 43200}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _strict_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {label}: {value}")

    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required in {label}")
    return value


def _sha256_bytes(payload: bytes) -> str:
    from hashlib import sha256

    return sha256(payload).hexdigest()


def _safe_member(name: str) -> bool:
    return isinstance(name, str) and _safe_relative(name) and name.startswith("ties_results/")


def _zip_symlink(info: zipfile.ZipInfo) -> bool:
    return ((info.external_attr >> 16) & 0o170000) == 0o120000


def validate_monitor_evidence(path: Path, *, expected_command: list[str]) -> dict[str, Any]:
    """Validate the frozen production-monitor JSONL contract."""
    path = Path(path)
    if not isinstance(expected_command, list) or not expected_command or not all(isinstance(item, str) and item for item in expected_command):
        raise ValueError("monitor expected command is invalid")
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"monitor evidence is unreadable: {path}") from error
    if not lines:
        raise ValueError("monitor evidence is empty")
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise ValueError(f"monitor evidence contains an empty record at line {line_number}")
        record = _strict_json_bytes(line.encode("utf-8"), label=f"monitor line {line_number}")
        event = record.get("event")
        if event in _FAILED_EVENTS:
            raise ValueError(f"monitor evidence records terminal failure {event}")
        if event not in _ALLOWED_EVENTS:
            raise ValueError(f"monitor evidence event is invalid: {event!r}")
        for field in ("timestamp", "elapsed_seconds"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"monitor {field} is invalid")
        if record.get("command") != expected_command:
            raise ValueError("monitor command does not bind commands.json")
        cwd = record.get("cwd")
        if not isinstance(cwd, str) or not cwd or not Path(cwd).is_absolute():
            raise ValueError("monitor cwd must be absolute")
        records.append(record)
    if records[0].get("event") != "STARTED" or records[0].get("policy") != _POLICY:
        raise ValueError("monitor STARTED policy is not the frozen production policy")
    if records[-1].get("event") != "COMPLETED" or records[-1].get("returncode") != 0:
        raise ValueError("monitor must end with successful COMPLETED")
    cwd = records[0]["cwd"]
    if any(record["cwd"] != cwd for record in records):
        raise ValueError("monitor cwd changed during execution")
    for previous, current in zip(records, records[1:]):
        if current["timestamp"] < previous["timestamp"] or current["elapsed_seconds"] < previous["elapsed_seconds"]:
            raise ValueError("monitor timestamps/elapsed_seconds are out of order")
    return {"state": "pass", "events": len(records), "cwd": cwd}


def _run_matrix(root: Path) -> tuple[list[int], dict[str, list[str]]]:
    matrix = _strict_json(root / "manifests" / "run_matrix.json")
    seeds, orders = matrix.get("training_seeds"), matrix.get("condition_orders")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds) or not isinstance(orders, dict):
        raise ValueError("evidence run matrix is invalid")
    normalized: dict[str, list[str]] = {}
    for seed in seeds:
        order = orders.get(str(seed))
        if not isinstance(order, list) or not order or not all(isinstance(tag, str) and tag for tag in order):
            raise ValueError("evidence condition matrix is invalid")
        normalized[str(seed)] = order
    if set(orders) != set(normalized):
        raise ValueError("evidence condition matrix contains unexpected seeds")
    return seeds, normalized


def _global_run_files(root_rel: str) -> set[str]:
    return {
        f"{root_rel}/commands.json",
        f"{root_rel}/protocol_snapshot/FROZEN_EXPERIMENT_PROTOCOL.md",
        f"{root_rel}/protocol_snapshot/protocol_sha256.txt",
        f"{root_rel}/manifests/data_manifest.json",
        f"{root_rel}/manifests/environment_manifest.json",
        f"{root_rel}/manifests/run_matrix.json",
        f"{root_rel}/manifests/data_access.jsonl",
        f"{root_rel}/stage2_validation.json",
        f"{root_rel}/stage2_validation.md",
    }


def _derive_run_evidence(repo_root: Path, root_rel: str, *, weights_may_be_omitted: bool = False) -> tuple[set[str], list[dict[str, str]]]:
    root = repo_root / PurePosixPath(root_rel)
    required = _global_run_files(root_rel)
    omitted: list[dict[str, str]] = []
    seeds, orders = _run_matrix(root)
    for seed in seeds:
        for tag in ["shared_phase2", *orders[str(seed)]]:
            directory = root / f"seed_{seed}" / tag
            status = _strict_json(directory / "status.json")
            status_keys = {
                "schema_version", "state", "started_at", "finished_at", "wall_time_seconds",
                "exit_status", "peak_gpu_memory_bytes", "error_type", "error", "output_hashes",
            }
            wall_time = status.get("wall_time_seconds")
            peak = status.get("peak_gpu_memory_bytes")
            if (
                set(status) != status_keys
                or status.get("schema_version") != "canonical_status_v1"
                or status.get("state") != "success"
                or not isinstance(status.get("started_at"), str) or not status["started_at"]
                or not isinstance(status.get("finished_at"), str) or not status["finished_at"]
                or isinstance(wall_time, bool) or not isinstance(wall_time, (int, float)) or not math.isfinite(float(wall_time)) or wall_time < 0
                or status.get("exit_status") != 0
                or not (peak is None or (isinstance(peak, int) and not isinstance(peak, bool) and peak >= 0))
                or status.get("error_type") is not None or status.get("error") is not None
                or not isinstance(status.get("output_hashes"), dict)
            ):
                raise ValueError(f"evidence status is invalid: {directory}")
            prefix = f"{root_rel}/seed_{seed}/{tag}"
            required.add(f"{prefix}/status.json")
            for relative, expected_hash in status["output_hashes"].items():
                if not isinstance(relative, str) or not _safe_relative(relative) or not _is_sha256(expected_hash):
                    raise ValueError(f"evidence status contains unsafe output metadata: {relative!r}")
                full = f"{prefix}/{relative}"
                if PurePosixPath(relative).suffix.casefold() in _WEIGHT_SUFFIXES:
                    artifact = directory / PurePosixPath(relative)
                    if not weights_may_be_omitted and (not artifact.is_file() or artifact.is_symlink() or sha256_file(artifact) != expected_hash):
                        raise ValueError(f"evidence status artifact hash mismatch: {artifact}")
                    omitted.append({"path": full, "sha256": expected_hash, "reason": "model_weight_excluded"})
                else:
                    artifact = directory / PurePosixPath(relative)
                    if not artifact.is_file() or artifact.is_symlink() or sha256_file(artifact) != expected_hash:
                        raise ValueError(f"evidence status artifact hash mismatch: {artifact}")
                    required.add(full)
    return required, omitted


def _validate_omitted_metadata(repo_root: Path, omitted: list[dict[str, str]]) -> None:
    omitted_map = {entry["path"]: entry["sha256"] for entry in omitted}
    if len(omitted_map) != len(omitted) or not omitted:
        raise ValueError("omitted weight inventory is empty or duplicated")
    for root_rel in (_PRIMARY, _REPEAT):
        shared_rel = f"{root_rel}/seed_42/shared_phase2"
        checkpoint = _strict_json(repo_root / shared_rel / "shared_checkpoint.json")
        metadata = _strict_json(repo_root / shared_rel / "shared_checkpoint_metadata.json")
        relative = checkpoint.get("path_relative")
        if not isinstance(relative, str) or not _safe_relative(relative):
            raise ValueError("shared checkpoint metadata path is unsafe")
        full = f"{shared_rel}/{relative}"
        expected = omitted_map.get(full)
        if expected is None or checkpoint.get("sha256") != expected or metadata.get("checkpoint_sha256") != expected:
            raise ValueError("omitted weight does not bind shared checkpoint metadata")
        recorded_path = metadata.get("checkpoint_path")
        normalized_recorded = recorded_path.replace("\\", "/") if isinstance(recorded_path, str) else ""
        if not normalized_recorded.endswith("/" + full):
            raise ValueError("shared checkpoint metadata path is inconsistent")


def _derive_required(repo_root: Path, *, weights_may_be_omitted: bool = False) -> tuple[set[str], list[dict[str, str]]]:
    primary_files, primary_omitted = _derive_run_evidence(repo_root, _PRIMARY, weights_may_be_omitted=weights_may_be_omitted)
    repeat_files, repeat_omitted = _derive_run_evidence(repo_root, _REPEAT, weights_may_be_omitted=weights_may_be_omitted)
    freeze_root = repo_root / PurePosixPath(_FREEZE)
    freeze_inventory = _strict_json(freeze_root / "checksum_inventory.json")
    if freeze_inventory.get("schema_version") != "stage2_freeze_inventory_v1" or not isinstance(freeze_inventory.get("files"), dict):
        raise ValueError("freeze inventory is invalid")
    freeze_files = {f"{_FREEZE}/checksum_inventory.json", *(f"{_FREEZE}/{name}" for name in freeze_inventory["files"])}
    required = primary_files | repeat_files | freeze_files | set(_MONITORS.values()) | {_EXPECTATIONS}
    omitted = sorted(primary_omitted + repeat_omitted, key=lambda item: item["path"])
    _validate_omitted_metadata(repo_root, omitted)
    return required, omitted


def _stored_validation(root: Path, *, expected_conditions: list[str], comparison: dict[str, Any] | None) -> None:
    report = _strict_json(root / "stage2_validation.json")
    if report.get("schema_version") != "stage2_validation_v1" or report.get("state") != "pass":
        raise ValueError("stored Stage-2 validation output is not successful")
    if comparison is not None and report.get("repeat_comparison") != comparison:
        raise ValueError("stored primary validation does not bind repeat comparison")
    matrix_conditions = _run_matrix(root)[1].get("42")
    if matrix_conditions != expected_conditions:
        raise ValueError("stored validation condition matrix is inconsistent")
    markdown = (root / "stage2_validation.md").read_text(encoding="utf-8")
    if "pass" not in markdown.casefold():
        raise ValueError("stored Stage-2 validation markdown is not successful")


def _verify_transport_tree(repo_root: Path, inventory: dict[str, Any]) -> tuple[set[str], list[dict[str, str]]]:
    verify_freeze_bundle(repo_root / PurePosixPath(_FREEZE))
    primary = repo_root / PurePosixPath(_PRIMARY)
    repeat = repo_root / PurePosixPath(_REPEAT)
    primary_commands = _strict_json(primary / "commands.json")
    repeat_commands = _strict_json(repeat / "commands.json")
    freeze = repo_root / PurePosixPath(_FREEZE)
    commit = (freeze / "source_commit.txt").read_text(encoding="utf-8").strip()
    if (primary / "commands.json").read_bytes() != (freeze / "commands" / "primary.json").read_bytes() or (repeat / "commands.json").read_bytes() != (freeze / "commands" / "repeat_full_sr.json").read_bytes():
        raise ValueError("transported run commands differ from frozen commands")
    frozen_environment = freeze / "manifests" / "environment_manifest.json"
    for run in (primary, repeat):
        if (run / "manifests" / "environment_manifest.json").read_bytes() != frozen_environment.read_bytes():
            raise ValueError("transported run environment differs from frozen environment")
    environment = _strict_json(frozen_environment)
    _strict_environment(environment)
    _commands(primary, primary / "commands.json", mode="primary", gpu=environment["gpu"])
    _commands(repeat, repeat / "commands.json", mode="repeat_full_sr", gpu=environment["gpu"])
    _manifest_commits(primary, commit, primary_commands, mode="primary")
    _manifest_commits(repeat, commit, repeat_commands, mode="repeat_full_sr")
    frozen_protocol = freeze / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md"
    if any((run / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md").read_bytes() != frozen_protocol.read_bytes() for run in (primary, repeat)):
        raise ValueError("transported run protocol differs from frozen protocol")
    comparison = _strict_json(primary / "stage2_validation.json").get("repeat_comparison")
    if not isinstance(comparison, dict) or comparison.get("schema_version") != "stage2_a100_repeat_comparison_v1" or comparison.get("state") != "pass":
        raise ValueError("stored A100 repeat comparison is invalid")
    _stored_validation(primary, expected_conditions=["standard_lora", "full_sr", "class_prior_reweight"], comparison=comparison)
    _stored_validation(repeat, expected_conditions=["full_sr"], comparison=None)
    validate_monitor_evidence(repo_root / PurePosixPath(_MONITORS["primary"]), expected_command=primary_commands.get("argv"))
    validate_monitor_evidence(repo_root / PurePosixPath(_MONITORS["repeat_full_sr"]), expected_command=repeat_commands.get("argv"))
    required, omitted = _derive_required(repo_root, weights_may_be_omitted=True)
    if omitted != inventory.get("omitted_weights"):
        raise ValueError("evidence omitted_weights do not bind status/checkpoint metadata")
    frozen = _strict_json(repo_root / PurePosixPath(_FREEZE) / "source_expectations.json")
    external = _strict_json(repo_root / PurePosixPath(_EXPECTATIONS))
    if external != frozen:
        raise ValueError("evidence source expectations mismatch")
    return required, omitted


def build_evidence_archive(repo_root: Path, output_path: Path, *, expectations_path: Path | None = None) -> dict[str, Any]:
    """Validate full A100 outputs, then export a no-weight transport archive."""
    repo_root, output_path = Path(repo_root).resolve(), Path(output_path).resolve()
    if output_path.exists():
        raise ValueError(f"refusing to overwrite existing evidence archive: {output_path}")
    if expectations_path is None:
        raise ValueError("evidence export requires source expectations")
    expectations_path = Path(expectations_path).resolve()
    primary, repeat = repo_root / PurePosixPath(_PRIMARY), repo_root / PurePosixPath(_REPEAT)
    primary_report = validate_smoke_root(primary, expected_conditions=("standard_lora", "full_sr", "class_prior_reweight"), canonical_dir=repo_root / "ties_results" / "canonical_v1")
    repeat_report = validate_smoke_root(repeat, expected_conditions=("full_sr",), canonical_dir=repo_root / "ties_results" / "canonical_v1")
    comparison = compare_a100_repeat(primary, repeat, canonical_dir=repo_root / "ties_results" / "canonical_v1")
    if primary_report.get("state") != "pass" or repeat_report.get("state") != "pass" or comparison.get("state") != "pass":
        raise ValueError("evidence export requires successful production validation")
    stored_primary = _strict_json(primary / "stage2_validation.json")
    stored_repeat = _strict_json(repeat / "stage2_validation.json")
    recomputed_primary = dict(primary_report)
    recomputed_primary["repeat_comparison"] = comparison
    if stored_primary != recomputed_primary or stored_repeat != repeat_report:
        raise ValueError("stored Stage-2 validation output does not bind recomputed results")
    verify_freeze_bundle(repo_root / PurePosixPath(_FREEZE))
    frozen_expectations = _strict_json(repo_root / PurePosixPath(_FREEZE) / "source_expectations.json")
    if _strict_json(expectations_path) != frozen_expectations:
        raise ValueError("evidence source expectations mismatch")
    expectation_copy = repo_root / PurePosixPath(_EXPECTATIONS)
    if expectation_copy.exists() and expectation_copy.read_bytes() != expectations_path.read_bytes():
        raise ValueError("existing evidence expectations copy differs")
    expectation_copy.parent.mkdir(parents=True, exist_ok=True)
    expectation_copy.write_bytes(expectations_path.read_bytes())
    required, omitted = _derive_required(repo_root)
    primary_commands = _strict_json(primary / "commands.json")["argv"]
    repeat_commands = _strict_json(repeat / "commands.json")["argv"]
    validate_monitor_evidence(repo_root / PurePosixPath(_MONITORS["primary"]), expected_command=primary_commands)
    validate_monitor_evidence(repo_root / PurePosixPath(_MONITORS["repeat_full_sr"]), expected_command=repeat_commands)
    candidates: dict[str, Path] = {}
    ties_results = repo_root / "ties_results"
    for path in sorted(ties_results.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"evidence tree cannot contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(repo_root).as_posix()
        if relative == INVENTORY_MEMBER:
            continue
        if PurePosixPath(relative).suffix.casefold() in _WEIGHT_SUFFIXES:
            if relative not in {entry["path"] for entry in omitted}:
                raise ValueError(f"untracked model weight in evidence tree: {relative}")
            continue
        if PurePosixPath(relative).suffix.casefold() == ".zip":
            continue
        if relative in required:
            candidates[relative] = path
        elif relative.startswith((_PRIMARY + "/", _REPEAT + "/", _FREEZE + "/")):
            raise ValueError(f"evidence archive has unexpected members: {relative}")
    if set(candidates) != required:
        raise ValueError(f"evidence archive is missing required members: {sorted(required - set(candidates))}")
    inventory = {
        "schema_version": "stage2_evidence_inventory_v2",
        "files": {relative: sha256_file(path) for relative, path in sorted(candidates.items())},
        "omitted_weights": omitted,
    }
    write_json(repo_root / PurePosixPath(INVENTORY_MEMBER), inventory)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, path in sorted(candidates.items()):
            archive.write(path, relative)
        archive.write(repo_root / PurePosixPath(INVENTORY_MEMBER), INVENTORY_MEMBER)
    verified = verify_evidence_archive(output_path)
    return {"schema_version": "stage2_evidence_archive_v2", "state": "pass", "files": len(candidates), "verified": verified}


def verify_evidence_archive(archive_path: Path, *, extract_dir: Path | None = None) -> dict[str, Any]:
    """Verify an evidence ZIP directly; optionally extract only after it passes."""
    archive_path = Path(archive_path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"evidence archive not found: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or any(not _safe_member(name) for name in names) or any(info.is_dir() or _zip_symlink(info) for info in infos):
            raise ValueError("evidence archive contains unsafe, duplicate, directory, or symlink members")
        if INVENTORY_MEMBER not in names:
            raise ValueError("evidence archive is missing its inventory")
        inventory = _strict_json_bytes(archive.read(INVENTORY_MEMBER), label=INVENTORY_MEMBER)
        if set(inventory) != {"schema_version", "files", "omitted_weights"} or inventory.get("schema_version") != "stage2_evidence_inventory_v2" or not isinstance(inventory.get("files"), dict) or not isinstance(inventory.get("omitted_weights"), list):
            raise ValueError("evidence inventory schema is invalid")
        if set(names) != set(inventory["files"]) | {INVENTORY_MEMBER}:
            raise ValueError("evidence archive members do not exactly match inventory files")
        for relative, expected in inventory["files"].items():
            if not _safe_member(relative) or not _is_sha256(expected) or _sha256_bytes(archive.read(relative)) != expected:
                raise ValueError(f"evidence archive checksum mismatch for {relative}")
        for entry in inventory["omitted_weights"]:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "reason"} or not _safe_member(entry.get("path")) or PurePosixPath(entry["path"]).suffix.casefold() not in _WEIGHT_SUFFIXES or entry.get("reason") != "model_weight_excluded" or not _is_sha256(entry.get("sha256")) or entry["path"] in names:
                raise ValueError("evidence omitted_weights entry is invalid")
        payloads = {name: archive.read(name) for name in names}
    with tempfile.TemporaryDirectory(prefix="stage2-evidence-verify-") as temporary:
        root = Path(temporary)
        for relative, payload in payloads.items():
            target = root / PurePosixPath(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        required, omitted = _verify_transport_tree(root, inventory)
        if set(inventory["files"]) != required or omitted != inventory["omitted_weights"]:
            raise ValueError("evidence inventory does not exactly match semantic required files")
    if extract_dir is not None:
        destination = Path(extract_dir).resolve()
        if destination.exists() and not destination.is_dir():
            raise ValueError("evidence extraction destination must be a directory")
        destination.mkdir(parents=True, exist_ok=True)
        targets = [destination / PurePosixPath(relative) for relative in payloads]
        if any(target.exists() or target.is_symlink() for target in targets):
            raise ValueError("evidence extraction refuses to overwrite existing members")
        for target in targets:
            parent = target.parent
            while parent != destination:
                if parent.is_symlink():
                    raise ValueError("evidence extraction refuses symlinked parent directories")
                parent = parent.parent
            if os.path.commonpath((str(destination), str(target.resolve(strict=False)))) != str(destination):
                raise ValueError("evidence extraction target escapes destination")
        for relative, payload in payloads.items():
            target = destination / PurePosixPath(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    return {"schema_version": "stage2_evidence_transport_verify_v1", "state": "pass", "files": len(inventory["files"]), "omitted_weights": len(inventory["omitted_weights"])}
