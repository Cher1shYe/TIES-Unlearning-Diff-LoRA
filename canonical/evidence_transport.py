"""Build and independently verify the no-weight Stage-2 evidence transport."""

from __future__ import annotations

import json
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
from canonical.monitoring import validate_monitor_jsonl
from canonical.stage2_contract import (
    EVIDENCE_INVENTORY_MEMBER, EXPECTATIONS_MEMBER, FREEZE_ROOT, METHOD_OUTPUTS,
    MONITOR_PATHS, OMITTED_WEIGHT_PATHS, PRIMARY_CONDITIONS, PRIMARY_ROOT,
    REPEAT_CONDITIONS, REPEAT_ROOT, SHARED_OUTPUTS, STAGE2_SEED,
)
from canonical.stage2_validation import (
    compare_a100_repeat, render_validation_markdown, validate_smoke_root,
    validation_report_semantics,
)


INVENTORY_MEMBER = EVIDENCE_INVENTORY_MEMBER
_PRIMARY = PRIMARY_ROOT
_REPEAT = REPEAT_ROOT
_FREEZE = FREEZE_ROOT
_EXPECTATIONS = EXPECTATIONS_MEMBER
_MONITORS = MONITOR_PATHS
_WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}


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


def validate_monitor_evidence(
    path: Path, *, expected_command: list[str], expected_cwd: Path | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper around the producer-owned monitor validator."""
    if expected_cwd is None:
        expected_cwd = Path(expected_command[1]).resolve().parent
    return validate_monitor_jsonl(path, expected_command=expected_command, expected_cwd=expected_cwd)


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


def _derive_run_evidence(repo_root: Path, root_rel: str, *, role: str, weights_may_be_omitted: bool = False) -> tuple[set[str], list[dict[str, str]]]:
    root = repo_root / PurePosixPath(root_rel)
    required = _global_run_files(root_rel)
    omitted: list[dict[str, str]] = []
    conditions = PRIMARY_CONDITIONS if role == "primary" else REPEAT_CONDITIONS
    expected_tags = ("shared_phase2", *conditions)
    matrix = _strict_json(root / "manifests" / "run_matrix.json")
    if matrix.get("training_seeds") != [STAGE2_SEED] or set(matrix.get("condition_orders", {})) != {str(STAGE2_SEED)} or set(matrix["condition_orders"][str(STAGE2_SEED)]) != set(conditions) or len(matrix["condition_orders"][str(STAGE2_SEED)]) != len(conditions):
        raise ValueError("evidence run matrix does not match the fixed Stage-2 contract")
    for tag in expected_tags:
        directory = root / f"seed_{STAGE2_SEED}" / tag
        status = _strict_json(directory / "status.json")
        hashes = status.get("output_hashes")
        expected_outputs = (*SHARED_OUTPUTS, "checkpoints/shared.pt") if tag == "shared_phase2" else METHOD_OUTPUTS
        if status.get("schema_version") != "canonical_status_v1" or status.get("state") != "success" or not isinstance(hashes, dict) or set(hashes) != set(expected_outputs):
            raise ValueError(f"evidence status outputs do not match the fixed runner contract: {directory}")
        prefix = f"{root_rel}/seed_{STAGE2_SEED}/{tag}"
        required.add(f"{prefix}/status.json")
        for relative in expected_outputs:
            expected_hash = hashes[relative]
            if not _is_sha256(expected_hash):
                raise ValueError(f"evidence status contains an invalid hash: {relative}")
            full = f"{prefix}/{relative}"
            artifact = directory / PurePosixPath(relative)
            if relative == "checkpoints/shared.pt":
                expected_omitted = OMITTED_WEIGHT_PATHS[role]
                if full != expected_omitted:
                    raise ValueError("shared checkpoint omission path is not fixed")
                if not weights_may_be_omitted and (not artifact.is_file() or artifact.is_symlink() or sha256_file(artifact) != expected_hash):
                    raise ValueError(f"evidence shared checkpoint hash mismatch: {artifact}")
                if weights_may_be_omitted and artifact.exists():
                    raise ValueError("transport unexpectedly contains an omitted checkpoint")
                omitted.append({"path": full, "sha256": expected_hash, "reason": "model_weight_excluded"})
            else:
                if not artifact.is_file() or artifact.is_symlink() or sha256_file(artifact) != expected_hash:
                    raise ValueError(f"evidence status artifact hash mismatch: {artifact}")
                required.add(full)
    return required, omitted


def _validate_omitted_metadata(repo_root: Path, omitted: list[dict[str, str]]) -> None:
    omitted_map = {entry["path"]: entry["sha256"] for entry in omitted}
    if set(omitted_map) != set(OMITTED_WEIGHT_PATHS.values()) or len(omitted) != 2:
        raise ValueError("omitted_weights must be exactly the two fixed shared checkpoints")
    for role, root_rel, conditions in (
        ("primary", _PRIMARY, PRIMARY_CONDITIONS),
        ("repeat_full_sr", _REPEAT, REPEAT_CONDITIONS),
    ):
        shared_rel = f"{root_rel}/seed_{STAGE2_SEED}/shared_phase2"
        checkpoint = _strict_json(repo_root / shared_rel / "shared_checkpoint.json")
        metadata = _strict_json(repo_root / shared_rel / "shared_checkpoint_metadata.json")
        relative = "checkpoints/shared.pt"
        if checkpoint.get("path_relative") != relative:
            raise ValueError("shared checkpoint reference path is not fixed")
        full = f"{shared_rel}/{relative}"
        expected = omitted_map.get(full)
        if expected is None or checkpoint.get("sha256") != expected or metadata.get("checkpoint_sha256") != expected:
            raise ValueError("omitted weight does not bind shared checkpoint metadata")
        recorded_path = metadata.get("checkpoint_path")
        normalized_recorded = recorded_path.replace("\\", "/") if isinstance(recorded_path, str) else ""
        if not normalized_recorded.endswith("/" + full):
            raise ValueError("shared checkpoint metadata path is inconsistent")
        status = _strict_json(repo_root / shared_rel / "status.json")
        if status.get("output_hashes", {}).get(relative) != expected:
            raise ValueError("omitted weight does not bind shared status")
        shared_manifest = _strict_json(repo_root / shared_rel / "run_manifest.json")
        if shared_manifest.get("result", {}).get("checkpoint_sha256") != expected:
            raise ValueError("omitted weight does not bind shared run manifest")
        expected_branch = {"path": recorded_path, "sha256": expected}
        seed_root = repo_root / root_rel / f"seed_{STAGE2_SEED}"
        for method in conditions:
            branch = _strict_json(seed_root / method / "run_manifest.json").get("shared_phase2_checkpoint")
            if (method == "standard_lora" and branch is not None) or (method != "standard_lora" and branch != expected_branch):
                raise ValueError("omitted weight does not bind every branch manifest")


def _derive_required(repo_root: Path, *, weights_may_be_omitted: bool = False) -> tuple[set[str], list[dict[str, str]]]:
    primary_files, primary_omitted = _derive_run_evidence(repo_root, _PRIMARY, role="primary", weights_may_be_omitted=weights_may_be_omitted)
    repeat_files, repeat_omitted = _derive_run_evidence(repo_root, _REPEAT, role="repeat_full_sr", weights_may_be_omitted=weights_may_be_omitted)
    freeze_root = repo_root / PurePosixPath(_FREEZE)
    freeze_inventory = _strict_json(freeze_root / "checksum_inventory.json")
    if freeze_inventory.get("schema_version") != "stage2_freeze_inventory_v1" or not isinstance(freeze_inventory.get("files"), dict):
        raise ValueError("freeze inventory is invalid")
    freeze_files = {f"{_FREEZE}/checksum_inventory.json", *(f"{_FREEZE}/{name}" for name in freeze_inventory["files"])}
    required = primary_files | repeat_files | freeze_files | set(_MONITORS.values()) | {_EXPECTATIONS}
    omitted = sorted(primary_omitted + repeat_omitted, key=lambda item: item["path"])
    _validate_omitted_metadata(repo_root, omitted)
    return required, omitted


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
    required, omitted = _derive_required(repo_root, weights_may_be_omitted=True)
    if omitted != inventory.get("omitted_weights"):
        raise ValueError("evidence omitted_weights do not bind status/checkpoint metadata")
    omitted_map = {entry["path"]: entry["sha256"] for entry in omitted}
    primary_omitted = {path.removeprefix(_PRIMARY + "/"): digest for path, digest in omitted_map.items() if path.startswith(_PRIMARY + "/")}
    repeat_omitted = {path.removeprefix(_REPEAT + "/"): digest for path, digest in omitted_map.items() if path.startswith(_REPEAT + "/")}
    canonical_dir = repo_root / "ties_results" / "canonical_v1"
    primary_report = validate_smoke_root(
        primary, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=canonical_dir,
        omitted_weights=primary_omitted,
    )
    repeat_report = validate_smoke_root(
        repeat, expected_conditions=REPEAT_CONDITIONS, canonical_dir=canonical_dir,
        omitted_weights=repeat_omitted,
    )
    comparison = compare_a100_repeat(
        primary, repeat, canonical_dir=canonical_dir,
        primary_omitted_weights=primary_omitted,
        repeat_omitted_weights=repeat_omitted,
    )
    recomputed_primary = dict(primary_report)
    recomputed_primary["repeat_comparison"] = comparison
    stored_primary = _strict_json(primary / "stage2_validation.json")
    stored_repeat = _strict_json(repeat / "stage2_validation.json")
    if validation_report_semantics(stored_primary) != validation_report_semantics(recomputed_primary) or validation_report_semantics(stored_repeat) != validation_report_semantics(repeat_report):
        raise ValueError("stored validation JSON semantics differ from weight-optional recomputation")
    if (primary / "stage2_validation.md").read_text(encoding="utf-8") != render_validation_markdown(recomputed_primary) or (repeat / "stage2_validation.md").read_text(encoding="utf-8") != render_validation_markdown(repeat_report):
        raise ValueError("stored validation Markdown differs from weight-optional recomputation")
    validate_monitor_evidence(
        repo_root / PurePosixPath(_MONITORS["primary"]),
        expected_command=primary_commands["argv"], expected_cwd=Path(primary_commands["argv"][1]).parent,
    )
    validate_monitor_evidence(
        repo_root / PurePosixPath(_MONITORS["repeat_full_sr"]),
        expected_command=repeat_commands["argv"], expected_cwd=Path(repeat_commands["argv"][1]).parent,
    )
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
    primary_report = validate_smoke_root(primary, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=repo_root / "ties_results" / "canonical_v1")
    repeat_report = validate_smoke_root(repeat, expected_conditions=REPEAT_CONDITIONS, canonical_dir=repo_root / "ties_results" / "canonical_v1")
    comparison = compare_a100_repeat(primary, repeat, canonical_dir=repo_root / "ties_results" / "canonical_v1")
    if primary_report.get("state") != "pass" or repeat_report.get("state") != "pass" or comparison.get("state") != "pass":
        raise ValueError("evidence export requires successful production validation")
    stored_primary = _strict_json(primary / "stage2_validation.json")
    stored_repeat = _strict_json(repeat / "stage2_validation.json")
    recomputed_primary = dict(primary_report)
    recomputed_primary["repeat_comparison"] = comparison
    if stored_primary != recomputed_primary or stored_repeat != repeat_report or (primary / "stage2_validation.md").read_text(encoding="utf-8") != render_validation_markdown(recomputed_primary) or (repeat / "stage2_validation.md").read_text(encoding="utf-8") != render_validation_markdown(repeat_report):
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
    validate_monitor_evidence(repo_root / PurePosixPath(_MONITORS["primary"]), expected_command=primary_commands, expected_cwd=Path(primary_commands[1]).parent)
    validate_monitor_evidence(repo_root / PurePosixPath(_MONITORS["repeat_full_sr"]), expected_command=repeat_commands, expected_cwd=Path(repeat_commands[1]).parent)
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
