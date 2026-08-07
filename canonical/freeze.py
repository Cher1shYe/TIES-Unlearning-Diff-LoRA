"""Build and independently verify the Stage-2 canonical environment freeze bundle."""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Callable

from canonical.artifacts import sha256_file, write_json
from canonical.source_package import verify_source_package
from canonical.stage2_validation import compare_a100_repeat


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_FILES = {
    "commands.json",
    "manifests/data_manifest.json",
    "manifests/environment_manifest.json",
    "pip_freeze.txt",
    "protocol_snapshot/FROZEN_EXPERIMENT_PROTOCOL.md",
    "protocol_snapshot/protocol_sha256.txt",
    "source_archive_sha256.txt",
    "source_commit.txt",
    "source_metadata.json",
}


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden in {path}: {value}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")

    def finite(item: Any, location: str = "$") -> None:
        if isinstance(item, Real) and not isinstance(item, bool) and not math.isfinite(float(item)):
            raise ValueError(f"non-finite JSON number in {path} at {location}")
        if isinstance(item, dict):
            for key, nested in item.items():
                finite(nested, f"{location}.{key}")
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                finite(nested, f"{location}[{index}]")

    finite(value)
    return value


def _safe_relative(relative: str) -> bool:
    path = PurePosixPath(relative)
    return bool(relative) and not path.is_absolute() and ".." not in path.parts and "\\" not in relative


def _reject_canonical_path(path: Path) -> None:
    if "canonical_v1" in path.resolve().parts:
        raise ValueError("freeze output must not use canonical_v1")


def _prepare_output(output_dir: Path) -> None:
    _reject_canonical_path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("--fresh requires a new or empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)


def _validated_evidence(smoke_root: Path) -> None:
    report = _strict_json(smoke_root / "stage2_validation.json")
    if report.get("state") != "pass" or not isinstance(report.get("repeat_comparison"), dict) or report["repeat_comparison"].get("state") != "pass":
        raise ValueError("freeze requires validated successful A100 primary and repeat evidence")
    environment = _strict_json(smoke_root / "manifests" / "environment_manifest.json")
    if not isinstance(environment.get("gpu"), str) or "A100" not in environment["gpu"]:
        raise ValueError("freeze requires a recorded A100 environment")


def _repeat_root(smoke_root: Path) -> Path:
    return smoke_root.parent / "colab_a100_repeat_full_sr"


def _copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise ValueError(f"required freeze input is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _pip_freeze(environment: dict[str, Any]) -> str:
    entries = environment.get("pip_freeze")
    if not isinstance(entries, list) or not entries or not all(isinstance(item, str) and item for item in entries):
        raise ValueError("A100 environment manifest must include a complete pip_freeze list")
    return "\n".join(entries) + "\n"


def _write_inventory(output_dir: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"freeze bundle cannot contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir).as_posix()
        if relative == "checksum_inventory.json":
            continue
        files[relative] = sha256_file(path)
    inventory = {"schema_version": "stage2_freeze_inventory_v1", "files": files}
    write_json(output_dir / "checksum_inventory.json", inventory)
    return inventory


def build_freeze_bundle(
    protocol_path: Path,
    smoke_root: Path,
    output_dir: Path,
    repo_root: Path,
    *,
    source_archive_path: Path,
    commands_path: Path,
    backend_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Freeze only independently validated A100 evidence into a non-canonical output tree."""
    protocol_path = Path(protocol_path).resolve()
    smoke_root = Path(smoke_root).resolve()
    output_dir = Path(output_dir).resolve()
    repo_root = Path(repo_root).resolve()
    source_archive_path = Path(source_archive_path).resolve()
    commands_path = Path(commands_path).resolve()
    if not protocol_path.is_file():
        raise FileNotFoundError(f"frozen protocol not found: {protocol_path}")
    _reject_canonical_path(output_dir)
    smoke_protocol = smoke_root / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md"
    if not smoke_protocol.is_file() or sha256_file(smoke_protocol) != sha256_file(protocol_path):
        raise ValueError("freeze protocol does not bind the validated A100 smoke snapshot")
    if commands_path != (smoke_root / "commands.json").resolve():
        raise ValueError("freeze commands must be the validated A100 primary commands.json")
    _validated_evidence(smoke_root)
    # Re-run the artifact validator instead of trusting its report alone.
    comparison = compare_a100_repeat(smoke_root, _repeat_root(smoke_root))
    if comparison.get("state") != "pass":
        raise ValueError("freeze requires a successful validated A100 repeat comparison")
    source_metadata = verify_source_package(source_archive_path, repo_root=repo_root)
    if source_metadata.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("source package protocol checksum does not bind the frozen protocol")
    _prepare_output(output_dir)
    try:
        _copy_file(protocol_path, output_dir / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md")
        (output_dir / "protocol_snapshot" / "protocol_sha256.txt").write_text(
            sha256_file(protocol_path) + "\n", encoding="utf-8", newline="\n"
        )
        _copy_file(commands_path, output_dir / "commands.json")
        _strict_json(output_dir / "commands.json")
        environment_source = smoke_root / "manifests" / "environment_manifest.json"
        _copy_file(environment_source, output_dir / "manifests" / "environment_manifest.json")
        environment = _strict_json(output_dir / "manifests" / "environment_manifest.json")
        (output_dir / "pip_freeze.txt").write_text(_pip_freeze(environment), encoding="utf-8", newline="\n")
        factory = backend_factory
        uses_default_backend = factory is None
        if uses_default_backend:
            from canonical.backend import RealCanonicalBackend

            factory = RealCanonicalBackend
        from configs.config import TrainConfig

        with tempfile.TemporaryDirectory(prefix="stage2-canonical-manifest-") as temporary:
            config = TrainConfig(output_dir=str(Path(temporary) / "unused"))
            if uses_default_backend:
                backend = factory(config)
            else:
                try:
                    backend = factory(config)
                except TypeError as error:
                    try:
                        backend = factory()
                    except TypeError:
                        raise error
            generated = Path(temporary) / "manifest"
            backend.initialize_manifests(generated, protocol_path)
            data_manifest = generated / "manifests" / "data_manifest.json"
            _copy_file(data_manifest, output_dir / "manifests" / "data_manifest.json")
        data = _strict_json(output_dir / "manifests" / "data_manifest.json")
        if data.get("scope") != "canonical_v1":
            raise ValueError("canonical freeze data manifest must have scope canonical_v1")
        (output_dir / "source_archive_sha256.txt").write_text(
            sha256_file(source_archive_path) + "\n", encoding="utf-8", newline="\n"
        )
        (output_dir / "source_commit.txt").write_text(
            source_metadata["git"]["commit"] + "\n", encoding="utf-8", newline="\n"
        )
        write_json(output_dir / "source_metadata.json", source_metadata)
        inventory = _write_inventory(output_dir)
    except BaseException:
        # No recursive cleanup: an incomplete output is valuable forensic evidence.
        raise
    return {"schema_version": "stage2_freeze_bundle_v1", "target_schema": "canonical_v1", "state": "pass", "output_dir": str(output_dir), "inventory_entries": len(inventory["files"])}


def verify_freeze_bundle(output_dir: Path) -> dict[str, Any]:
    """Verify a freeze bundle without datasets, torch, or network access."""
    output_dir = Path(output_dir).resolve()
    _reject_canonical_path(output_dir)
    inventory_path = output_dir / "checksum_inventory.json"
    inventory = _strict_json(inventory_path)
    if inventory.get("schema_version") != "stage2_freeze_inventory_v1" or not isinstance(inventory.get("files"), dict):
        raise ValueError("checksum inventory schema is invalid")
    files = inventory["files"]
    if "checksum_inventory.json" in files:
        raise ValueError("checksum inventory must not checksum itself")
    if not _REQUIRED_FILES.issubset(files):
        raise ValueError(f"checksum inventory is missing required freeze files: {sorted(_REQUIRED_FILES - set(files))}")
    actual: set[str] = set()
    for path in output_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"unsafe symlink in freeze bundle: {path}")
        if path.is_file() and path != inventory_path:
            relative = path.relative_to(output_dir).as_posix()
            if not _safe_relative(relative):
                raise ValueError(f"unsafe freeze bundle path: {relative}")
            actual.add(relative)
    declared = set(files)
    if missing := declared - actual:
        raise ValueError(f"freeze inventory has missing files: {sorted(missing)}")
    if extra := actual - declared:
        raise ValueError(f"freeze bundle has extra files: {sorted(extra)}")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not _safe_relative(relative):
            raise ValueError(f"unsafe checksum inventory path: {relative!r}")
        if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
            raise ValueError(f"invalid checksum inventory hash for {relative}")
        if sha256_file(output_dir / relative) != expected:
            raise ValueError(f"freeze inventory hash mismatch for {relative}")
    for relative in ("commands.json", "manifests/data_manifest.json", "manifests/environment_manifest.json", "source_metadata.json"):
        _strict_json(output_dir / relative)
    protocol = output_dir / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md"
    protocol_hash = (output_dir / "protocol_snapshot" / "protocol_sha256.txt").read_text(encoding="utf-8").strip()
    if _SHA256_RE.fullmatch(protocol_hash) is None or sha256_file(protocol) != protocol_hash:
        raise ValueError("freeze protocol snapshot checksum mismatch")
    data = _strict_json(output_dir / "manifests" / "data_manifest.json")
    environment = _strict_json(output_dir / "manifests" / "environment_manifest.json")
    metadata = _strict_json(output_dir / "source_metadata.json")
    if data.get("scope") != "canonical_v1":
        raise ValueError("freeze data manifest scope is not canonical_v1")
    if not isinstance(environment.get("gpu"), str) or "A100" not in environment["gpu"]:
        raise ValueError("freeze environment is not an A100")
    commit = (output_dir / "source_commit.txt").read_text(encoding="utf-8").strip()
    if not isinstance(metadata.get("git"), dict) or metadata["git"].get("commit") != commit:
        raise ValueError("freeze source commit does not bind source metadata")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("freeze source commit is not an exact 40-character hexadecimal value")
    source_archive_sha = (output_dir / "source_archive_sha256.txt").read_text(encoding="utf-8").strip()
    if _SHA256_RE.fullmatch(source_archive_sha) is None:
        raise ValueError("freeze source archive hash is invalid")
    if not (output_dir / "pip_freeze.txt").read_text(encoding="utf-8").strip():
        raise ValueError("freeze pip_freeze.txt is empty")
    return {"schema_version": "stage2_freeze_verify_v1", "state": "pass", "inventory_entries": len(files)}
