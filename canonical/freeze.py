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
import zipfile

from canonical.artifacts import sha256_file, write_json
from canonical.source_package import _clean_git_metadata, verify_source_package
from canonical.stage2_validation import compare_a100_repeat


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_FILES = {
    "commands/primary.json",
    "commands/repeat_full_sr.json",
    "manifests/data_manifest.json",
    "manifests/environment_manifest.json",
    "pip_freeze.txt",
    "protocol_snapshot/FROZEN_EXPERIMENT_PROTOCOL.md",
    "protocol_snapshot/protocol_sha256.txt",
    "source_archive_sha256.txt",
    "source_commit.txt",
    "source_metadata.json",
    "execution_provenance.json",
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
    if any(part.casefold() == "canonical_v1" for part in path.resolve().parts):
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


def _strict_environment(environment: dict[str, Any]) -> None:
    required = {"schema_version", "python", "platform", "packages", "cuda_runtime", "cuda_driver", "gpu", "pip_freeze"}
    if not required.issubset(environment) or environment.get("schema_version") != "canonical_environment_manifest_v1":
        raise ValueError("A100 environment manifest schema is incomplete")
    if not isinstance(environment["python"], str) or not environment["python"].startswith("3.12."):
        raise ValueError("A100 environment must record Python 3.12")
    packages = environment["packages"]
    if not isinstance(packages, dict) or not isinstance(packages.get("torch"), str) or not packages["torch"].startswith("2.11.0"):
        raise ValueError("A100 environment must record torch 2.11.0")
    if not all(isinstance(environment[key], str) and environment[key] for key in ("platform", "cuda_runtime", "cuda_driver", "gpu")):
        raise ValueError("A100 environment has incomplete CUDA/platform/GPU fields")
    if "A100" not in environment["gpu"]:
        raise ValueError("A100 environment GPU is invalid")
    _pip_freeze(environment)


def _gpu_probe() -> dict[str, str]:
    import subprocess
    import torch

    nvidia = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=True, text=True, capture_output=True).stdout.strip().splitlines()[0]
    if not torch.cuda.is_available():
        raise ValueError("torch CUDA is unavailable")
    return {"nvidia_smi": nvidia, "torch_gpu": torch.cuda.get_device_name(0), "torch_cuda": str(torch.version.cuda)}


def _strict_data_manifest(data: dict[str, Any]) -> None:
    if data.get("schema_version") != "canonical_data_manifest_v2" or data.get("scope") != "canonical_v1":
        raise ValueError("canonical data manifest schema/scope is invalid")
    groups = (("mnli", ("train", 100000), ("validation_matched", 5000)), ("hans", ("build", None), ("dev", None), ("evaluation", None)), ("ood", ("esnli", None), ("anli", None), ("snli_hard", None), ("wanli", None)))
    for group, *entries in groups:
        mapping = data.get(group)
        if not isinstance(mapping, dict):
            raise ValueError(f"canonical data manifest lacks {group}")
        for name, fixed_count in entries:
            entry = mapping.get(name)
            if not isinstance(entry, dict) or not isinstance(entry.get("full_ids"), list) or not isinstance(entry.get("selected_ids"), list):
                raise ValueError(f"canonical data manifest lacks ID arrays for {group}.{name}")
            if entry.get("selected_ids") != entry.get("full_ids") or entry.get("selected_count") != entry.get("full_count"):
                raise ValueError(f"canonical data manifest contains a smoke cap for {group}.{name}")
            if len(entry["full_ids"]) != entry.get("full_count") or len(entry["selected_ids"]) != entry.get("selected_count") or len(set(entry["full_ids"])) != len(entry["full_ids"]):
                raise ValueError(f"canonical data manifest count/identity array mismatch for {group}.{name}")
            if fixed_count is not None and entry.get("full_count") != fixed_count:
                raise ValueError(f"canonical data manifest count mismatch for {group}.{name}")
            for key in ("full_ids_sha256", "selected_ids_sha256"):
                if not isinstance(entry.get(key), str) or _SHA256_RE.fullmatch(entry[key]) is None:
                    raise ValueError(f"canonical data manifest checksum is invalid for {group}.{name}")


def _commands(root: Path, path: Path, *, mode: str, commit: str, gpu: str) -> dict[str, Any]:
    expected = root / ("commands.json" if root.name != "commands" else ("primary.json" if mode == "primary" else "repeat_full_sr.json"))
    if path.resolve() != expected.resolve():
        raise ValueError("freeze commands must be the exact smoke-root commands.json")
    value = _strict_json(path)
    needed = {"schema_version", "mode", "environment", "argv", "expected_condition_tags", "profile_name", "gpu_name", "started_at"}
    if not needed.issubset(value) or value.get("schema_version") != "stage2_smoke_commands_v1" or value.get("mode") != mode or value.get("environment") != "colab_a100" or value.get("gpu_name") != gpu:
        raise ValueError("A100 commands schema/provenance is invalid")
    return value


def _manifest_commits(root: Path, commit: str) -> None:
    manifests = list((root / "seed_42").glob("*/run_manifest.json"))
    if not manifests or any(_strict_json(path).get("git", {}).get("commit") != commit for path in manifests):
        raise ValueError("smoke run manifests do not bind the source/current commit")


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
    repeat_root: Path | None = None,
    repeat_commands_path: Path | None = None,
    backend_factory: Callable[..., Any] | None = None,
    gpu_probe: Callable[[], dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Freeze only independently validated A100 evidence into a non-canonical output tree."""
    protocol_path = Path(protocol_path).resolve()
    smoke_root = Path(smoke_root).resolve()
    output_dir = Path(output_dir).resolve()
    repo_root = Path(repo_root).resolve()
    source_archive_path = Path(source_archive_path).resolve()
    commands_path = Path(commands_path).resolve()
    repeat_root = Path(repeat_root).resolve() if repeat_root is not None else _repeat_root(smoke_root)
    repeat_commands_path = Path(repeat_commands_path).resolve() if repeat_commands_path is not None else repeat_root / "commands.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"frozen protocol not found: {protocol_path}")
    _reject_canonical_path(output_dir)
    smoke_protocol = smoke_root / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md"
    if not smoke_protocol.is_file() or sha256_file(smoke_protocol) != sha256_file(protocol_path):
        raise ValueError("freeze protocol does not bind the validated A100 smoke snapshot")
    _validated_evidence(smoke_root)
    current = _clean_git_metadata(repo_root)
    commit = current["commit"]
    source_metadata = verify_source_package(source_archive_path, repo_root=repo_root)
    if source_metadata.get("git", {}).get("commit") != commit:
        raise ValueError("source archive commit does not match current clean HEAD")
    if source_metadata.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("source package protocol checksum does not bind the frozen protocol")
    environment = _strict_json(smoke_root / "manifests" / "environment_manifest.json")
    _strict_environment(environment)
    primary_commands = _commands(smoke_root, commands_path, mode="primary", commit=commit, gpu=environment["gpu"])
    repeat_environment = _strict_json(repeat_root / "manifests" / "environment_manifest.json")
    _strict_environment(repeat_environment)
    if repeat_environment != environment:
        raise ValueError("A100 repeat environment differs from primary")
    repeat_commands = _commands(repeat_root, repeat_commands_path, mode="repeat_full_sr", commit=commit, gpu=environment["gpu"])
    _manifest_commits(smoke_root, commit)
    _manifest_commits(repeat_root, commit)
    probe = (gpu_probe or _gpu_probe)()
    if not isinstance(probe, dict) or any("A100" not in str(probe.get(key, "")) for key in ("nvidia_smi", "torch_gpu")) or probe.get("nvidia_smi") != environment["gpu"] or probe.get("torch_gpu") != environment["gpu"] or str(probe.get("torch_cuda")) != environment["cuda_runtime"]:
        raise ValueError("live A100 GPU probe does not match recorded environment")
    # Re-run the artifact validator instead of trusting its report alone.
    comparison = compare_a100_repeat(smoke_root, repeat_root, canonical_dir=repo_root / "ties_results" / "canonical_v1")
    if comparison.get("state") != "pass":
        raise ValueError("freeze requires a successful validated A100 repeat comparison")
    _prepare_output(output_dir)
    try:
        _copy_file(protocol_path, output_dir / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md")
        (output_dir / "protocol_snapshot" / "protocol_sha256.txt").write_text(
            sha256_file(protocol_path) + "\n", encoding="utf-8", newline="\n"
        )
        _copy_file(commands_path, output_dir / "commands" / "primary.json")
        _copy_file(repeat_commands_path, output_dir / "commands" / "repeat_full_sr.json")
        environment_source = smoke_root / "manifests" / "environment_manifest.json"
        _copy_file(environment_source, output_dir / "manifests" / "environment_manifest.json")
        frozen_environment = _strict_json(output_dir / "manifests" / "environment_manifest.json")
        (output_dir / "pip_freeze.txt").write_text(_pip_freeze(frozen_environment), encoding="utf-8", newline="\n")
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
        _strict_data_manifest(data)
        (output_dir / "source_archive_sha256.txt").write_text(
            sha256_file(source_archive_path) + "\n", encoding="utf-8", newline="\n"
        )
        (output_dir / "source_commit.txt").write_text(
            source_metadata["git"]["commit"] + "\n", encoding="utf-8", newline="\n"
        )
        write_json(output_dir / "source_metadata.json", source_metadata)
        write_json(output_dir / "execution_provenance.json", {
            "schema_version": "stage2_freeze_execution_provenance_v1",
            "commit": commit,
            "primary_commands_sha256": sha256_file(output_dir / "commands" / "primary.json"),
            "repeat_commands_sha256": sha256_file(output_dir / "commands" / "repeat_full_sr.json"),
            "environment_sha256": sha256_file(output_dir / "manifests" / "environment_manifest.json"),
        })
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
    for relative in ("commands/primary.json", "commands/repeat_full_sr.json", "manifests/data_manifest.json", "manifests/environment_manifest.json", "source_metadata.json", "execution_provenance.json"):
        _strict_json(output_dir / relative)
    protocol = output_dir / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md"
    protocol_hash = (output_dir / "protocol_snapshot" / "protocol_sha256.txt").read_text(encoding="utf-8").strip()
    if _SHA256_RE.fullmatch(protocol_hash) is None or sha256_file(protocol) != protocol_hash:
        raise ValueError("freeze protocol snapshot checksum mismatch")
    data = _strict_json(output_dir / "manifests" / "data_manifest.json")
    environment = _strict_json(output_dir / "manifests" / "environment_manifest.json")
    metadata = _strict_json(output_dir / "source_metadata.json")
    _strict_data_manifest(data)
    _strict_environment(environment)
    commit = (output_dir / "source_commit.txt").read_text(encoding="utf-8").strip()
    if metadata.get("schema_version") != "stage2_source_package_v1" or not isinstance(metadata.get("git"), dict) or metadata["git"].get("dirty") is not False or metadata["git"].get("commit") != commit:
        raise ValueError("freeze source commit does not bind source metadata")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("freeze source commit is not an exact 40-character hexadecimal value")
    source_archive_sha = (output_dir / "source_archive_sha256.txt").read_text(encoding="utf-8").strip()
    if _SHA256_RE.fullmatch(source_archive_sha) is None:
        raise ValueError("freeze source archive hash is invalid")
    if not isinstance(metadata.get("bundle_sha256"), str) or _SHA256_RE.fullmatch(metadata["bundle_sha256"]) is None:
        raise ValueError("freeze source metadata bundle checksum is invalid")
    if (output_dir / "pip_freeze.txt").read_text(encoding="utf-8") != _pip_freeze(environment):
        raise ValueError("freeze pip_freeze.txt does not bind environment")
    if metadata.get("protocol_sha256") != protocol_hash:
        raise ValueError("freeze source metadata does not bind protocol snapshot")
    _commands(output_dir / "commands", output_dir / "commands" / "primary.json", mode="primary", commit=commit, gpu=environment["gpu"])
    _commands(output_dir / "commands", output_dir / "commands" / "repeat_full_sr.json", mode="repeat_full_sr", commit=commit, gpu=environment["gpu"])
    provenance = _strict_json(output_dir / "execution_provenance.json")
    if provenance.get("schema_version") != "stage2_freeze_execution_provenance_v1" or provenance.get("commit") != commit or provenance.get("primary_commands_sha256") != sha256_file(output_dir / "commands" / "primary.json") or provenance.get("repeat_commands_sha256") != sha256_file(output_dir / "commands" / "repeat_full_sr.json") or provenance.get("environment_sha256") != sha256_file(output_dir / "manifests" / "environment_manifest.json"):
        raise ValueError("freeze execution provenance is inconsistent")
    return {"schema_version": "stage2_freeze_verify_v1", "state": "pass", "inventory_entries": len(files)}


_EVIDENCE_REQUIRED = {
    "ties_results/stage2_smoke/colab_a100_run1/commands.json",
    "ties_results/stage2_smoke/colab_a100_run1/stage2_validation.json",
    "ties_results/stage2_smoke/colab_a100_run1/manifests/environment_manifest.json",
    "ties_results/stage2_smoke/colab_a100_repeat_full_sr/commands.json",
    "ties_results/stage2_smoke/colab_a100_repeat_full_sr/manifests/environment_manifest.json",
    "ties_results/stage2_smoke/freeze_bundle/checksum_inventory.json",
    "ties_results/.stage2_monitor/colab_a100_run1.events.jsonl",
    "ties_results/.stage2_monitor/colab_a100_repeat_full_sr.events.jsonl",
}
_WEIGHT_SUFFIXES = {".pt", ".bin", ".safetensors"}


def _allowed_evidence_path(relative: str) -> bool:
    return relative.startswith("ties_results/stage2_smoke/colab_a100_run1/") or relative.startswith(
        "ties_results/stage2_smoke/colab_a100_repeat_full_sr/"
    ) or relative.startswith("ties_results/stage2_smoke/freeze_bundle/") or relative in {
        "ties_results/.stage2_monitor/colab_a100_run1.events.jsonl",
        "ties_results/.stage2_monitor/colab_a100_repeat_full_sr.events.jsonl",
    }


def build_evidence_archive(repo_root: Path, output_path: Path, *, expectations_path: Path | None = None) -> dict[str, Any]:
    """Export only lightweight Stage-2 runtime evidence with a checked inventory."""
    repo_root = Path(repo_root).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise ValueError(f"refusing to overwrite existing evidence archive: {output_path}")
    candidates: dict[str, Path] = {}
    ties_results = repo_root / "ties_results"
    for path in sorted(ties_results.rglob("*")) if ties_results.exists() else ():
        if path.is_symlink():
            raise ValueError(f"evidence tree cannot contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(repo_root).as_posix()
        if path.suffix.lower() in _WEIGHT_SUFFIXES:
            continue
        if not _safe_relative(relative) or not _allowed_evidence_path(relative):
            continue
        candidates[relative] = path
    if missing := _EVIDENCE_REQUIRED - set(candidates):
        raise ValueError(f"evidence archive is missing required members: {sorted(missing)}")
    inventory = {
        "schema_version": "stage2_evidence_inventory_v1",
        "files": {relative: sha256_file(path) for relative, path in candidates.items()},
    }
    expectations_bytes = None
    if expectations_path is not None:
        expectations_path = Path(expectations_path)
        expectations = _strict_json(expectations_path)
        if set(expectations) != {"archive_sha256", "commit"} or not isinstance(expectations["archive_sha256"], str) or _SHA256_RE.fullmatch(expectations["archive_sha256"]) is None or not isinstance(expectations["commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", expectations["commit"]):
            raise ValueError("source expectations schema is invalid")
        expectations_bytes = expectations_path.read_bytes()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, path in candidates.items():
            archive.write(path, relative)
        if expectations_bytes is not None:
            archive.writestr("stage2_source_expectations.json", expectations_bytes)
        archive.writestr("evidence_checksum_inventory.json", json.dumps(inventory, sort_keys=True, indent=2) + "\n")
    with zipfile.ZipFile(output_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(not _safe_relative(name) for name in names):
            raise ValueError("evidence archive contains unsafe or duplicate members")
        parsed = _strict_json_bytes_for_evidence(archive.read("evidence_checksum_inventory.json"))
        if parsed != inventory:
            raise ValueError("evidence checksum inventory was not written faithfully")
        for relative, expected in inventory["files"].items():
            if relative not in names or sha256_file_from_bytes(archive.read(relative)) != expected:
                raise ValueError(f"evidence archive checksum mismatch for {relative}")
    return {"schema_version": "stage2_evidence_archive_v1", "state": "pass", "files": len(candidates)}


def sha256_file_from_bytes(payload: bytes) -> str:
    from hashlib import sha256

    return sha256(payload).hexdigest()


def _strict_json_bytes_for_evidence(payload: bytes) -> dict[str, Any]:
    path = Path("evidence_checksum_inventory.json")
    with tempfile.TemporaryDirectory(prefix="stage2-evidence-json-") as temporary:
        temporary_path = Path(temporary) / path
        temporary_path.write_bytes(payload)
        return _strict_json(temporary_path)
