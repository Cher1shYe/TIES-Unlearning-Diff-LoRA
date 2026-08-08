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
from datetime import datetime
from hashlib import sha256

from canonical.artifacts import collect_environment_metadata, sha256_file, write_json
from canonical.data import HANS_OFFICIAL_ANCHORS_V1, validate_hans_manifest_identities
from canonical.source_package import _EXCLUSIONS_METADATA, _allowed_source, _clean_git_metadata, verify_source_package
from canonical.stage2_validation import compare_a100_repeat


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_KEYS = {
    "source", "split", "id_strategy", "preferred_id_fields", "strata_fields",
    "selection_seed", "selected_limit", "full_count", "selected_count",
    "full_ids", "selected_ids", "full_ids_sha256", "selected_ids_sha256",
}
_DATA_PROVENANCE = {
    ("mnli", "train"): ("nyu-mll/glue:mnli", "train", ["idx", "row_id", "id", "uid"], [], 100000),
    ("mnli", "validation_matched"): ("nyu-mll/glue:mnli", "validation_matched", ["idx", "row_id", "id", "uid"], [], 5000),
    ("hans", "build"): ("tommccoy1/hans", "build", ["pairID"], [], None),
    ("hans", "dev"): ("tommccoy1/hans", "dev", ["pairID"], [], None),
    ("hans", "evaluation"): ("tommccoy1/hans", "evaluation", ["pairID"], ["gold_label", "heuristic", "subcase"], None),
    ("ood", "esnli"): ("e-SNLI", "test", ["pairID", "uid", "id", "idx"], [], None),
    ("ood", "anli"): ("facebook/anli", "test", ["pairID", "uid", "id", "idx"], [], None),
    ("ood", "snli_hard"): ("snli_1.0_test_hard", "test", ["pairID", "uid", "id", "idx"], [], None),
    ("ood", "wanli"): ("alisawuffles/WANLI", "test", ["pairID", "uid", "id", "idx"], [], None),
}
_REQUIRED_FILES = {
    "commands/primary.json",
    "commands/repeat_full_sr.json",
    "manifests/data_manifest.json",
    "manifests/environment_manifest.json",
    "pip_freeze.txt",
    "protocol_snapshot/FROZEN_EXPERIMENT_PROTOCOL.md",
    "protocol_snapshot/protocol_sha256.txt",
    "source_archive_sha256.txt",
    "source_origin_commit.txt",
    "source_commit.txt",
    "source_expectations.json",
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


def _validated_evidence(smoke_root: Path) -> dict[str, Any]:
    report = _strict_json(smoke_root / "stage2_validation.json")
    if report.get("state") != "pass" or not isinstance(report.get("repeat_comparison"), dict) or report["repeat_comparison"].get("state") != "pass":
        raise ValueError("freeze requires validated successful A100 primary and repeat evidence")
    environment = _strict_json(smoke_root / "manifests" / "environment_manifest.json")
    if not isinstance(environment.get("gpu"), str) or "A100" not in environment["gpu"]:
        raise ValueError("freeze requires a recorded A100 environment")
    return report


def _strict_environment(environment: dict[str, Any]) -> None:
    required = {"schema_version", "python", "python_implementation", "platform", "packages", "cuda_runtime", "cuda_driver", "gpu", "torch_gpu", "nvidia_smi_gpu", "pip_freeze"}
    if set(environment) != required or environment.get("schema_version") != "canonical_environment_manifest_v1":
        raise ValueError("A100 environment manifest schema is incomplete")
    if not isinstance(environment["python"], str) or not environment["python"].startswith("3.12."):
        raise ValueError("A100 environment must record Python 3.12")
    if environment.get("python_implementation") != "CPython":
        raise ValueError("A100 environment must record CPython")
    packages = environment["packages"]
    required_packages = ("torch", "transformers", "datasets", "numpy")
    if not isinstance(packages, dict) or set(packages) != set(required_packages) or any(not isinstance(packages.get(name), str) or not packages[name] for name in required_packages) or not packages["torch"].startswith("2.11.0"):
        raise ValueError("A100 environment must record torch 2.11.0 and all runner packages")
    if not all(isinstance(environment[key], str) and environment[key] for key in ("platform", "cuda_runtime", "cuda_driver", "gpu", "torch_gpu", "nvidia_smi_gpu")):
        raise ValueError("A100 environment has incomplete CUDA/platform/GPU fields")
    if any("A100" not in environment[key] for key in ("gpu", "torch_gpu", "nvidia_smi_gpu")) or not (environment["gpu"] == environment["torch_gpu"] == environment["nvidia_smi_gpu"]):
        raise ValueError("A100 environment GPU is invalid")
    freeze = _pip_freeze(environment).splitlines()
    parsed = [(line.split("==", 1)[0].casefold().replace("_", "-"), line.split("==", 1)[1]) for line in freeze if line.count("==") == 1]
    pinned = dict(parsed)
    for name in required_packages:
        normalized = name.replace("_", "-")
        if sum(package_name == normalized for package_name, _ in parsed) != 1 or pinned.get(normalized) != packages[name]:
            raise ValueError(f"pip freeze does not bind environment package {name}")


def _live_environment_probe() -> dict[str, Any]:
    return {"schema_version": "canonical_environment_manifest_v1", **collect_environment_metadata()}


def _strict_data_manifest(data: dict[str, Any]) -> None:
    if set(data) != {"schema_version", "scope", "data_seed", "hans_split_seed", "mnli", "hans", "ood"} or data.get("schema_version") != "canonical_data_manifest_v4" or data.get("scope") != "canonical_v1":
        raise ValueError("canonical data manifest schema/scope is invalid")
    if data.get("data_seed") != 42 or data.get("hans_split_seed") != 42:
        raise ValueError("canonical data manifest seeds must equal 42")
    groups = (("mnli", "train", "validation_matched"), ("hans", "build", "dev", "evaluation"), ("ood", "esnli", "anli", "snli_hard", "wanli"))
    for group, *entries in groups:
        mapping = data.get(group)
        expected_names = set(entries) | ({"split_integrity", "content_integrity", "selection_integrity"} if group == "hans" else set())
        if not isinstance(mapping, dict) or set(mapping) != expected_names:
            raise ValueError(f"canonical data manifest lacks {group}")
        for name in entries:
            entry = mapping.get(name)
            if not isinstance(entry, dict) or set(entry) != _IDENTITY_KEYS or not isinstance(entry.get("full_ids"), list) or not isinstance(entry.get("selected_ids"), list):
                raise ValueError(f"canonical data manifest lacks ID arrays for {group}.{name}")
            full_ids, selected_ids = entry["full_ids"], entry["selected_ids"]
            source, split, preferred, strata, fixed_count = _DATA_PROVENANCE[(group, name)]
            if selected_ids != full_ids or entry.get("selected_count") != entry.get("full_count"):
                raise ValueError(f"canonical data manifest contains a smoke cap for {group}.{name}")
            if not full_ids or len(full_ids) != entry.get("full_count") or len(selected_ids) != entry.get("selected_count") or len(set(full_ids)) != len(full_ids) or len(set(selected_ids)) != len(selected_ids):
                raise ValueError(f"canonical data manifest count/identity array mismatch for {group}.{name}")
            if not all(isinstance(identity, str) and identity for identity in full_ids):
                raise ValueError(f"canonical data manifest ID identity values are invalid for {group}.{name}")
            if entry.get("selection_seed") != 42 or entry.get("selected_limit") != fixed_count or entry.get("id_strategy") != "preferred_field_or_content_sha256" or entry.get("source") != source or entry.get("split") != split or entry.get("preferred_id_fields") != preferred or entry.get("strata_fields") != strata:
                raise ValueError(f"canonical data manifest provenance is invalid for {group}.{name}")
            if fixed_count is not None and (entry.get("full_count") != fixed_count or entry.get("selected_count") != fixed_count):
                raise ValueError(f"canonical data manifest count mismatch for {group}.{name}")
            for key in ("full_ids_sha256", "selected_ids_sha256"):
                if not isinstance(entry.get(key), str) or _SHA256_RE.fullmatch(entry[key]) is None:
                    raise ValueError(f"canonical data manifest checksum is invalid for {group}.{name}")
            full_checksum = sha256(json.dumps(entry["full_ids"], ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            selected_checksum = sha256(json.dumps(entry["selected_ids"], ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            if entry["full_ids_sha256"] != full_checksum or entry["selected_ids_sha256"] != selected_checksum:
                raise ValueError(f"canonical data manifest checksum mismatch for {group}.{name}")
    hans_sets = [set(data["hans"][name]["full_ids"]) for name in ("build", "dev", "evaluation")]
    if any(hans_sets[left] & hans_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("canonical HANS build/dev/evaluation IDs must be disjoint")
    validate_hans_manifest_identities(
        data["hans"],
        expected_seed=42,
        expected_selection_cap=None,
        official_anchors=HANS_OFFICIAL_ANCHORS_V1,
    )


def _commands(root: Path, path: Path, *, mode: str, gpu: str) -> dict[str, Any]:
    expected = root / ("commands.json" if root.name != "commands" else ("primary.json" if mode == "primary" else "repeat_full_sr.json"))
    if path.resolve() != expected.resolve():
        raise ValueError("freeze commands must be the exact smoke-root commands.json")
    value = _strict_json(path)
    needed = {"schema_version", "mode", "environment", "argv", "expected_condition_tags", "profile_name", "gpu_name", "started_at"}
    expected_tags = ["standard_lora", "full_sr", "class_prior_reweight"] if mode == "primary" else ["full_sr"]
    argv = value.get("argv")
    if set(value) != needed or value.get("schema_version") != "stage2_smoke_commands_v1" or value.get("mode") != mode or value.get("environment") != "colab_a100" or value.get("gpu_name") != gpu or value.get("profile_name") != "stage2_smoke_v1" or value.get("expected_condition_tags") != expected_tags or not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
        raise ValueError("A100 commands schema/provenance is invalid")
    try:
        timestamp = datetime.fromisoformat(value["started_at"])
    except (TypeError, ValueError) as error:
        raise ValueError("A100 commands timestamp is invalid") from error
    if timestamp.tzinfo is None:
        raise ValueError("A100 commands timestamp must include timezone")
    expected_layout = ["--mode", mode, "--environment", "colab_a100", "--protocol", None, "--output-dir", None, "--fresh"]
    if len(argv) != 11 or not Path(argv[0]).name.casefold().startswith("python") or Path(argv[1]).name != "run_stage2_smoke.py" or argv[2] != expected_layout[0] or argv[3] != expected_layout[1] or argv[4] != expected_layout[2] or argv[5] != expected_layout[3] or argv[6] != expected_layout[4] or argv[8] != expected_layout[6] or argv[10] != expected_layout[8]:
        raise ValueError("A100 commands argv does not bind mode/environment/output/fresh")
    try:
        from run_stage2_smoke import build_parser
        parsed = build_parser().parse_args(argv[2:])
    except SystemExit as error:
        raise ValueError("A100 commands argv is rejected by the production parser") from error
    if parsed.mode != mode or parsed.environment != "colab_a100" or not parsed.fresh:
        raise ValueError("A100 commands argv parser result is invalid")
    def option(name: str) -> str:
        if argv.count(name) != 1:
            raise ValueError(f"A100 commands argv must contain exactly one {name}")
        index = argv.index(name)
        if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            raise ValueError(f"A100 commands argv has no value for {name}")
        return argv[index + 1]
    if option("--mode") != mode or option("--environment") != "colab_a100":
        raise ValueError("A100 commands argv mode/environment is invalid")
    protocol_parts = PurePosixPath(option("--protocol").replace("\\", "/")).parts
    if len(protocol_parts) < 3 or tuple(part.casefold() for part in protocol_parts[-3:]) != ("docs", "paper_rebuild", "frozen_experiment_protocol.md"):
        raise ValueError("A100 commands argv protocol is invalid")
    expected_leaf = "colab_a100_run1" if mode == "primary" else "colab_a100_repeat_full_sr"
    output_parts = PurePosixPath(option("--output-dir").replace("\\", "/")).parts
    if len(output_parts) < 3 or tuple(part.casefold() for part in output_parts[-3:]) != ("ties_results", "stage2_smoke", expected_leaf):
        raise ValueError("A100 commands argv output root is invalid")
    return value


def _manifest_commits(root: Path, commit: str, commands: dict[str, Any], *, mode: str) -> None:
    tags = {"shared_phase2", "standard_lora", "full_sr", "class_prior_reweight"} if mode == "primary" else {"shared_phase2", "full_sr"}
    manifests = {path.parent.name: path for path in (root / "seed_42").glob("*/run_manifest.json")}
    if set(manifests) != tags:
        raise ValueError("smoke run manifests do not exactly match the required condition matrix")
    for path in manifests.values():
        manifest = _strict_json(path)
        git = manifest.get("git")
        if not isinstance(git, dict) or set(git) != {"commit", "branch", "dirty", "status_porcelain"} or git.get("commit") != commit or git.get("branch") is not None or git.get("dirty") is not False or git.get("status_porcelain") != []:
            raise ValueError("smoke run manifests do not bind a detached clean execution commit")
        if manifest.get("command") != commands["argv"]:
            raise ValueError("smoke run manifest command does not equal commands.json argv")


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
    expectations_path: Path | None = None,
    commands_path: Path,
    repeat_root: Path | None = None,
    repeat_commands_path: Path | None = None,
    backend_factory: Callable[..., Any] | None = None,
    environment_probe: Callable[[], dict[str, Any]] | None = None,
    gpu_probe: Callable[[], dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Freeze only independently validated A100 evidence into a non-canonical output tree."""
    protocol_path = Path(protocol_path).resolve()
    smoke_root = Path(smoke_root).resolve()
    output_dir = Path(output_dir).resolve()
    repo_root = Path(repo_root).resolve()
    source_archive_path = Path(source_archive_path).resolve()
    expectations_path = Path(expectations_path).resolve() if expectations_path is not None else None
    commands_path = Path(commands_path).resolve()
    repeat_root = Path(repeat_root).resolve() if repeat_root is not None else _repeat_root(smoke_root)
    repeat_commands_path = Path(repeat_commands_path).resolve() if repeat_commands_path is not None else repeat_root / "commands.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"frozen protocol not found: {protocol_path}")
    _reject_canonical_path(output_dir)
    smoke_protocol = smoke_root / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md"
    if not smoke_protocol.is_file() or sha256_file(smoke_protocol) != sha256_file(protocol_path):
        raise ValueError("freeze protocol does not bind the validated A100 smoke snapshot")
    validation_report = _validated_evidence(smoke_root)
    current = _clean_git_metadata(repo_root)
    commit = current["origin_commit"]
    source_metadata = verify_source_package(source_archive_path, repo_root=repo_root)
    if source_metadata.get("git", {}).get("execution_commit") != commit:
        raise ValueError("source archive commit does not match current clean HEAD")
    if source_metadata.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("source package protocol checksum does not bind the frozen protocol")
    if expectations_path is None:
        raise ValueError("freeze creation requires external source expectations")
    expectations = _strict_json(expectations_path)
    expected_keys = {"schema_version", "archive_sha256", "origin_commit", "execution_commit", "source_manifest_sha256"}
    if set(expectations) != expected_keys or expectations.get("schema_version") != "stage2_source_expectations_v1" or expectations.get("archive_sha256") != sha256_file(source_archive_path) or expectations.get("origin_commit") != source_metadata["git"]["origin_commit"] or expectations.get("execution_commit") != commit or expectations.get("source_manifest_sha256") != source_metadata.get("source_manifest_sha256"):
        raise ValueError("source expectations do not bind the outer archive and source metadata")
    environment = _strict_json(smoke_root / "manifests" / "environment_manifest.json")
    _strict_environment(environment)
    primary_commands = _commands(smoke_root, commands_path, mode="primary", gpu=environment["gpu"])
    repeat_environment = _strict_json(repeat_root / "manifests" / "environment_manifest.json")
    _strict_environment(repeat_environment)
    if repeat_environment != environment or (repeat_root / "manifests" / "environment_manifest.json").read_bytes() != (smoke_root / "manifests" / "environment_manifest.json").read_bytes():
        raise ValueError("A100 repeat environment differs from primary")
    repeat_commands = _commands(repeat_root, repeat_commands_path, mode="repeat_full_sr", gpu=environment["gpu"])
    _manifest_commits(smoke_root, commit, primary_commands, mode="primary")
    _manifest_commits(repeat_root, commit, repeat_commands, mode="repeat_full_sr")
    if gpu_probe is not None:
        raise ValueError("gpu_probe is obsolete; freeze creation requires a complete environment_probe")
    live_environment = (environment_probe or _live_environment_probe)()
    try:
        _strict_environment(live_environment)
    except ValueError as error:
        raise ValueError("live environment is not a complete A100 environment") from error
    if live_environment != environment:
        raise ValueError("live environment does not exactly match primary/repeat recorded environment")
    # Re-run the artifact validator instead of trusting its report alone.
    comparison = compare_a100_repeat(smoke_root, repeat_root, canonical_dir=repo_root / "ties_results" / "canonical_v1")
    if comparison.get("state") != "pass":
        raise ValueError("freeze requires a successful validated A100 repeat comparison")
    if validation_report.get("repeat_comparison") != comparison:
        raise ValueError("stored A100 validation report does not bind the recomputed repeat comparison")
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
            source_metadata["git"]["execution_commit"] + "\n", encoding="utf-8", newline="\n"
        )
        (output_dir / "source_origin_commit.txt").write_text(source_metadata["git"]["origin_commit"] + "\n", encoding="utf-8", newline="\n")
        _copy_file(expectations_path, output_dir / "source_expectations.json")
        write_json(output_dir / "source_metadata.json", source_metadata)
        write_json(output_dir / "execution_provenance.json", {
            "schema_version": "stage2_freeze_execution_provenance_v1",
            "commit": commit,
            "origin_commit": source_metadata["git"]["origin_commit"],
            "source_manifest_sha256": source_metadata["source_manifest_sha256"],
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
    if set(files) != _REQUIRED_FILES:
        missing = sorted(_REQUIRED_FILES - set(files))
        extra = sorted(set(files) - _REQUIRED_FILES)
        raise ValueError(f"freeze inventory is not the exact required file set; missing={missing}; extra={extra}")
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
    for relative in ("commands/primary.json", "commands/repeat_full_sr.json", "manifests/data_manifest.json", "manifests/environment_manifest.json", "source_metadata.json", "source_expectations.json", "execution_provenance.json"):
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
    metadata_keys = {"schema_version", "git", "protocol_path", "protocol_sha256", "bundle_sha256", "source_manifest", "source_manifest_sha256", "exclusions"}
    git_keys = {"origin_commit", "execution_commit", "branch", "dirty", "status_porcelain"}
    if set(metadata) != metadata_keys or metadata.get("schema_version") != "stage2_source_package_v2" or not isinstance(metadata.get("git"), dict) or set(metadata["git"]) != git_keys or metadata["git"].get("dirty") is not False or metadata["git"].get("status_porcelain") != [] or metadata["git"].get("execution_commit") != commit or metadata.get("exclusions") != _EXCLUSIONS_METADATA:
        raise ValueError("freeze source commit does not bind source metadata")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("freeze source commit is not an exact 40-character hexadecimal value")
    source_archive_sha = (output_dir / "source_archive_sha256.txt").read_text(encoding="utf-8").strip()
    if _SHA256_RE.fullmatch(source_archive_sha) is None:
        raise ValueError("freeze source archive hash is invalid")
    if not isinstance(metadata.get("bundle_sha256"), str) or _SHA256_RE.fullmatch(metadata["bundle_sha256"]) is None:
        raise ValueError("freeze source metadata bundle checksum is invalid")
    source_manifest = metadata.get("source_manifest")
    if not isinstance(source_manifest, list) or sha256(json.dumps(source_manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest() != metadata.get("source_manifest_sha256"):
        raise ValueError("freeze source manifest checksum is invalid")
    for entry in source_manifest:
        if not isinstance(entry, dict) or set(entry) != {"path", "mode", "blob_sha1", "content_sha256"} or not isinstance(entry.get("path"), str) or not _allowed_source(entry["path"]) or entry.get("mode") not in {"100644", "100755"} or re.fullmatch(r"[0-9a-f]{40}", str(entry.get("blob_sha1", ""))) is None or _SHA256_RE.fullmatch(str(entry.get("content_sha256", ""))) is None:
            raise ValueError("freeze source manifest contains unsafe or invalid entries")
    if source_manifest != sorted(source_manifest, key=lambda entry: entry["path"].encode("utf-8")) or len({entry["path"] for entry in source_manifest}) != len(source_manifest):
        raise ValueError("freeze source manifest is not uniquely and canonically ordered")
    if not isinstance(metadata.get("protocol_path"), str) or not any(entry["path"] == metadata["protocol_path"] and entry["content_sha256"] == metadata.get("protocol_sha256") for entry in source_manifest):
        raise ValueError("freeze source manifest does not bind protocol provenance")
    origin_commit = (output_dir / "source_origin_commit.txt").read_text(encoding="utf-8").strip()
    expectations = _strict_json(output_dir / "source_expectations.json")
    expectation_keys = {"schema_version", "archive_sha256", "origin_commit", "execution_commit", "source_manifest_sha256"}
    if set(expectations) != expectation_keys or expectations.get("schema_version") != "stage2_source_expectations_v1" or re.fullmatch(r"[0-9a-f]{40}", origin_commit) is None or expectations.get("archive_sha256") != source_archive_sha or expectations.get("origin_commit") != origin_commit or expectations.get("execution_commit") != commit or expectations.get("source_manifest_sha256") != metadata.get("source_manifest_sha256") or metadata["git"].get("origin_commit") != origin_commit:
        raise ValueError("freeze source expectations/provenance is inconsistent")
    if (output_dir / "pip_freeze.txt").read_text(encoding="utf-8") != _pip_freeze(environment):
        raise ValueError("freeze pip_freeze.txt does not bind environment")
    if metadata.get("protocol_sha256") != protocol_hash:
        raise ValueError("freeze source metadata does not bind protocol snapshot")
    _commands(output_dir / "commands", output_dir / "commands" / "primary.json", mode="primary", gpu=environment["gpu"])
    _commands(output_dir / "commands", output_dir / "commands" / "repeat_full_sr.json", mode="repeat_full_sr", gpu=environment["gpu"])
    provenance = _strict_json(output_dir / "execution_provenance.json")
    if provenance.get("schema_version") != "stage2_freeze_execution_provenance_v1" or provenance.get("commit") != commit or provenance.get("origin_commit") != origin_commit or provenance.get("source_manifest_sha256") != metadata.get("source_manifest_sha256") or provenance.get("primary_commands_sha256") != sha256_file(output_dir / "commands" / "primary.json") or provenance.get("repeat_commands_sha256") != sha256_file(output_dir / "commands" / "repeat_full_sr.json") or provenance.get("environment_sha256") != sha256_file(output_dir / "manifests" / "environment_manifest.json"):
        raise ValueError("freeze execution provenance is inconsistent")
    return {"schema_version": "stage2_freeze_verify_v1", "state": "pass", "inventory_entries": len(files)}


def build_evidence_archive(repo_root: Path, output_path: Path, *, expectations_path: Path | None = None) -> dict[str, Any]:
    """Compatibility wrapper for the independently verified v2 transport."""
    from canonical.evidence_transport import build_evidence_archive as build_v2

    return build_v2(repo_root, output_path, expectations_path=expectations_path)
