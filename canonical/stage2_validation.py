"""Independent, dependency-light validation for Stage 2 smoke artifacts."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from datetime import datetime
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from canonical.artifacts import sha256_file
from canonical.data import validate_hans_manifest_identities
from canonical.hans import aggregate_hans_predictions
from canonical.runner import _METHOD_OUTPUTS, _SHARED_OUTPUTS
from canonical.stage2_contract import STAGE2_SEED


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_non_finite_numbers(value: Any, *, path: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite JSON number at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite_numbers(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite_numbers(item, path=f"{path}[{index}]")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    _reject_non_finite_numbers(value)
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line, parse_constant=_reject_json_constant)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must be a JSON object")
                _reject_non_finite_numbers(value, path=f"$[{line_number}]")
                rows.append(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSONL in {path}: {error}") from error
    return rows


def _check_success_status(
    directory: Path,
    *,
    required_outputs: Sequence[str],
    root: Path | None = None,
    omitted_weights: Mapping[str, str] | None = None,
) -> None:
    status_path = directory / "status.json"
    if not status_path.is_file():
        raise ValueError(f"missing status.json: {directory}")
    status = _read_json(status_path)
    if status.get("state") != "success":
        raise ValueError(f"run is not a successful completed artifact: {directory}")
    hashes = status.get("output_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"successful status has no output hashes: {status_path}")
    expected_paths = set(required_outputs)
    actual_paths = set(hashes)
    if missing := expected_paths - actual_paths:
        raise ValueError(f"status has missing required output hashes: {sorted(missing)}")
    if unexpected := actual_paths - expected_paths:
        raise ValueError(f"status has unexpected output hashes: {sorted(unexpected)}")
    for relative, expected_hash in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError(f"invalid output hash entry in {status_path}")
        path = (directory / relative).resolve()
        if directory.resolve() not in path.parents:
            raise ValueError(f"status references a missing or unsafe artifact: {relative}")
        omission_key = path.relative_to(Path(root).resolve()).as_posix() if root is not None else None
        allowed_omission = omitted_weights is not None and omission_key in omitted_weights
        if allowed_omission and omitted_weights[omission_key] != expected_hash:
            raise ValueError(f"omitted weight hash does not match status: {relative}")
        if not path.is_file() and not allowed_omission:
            raise ValueError(f"status references a missing or unsafe artifact: {relative}")
        if path.is_file() and sha256_file(path) != expected_hash:
            raise ValueError(f"artifact hash mismatch for {path}")


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _finite_positive_weights(metadata: Mapping[str, Any]) -> dict[str, float]:
    weights = _require_mapping(metadata.get("class_prior_weights"), name="class_prior_weights")
    if set(weights) != {"0", "1", "2"}:
        raise ValueError("class_prior_weights must contain exactly labels 0, 1, and 2")
    normalized = {}
    for label, value in weights.items():
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"class_prior_weights[{label!r}] must be finite and positive")
        normalized[label] = float(value)
    return normalized


def _validate_manifest_identity_audit(events: Sequence[Mapping[str, Any]], *, source: Path) -> None:
    if len(events) != 2:
        raise ValueError(f"manifest identity audit must contain its access and summary events: {source}")
    for sequence, event in enumerate(events):
        if event.get("sequence") != sequence or not isinstance(event.get("timestamp"), str) or not event["timestamp"]:
            raise ValueError(f"invalid manifest identity audit sequence or timestamp: {source}")
        if event.get("dataset") != "hans" or event.get("split") != "evaluation" or event.get("purpose") != "manifest_identity_only":
            raise ValueError(f"invalid manifest identity audit event: {source}")
        if sequence == 0 and event.get("event") == "dataset_access":
            if set(event) != {"sequence", "timestamp", "event", "dataset", "split", "purpose"}:
                raise ValueError(f"invalid manifest identity audit access schema: {source}")
        elif sequence == 1 and event.get("event") == "manifest_identity_summary":
            required = {"sequence", "timestamp", "event", "dataset", "split", "purpose", "identity_counts", "identity_checksums"}
            if set(event) != required or not isinstance(event["identity_counts"], Mapping) or not isinstance(event["identity_checksums"], Mapping):
                raise ValueError(f"invalid manifest identity audit summary schema: {source}")
        else:
            raise ValueError(f"invalid manifest identity audit event type: {source}")


def _validate_method_audit(events: Sequence[Mapping[str, Any]], *, source: Path) -> None:
    marker_seen = False
    evaluation_seen = False
    previous_timestamp = None
    for sequence, event in enumerate(events):
        if event.get("sequence") != sequence:
            raise ValueError(f"method audit sequence must be continuous and ordered: {source}")
        timestamp = event.get("timestamp")
        if not isinstance(timestamp, str) or not timestamp:
            raise ValueError(f"method audit timestamp must be non-empty: {source}")
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise ValueError(f"method audit timestamp is invalid: {source}") from error
        if parsed_timestamp.tzinfo is None:
            raise ValueError(f"method audit timestamp must include a timezone: {source}")
        if previous_timestamp is not None and parsed_timestamp < previous_timestamp:
            raise ValueError(f"method audit timestamp order is invalid: {source}")
        previous_timestamp = parsed_timestamp
        if event.get("purpose") == "manifest_identity_only":
            raise ValueError(f"manifest_identity_only is only permitted in the root manifest audit: {source}")
        is_hans_evaluation = event.get("dataset") == "hans" and event.get("split") == "evaluation"
        if is_hans_evaluation:
            expected = {
                "sequence": sequence,
                "timestamp": timestamp,
                "event": "dataset_access",
                "dataset": "hans",
                "split": "evaluation",
                "purpose": "final",
            }
            if event != expected:
                raise ValueError(f"method audit official HANS evaluation schema is invalid: {source}")
            if not marker_seen:
                raise ValueError(f"official HANS evaluation access before final_evaluation_start: {source}")
            evaluation_seen = True
            continue
        is_marker_candidate = (
            event.get("event") == "final_evaluation_start"
            or event.get("purpose") == "boundary"
            or (event.get("dataset") == "hans" and event.get("split") is None)
        )
        if is_marker_candidate:
            expected = {
                "sequence": sequence,
                "timestamp": timestamp,
                "event": "final_evaluation_start",
                "dataset": "hans",
                "split": None,
                "purpose": "boundary",
            }
            if event != expected:
                raise ValueError(f"method audit final_evaluation_start marker schema is invalid: {source}")
            if marker_seen:
                raise ValueError(f"method audit has multiple final_evaluation_start markers: {source}")
            marker_seen = True
            continue
    if not marker_seen or not evaluation_seen:
        raise ValueError(f"completed method lacks final_evaluation_start or official HANS evaluation: {source}")


def _manifest_checkpoint(manifest: Mapping[str, Any], *, path: Path) -> str:
    result = _require_mapping(manifest.get("result"), name=f"result in {path}")
    checkpoint_hash = result.get("final_checkpoint_hash")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        raise ValueError(f"run manifest lacks final_checkpoint_hash: {path}")
    return checkpoint_hash.lower()


def _require_seed(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_manifest_provenance(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    training_seed: int,
    common: Mapping[str, Any],
    path: Path,
) -> None:
    git = _require_mapping(manifest.get("git"), name=f"git in {path}")
    commit = git.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise ValueError(f"git.commit must be a 40-character hexadecimal commit: {path}")
    for field in ("data_seed", "hans_split_seed"):
        manifest_value = _require_seed(manifest.get(field), name=f"{field} in {path}")
        config_value = _require_seed(config.get(field), name=f"{field} in {path} config")
        if manifest_value != common[field] or config_value != manifest_value:
            raise ValueError(f"{field} provenance does not bind root manifest and config: {path}")
    manifest_training_seed = _require_seed(manifest.get("training_seed"), name=f"training_seed in {path}")
    config_training_seed = _require_seed(config.get("training_seed"), name=f"training_seed in {path} config")
    if manifest_training_seed != training_seed or config_training_seed != manifest_training_seed:
        raise ValueError(f"training_seed provenance does not bind run matrix and config: {path}")


def _validate_method(
    run_dir: Path,
    *,
    method: str,
    seed: int,
    common: Mapping[str, str],
    expected_hans_ids: Sequence[str],
) -> None:
    _check_success_status(run_dir, required_outputs=_METHOD_OUTPUTS)
    config = _read_json(run_dir / "config.json")
    manifest = _read_json(run_dir / "run_manifest.json")
    _validate_manifest_provenance(
        manifest, config, training_seed=seed, common=common, path=run_dir / "run_manifest.json"
    )
    if manifest.get("method_tag") != method or manifest.get("training_seed") != seed:
        raise ValueError(f"run manifest method/seed mismatch: {run_dir}")
    for key, expected in common.items():
        if manifest.get(key) != expected:
            raise ValueError(f"run manifest {key} does not bind the root artifact: {run_dir}")
    checkpoint_hash = _manifest_checkpoint(manifest, path=run_dir / "run_manifest.json")
    rows = _read_jsonl(run_dir / "hans_predictions.jsonl")
    prediction_ids = [row.get("pair_id") for row in rows]
    if prediction_ids != list(expected_hans_ids):
        raise ValueError(
            f"ordered HANS prediction IDs do not exactly match data_manifest selected_ids: {run_dir}"
        )
    for index, row in enumerate(rows, start=1):
        if row.get("method_tag") != method:
            raise ValueError(f"HANS prediction method_tag mismatch at {run_dir}:{index}")
        if row.get("training_seed") != seed:
            raise ValueError(f"HANS prediction training_seed mismatch at {run_dir}:{index}")
        if str(row.get("checkpoint_hash", "")).lower() != checkpoint_hash:
            raise ValueError(f"HANS prediction checkpoint_hash mismatch at {run_dir}:{index}")
    recomputed = aggregate_hans_predictions(rows)
    metrics = _read_json(run_dir / "metrics.json")
    final = _require_mapping(metrics.get("final"), name=f"final metrics in {run_dir}")
    reported_hans = final.get("hans")
    if recomputed != reported_hans:
        raise ValueError(f"HANS metrics do not exactly match recomputed predictions: {run_dir}")
    _read_json(run_dir / "selected_layers.json")
    _validate_method_audit(_read_jsonl(run_dir / "data_access.jsonl"), source=run_dir)


def _validate_shared_checkpoint(
    root: Path,
    seed_dir: Path,
    *,
    seed: int,
    common: Mapping[str, str],
    omitted_weights: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    shared_dir = seed_dir / "shared_phase2"
    reference = _read_json(shared_dir / "shared_checkpoint.json")
    relative = reference.get("path_relative")
    if not isinstance(relative, str):
        raise ValueError(f"invalid shared checkpoint reference: {shared_dir}")
    if relative in _SHARED_OUTPUTS or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"shared checkpoint path must be a distinct safe artifact: {shared_dir}")
    _check_success_status(
        shared_dir, required_outputs=(*_SHARED_OUTPUTS, relative),
        root=root, omitted_weights=omitted_weights,
    )
    config = _read_json(shared_dir / "config.json")
    manifest = _read_json(shared_dir / "run_manifest.json")
    if manifest.get("role") != "shared_phase2" or manifest.get("training_seed") != seed:
        raise ValueError(f"shared Phase-2 run manifest mismatch: {shared_dir}")
    for key, expected in common.items():
        if manifest.get(key) != expected:
            raise ValueError(f"shared Phase-2 {key} does not bind the root artifact: {shared_dir}")
    _validate_manifest_provenance(
        manifest, config, training_seed=seed, common=common, path=shared_dir / "run_manifest.json"
    )
    digest = reference.get("sha256")
    if not isinstance(digest, str):
        raise ValueError(f"invalid shared checkpoint reference: {shared_dir}")
    checkpoint_path = (shared_dir / relative).resolve()
    omission_key = checkpoint_path.relative_to(root.resolve()).as_posix()
    omitted_digest = omitted_weights.get(omission_key) if omitted_weights is not None else None
    if shared_dir.resolve() not in checkpoint_path.parents or (
        checkpoint_path.is_file() and sha256_file(checkpoint_path) != digest
    ) or (not checkpoint_path.is_file() and omitted_digest != digest):
        raise ValueError(f"shared checkpoint hash mismatch: {shared_dir}")
    metadata = _read_json(shared_dir / "shared_checkpoint_metadata.json")
    metadata_path = str(metadata.get("checkpoint_path"))
    if omitted_digest is None:
        metadata_path_matches = Path(metadata_path).resolve() == checkpoint_path
        expected_path = str(checkpoint_path)
    else:
        normalized = metadata_path.replace("\\", "/")
        metadata_path_matches = normalized.endswith(f"/{root.name}/{omission_key}")
        expected_path = metadata_path
    if metadata.get("checkpoint_sha256") != digest or not metadata_path_matches:
        raise ValueError(f"shared checkpoint metadata does not match reference: {shared_dir}")
    weights = _finite_positive_weights(metadata)
    return {"path": expected_path, "sha256": digest}, weights


def _validate_class_prior_log(run_dir: Path, weights: Mapping[str, float]) -> None:
    log = (run_dir / "stdout.log").read_text(encoding="utf-8")
    if "class-prior reweighting ON" not in log:
        raise ValueError("class-prior branch log does not prove class-prior weighting was enabled")
    for label, weight in weights.items():
        needle = f"{label}: {weight}"
        if needle not in log:
            raise ValueError(f"class-prior branch log does not prove loading weight {label}: {weight}")


def _root_common(root: Path) -> tuple[dict[str, str], dict[str, Any], list[str]]:
    snapshot = root / "protocol_snapshot"
    protocol = snapshot / "FROZEN_EXPERIMENT_PROTOCOL.md"
    recorded_protocol = snapshot / "protocol_sha256.txt"
    if not protocol.is_file() or not recorded_protocol.is_file():
        raise ValueError("smoke root lacks its frozen protocol snapshot")
    protocol_hash = sha256_file(protocol)
    if recorded_protocol.read_text(encoding="utf-8").strip() != protocol_hash:
        raise ValueError("frozen protocol snapshot checksum mismatch")
    data = root / "manifests" / "data_manifest.json"
    environment = root / "manifests" / "environment_manifest.json"
    if not data.is_file() or not environment.is_file():
        raise ValueError("smoke root lacks data or environment manifest")
    data_manifest = _read_json(data)
    expected_hans_ids = validate_hans_manifest_identities(data_manifest.get("hans"))
    data_seed = _require_seed(data_manifest.get("data_seed"), name="data_manifest.data_seed")
    hans_split_seed = _require_seed(
        data_manifest.get("hans_split_seed"), name="data_manifest.hans_split_seed"
    )
    return (
        {
            "protocol_sha256": protocol_hash,
            "data_manifest_sha256": sha256_file(data),
            "environment_manifest_sha256": sha256_file(environment),
            "data_seed": data_seed,
            "hans_split_seed": hans_split_seed,
        },
        _read_json(environment),
        expected_hans_ids,
    )


def validate_smoke_root(
    root: Path,
    *,
    expected_conditions: Sequence[str],
    canonical_dir: Path,
    omitted_weights: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail closed unless a Stage-2 smoke root is internally reproducible."""
    root = Path(root).resolve()
    conditions = tuple(expected_conditions)
    if not root.is_dir() or not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("smoke root and unique expected conditions are required")
    omissions = dict(omitted_weights or {})
    if omissions:
        expected_omission = f"seed_{STAGE2_SEED}/shared_phase2/checkpoints/shared.pt"
        if set(omissions) != {expected_omission} or not isinstance(omissions[expected_omission], str) or re.fullmatch(r"[0-9a-f]{64}", omissions[expected_omission]) is None:
            raise ValueError("weight-optional validation requires the exact shared checkpoint omission")
    common, _environment, expected_hans_ids = _root_common(root)
    commands = _read_json(root / "commands.json")
    if commands.get("profile_name") != "stage2_smoke_v1":
        raise ValueError("commands.json profile_name is not the frozen Stage 2 smoke profile")
    if commands.get("expected_condition_tags") != list(conditions):
        raise ValueError("commands.json condition tags do not match validation conditions")
    _validate_manifest_identity_audit(
        _read_jsonl(root / "manifests" / "data_access.jsonl"),
        source=root / "manifests" / "data_access.jsonl",
    )
    matrix = _read_json(root / "manifests" / "run_matrix.json")
    seeds = matrix.get("training_seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("run_matrix.json must record at least one training seed")
    if omissions and seeds != [STAGE2_SEED]:
        raise ValueError("weight-optional validation requires only Stage-2 seed 42")
    checks: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("run_matrix.json contains an invalid training seed")
        order = _require_mapping(matrix.get("condition_orders"), name="condition_orders").get(str(seed))
        if not isinstance(order, list) or set(order) != set(conditions) or len(order) != len(conditions):
            raise ValueError(f"run matrix condition order mismatch for seed {seed}")
        seed_dir = root / f"seed_{seed}"
        expected_checkpoint, weights = _validate_shared_checkpoint(
            root, seed_dir, seed=seed, common=common,
            omitted_weights=omissions or None,
        )
        for method in conditions:
            run_dir = seed_dir / method
            _validate_method(
                run_dir,
                method=method,
                seed=seed,
                common=common,
                expected_hans_ids=expected_hans_ids,
            )
            if method != "standard_lora":
                branch = _read_json(run_dir / "run_manifest.json").get("shared_phase2_checkpoint")
                if branch != expected_checkpoint:
                    raise ValueError(f"dual branch shared checkpoint mismatch: {run_dir}")
            elif _read_json(run_dir / "run_manifest.json").get("shared_phase2_checkpoint") is not None:
                raise ValueError(f"standard_lora must not bind the shared checkpoint: {run_dir}")
            if method == "class_prior_reweight":
                _validate_class_prior_log(run_dir, weights)
    canonical_dir = Path(canonical_dir)
    if canonical_dir.exists() and any(canonical_dir.iterdir()):
        raise ValueError(f"formal canonical result directory must be absent or empty: {canonical_dir}")
    checks["artifact_hashes"] = {"state": "pass"}
    checks["hans_recomputation"] = {"state": "pass"}
    checks["audit_order"] = {"state": "pass"}
    checks["shared_checkpoint"] = {"state": "pass"}
    checks["canonical_directory"] = {"state": "pass"}
    return {"schema_version": "stage2_validation_v1", "root": str(root), "state": "pass", "checks": checks}


def validation_report_semantics(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return stable validation semantics, excluding only the extracted root path."""
    if report.get("schema_version") != "stage2_validation_v1":
        raise ValueError("validation report schema is invalid")
    return {key: value for key, value in report.items() if key != "root"}


def render_validation_markdown(report: Mapping[str, Any]) -> str:
    """Render the canonical stored validation Markdown."""
    semantics = validation_report_semantics(report)
    checks = semantics.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("validation report checks are invalid")
    lines = ["# Stage 2 Smoke Validation", "", f"State: `{semantics['state']}`", "", "## Checks", ""]
    lines.extend(f"- {name}: `{entry['state']}`" for name, entry in checks.items())
    if "repeat_comparison" in semantics:
        comparison = semantics["repeat_comparison"]
        lines.extend(["", "## A100 Repeat", "", f"State: `{comparison['state']}`"])
    return "\n".join(lines) + "\n"


def _metric_number(value: float, *, name: str) -> Decimal:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    try:
        return Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"{name} must be a finite number") from error


def compare_metric_values(primary: float, repeat: float, *, tolerance: float) -> dict[str, Any]:
    """Compare a repeated primary metric with an inclusive absolute tolerance."""
    primary_value = _metric_number(primary, name="primary")
    repeat_value = _metric_number(repeat, name="repeat")
    tolerance_value = _metric_number(tolerance, name="tolerance")
    if tolerance_value < 0:
        raise ValueError("tolerance must be non-negative")
    difference = abs(primary_value - repeat_value)
    return {
        "primary": float(primary_value),
        "repeat": float(repeat_value),
        "absolute_difference": float(difference),
        "tolerance": float(tolerance_value),
        "within_tolerance": difference <= tolerance_value,
    }


def _scrub_transient(value: Any) -> Any:
    transient = {"output_dir", "checkpoint_dir", "data_access_log", "started_at", "finished_at", "timestamp", "timestamps"}
    if isinstance(value, Mapping):
        return {key: _scrub_transient(item) for key, item in value.items() if key not in transient}
    if isinstance(value, list):
        return [_scrub_transient(item) for item in value]
    return value


def _repeat_full_sr(root: Path) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    common, environment, _expected_hans_ids = _root_common(root)
    matrices = _read_json(root / "manifests" / "run_matrix.json")
    seeds = matrices.get("training_seeds")
    if not isinstance(seeds, list) or len(seeds) != 1 or not isinstance(seeds[0], int):
        raise ValueError("repeat comparison requires exactly one integer training seed")
    run = root / f"seed_{seeds[0]}" / "full_sr"
    _check_success_status(run, required_outputs=_METHOD_OUTPUTS)
    manifest = _read_json(run / "run_manifest.json")
    for field, expected in common.items():
        if manifest.get(field) != expected:
            raise ValueError(f"repeat run manifest {field} does not bind its root artifact")
    config = _read_json(run / "config.json")
    commands = _read_json(root / "commands.json")
    metrics = _read_json(run / "metrics.json")
    final = _require_mapping(metrics.get("final"), name="repeat final metrics")
    hans = _require_mapping(final.get("hans"), name="repeat HANS metrics")
    mnli = _require_mapping(final.get("mnli"), name="repeat MNLI metrics")
    if "hans_non_entailment" not in hans or "mnli_accuracy" not in mnli:
        raise ValueError("repeat comparison requires HANS non-entailment and MNLI accuracy")
    provenance = {**common, "git_commit": _require_mapping(manifest.get("git"), name="git").get("commit"), "data_seed": manifest.get("data_seed"), "hans_split_seed": manifest.get("hans_split_seed"), "training_seed": manifest.get("training_seed"), "gpu": commands.get("gpu_name"), "environment": _scrub_transient(environment), "config": _scrub_transient(config)}
    if provenance["gpu"] != environment.get("gpu"):
        raise ValueError("repeat gpu provenance does not match its environment manifest")
    return provenance, hans, mnli, commands


def _validate_a100_root_identity(
    root: Path, *, mode: str, conditions: Sequence[str], canonical_dir: Path | None = None,
    omitted_weights: Mapping[str, str] | None = None,
) -> None:
    commands = _read_json(root / "commands.json")
    if commands.get("mode") != mode:
        raise ValueError(f"A100 smoke mode must be {mode}")
    if commands.get("environment") != "colab_a100":
        raise ValueError("A100 smoke environment must be colab_a100")
    if commands.get("expected_condition_tags") != list(conditions):
        raise ValueError("A100 smoke condition tags do not match its required matrix")
    common, environment, _expected_hans_ids = _root_common(root)
    del common
    gpu = commands.get("gpu_name")
    if not isinstance(gpu, str) or "A100" not in gpu or environment.get("gpu") != gpu:
        raise ValueError("A100 gpu evidence is missing or inconsistent")
    canonical_dir = Path("ties_results/canonical_v1") if canonical_dir is None else Path(canonical_dir)
    validate_smoke_root(
        root, expected_conditions=conditions, canonical_dir=canonical_dir,
        omitted_weights=omitted_weights,
    )


def compare_a100_repeat(
    primary_root: Path,
    repeat_root: Path,
    tolerance: float = 0.005,
    *,
    canonical_dir: Path | None = None,
    primary_omitted_weights: Mapping[str, str] | None = None,
    repeat_omitted_weights: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compare fresh A100 full_sr smoke runs without using MNLI as the gate."""
    primary_root = Path(primary_root).resolve()
    repeat_root = Path(repeat_root).resolve()
    _validate_a100_root_identity(primary_root, mode="primary", conditions=("standard_lora", "full_sr", "class_prior_reweight"), canonical_dir=canonical_dir, omitted_weights=primary_omitted_weights)
    _validate_a100_root_identity(repeat_root, mode="repeat_full_sr", conditions=("full_sr",), canonical_dir=canonical_dir, omitted_weights=repeat_omitted_weights)
    primary_provenance, primary_hans, primary_mnli, _ = _repeat_full_sr(primary_root)
    repeat_provenance, repeat_hans, repeat_mnli, _ = _repeat_full_sr(repeat_root)
    for field, primary_value in primary_provenance.items():
        if repeat_provenance.get(field) != primary_value:
            label = "gpu" if field == "gpu" else field
            raise ValueError(f"A100 repeat provenance mismatch for {label}")
    primary_metric = compare_metric_values(primary_hans["hans_non_entailment"], repeat_hans["hans_non_entailment"], tolerance=tolerance)
    return {
        "schema_version": "stage2_a100_repeat_comparison_v1",
        "primary_metric_name": "hans_non_entailment",
        "primary_metric": primary_metric,
        "mnli_diagnostic": {
            "primary": primary_mnli["mnli_accuracy"],
            "repeat": repeat_mnli["mnli_accuracy"],
        },
        "state": "pass" if primary_metric["within_tolerance"] else "fail",
    }
