"""Independent, dependency-light validation for Stage 2 smoke artifacts."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

from canonical.artifacts import sha256_file
from canonical.hans import aggregate_hans_predictions


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
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
                rows.append(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSONL in {path}: {error}") from error
    return rows


def _check_success_status(directory: Path) -> None:
    status_path = directory / "status.json"
    if not status_path.is_file():
        raise ValueError(f"missing status.json: {directory}")
    status = _read_json(status_path)
    if status.get("state") != "success":
        raise ValueError(f"run is not a successful completed artifact: {directory}")
    hashes = status.get("output_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"successful status has no output hashes: {status_path}")
    for relative, expected_hash in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError(f"invalid output hash entry in {status_path}")
        path = (directory / relative).resolve()
        if directory.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"status references a missing or unsafe artifact: {relative}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
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


def _validate_audit_events(events: Sequence[Mapping[str, Any]], *, source: Path, completed_method: bool) -> None:
    marker_seen = False
    evaluation_seen = False
    for event in events:
        if event.get("event") == "final_evaluation_start":
            marker_seen = True
            continue
        is_hans_evaluation = event.get("dataset") == "hans" and event.get("split") == "evaluation"
        if not is_hans_evaluation:
            continue
        # This approved pre-run identity event has no model inference and must
        # remain visible without being treated as official evaluation.
        if event.get("purpose") == "manifest_identity_only":
            continue
        if not marker_seen:
            raise ValueError(f"official HANS evaluation access before final_evaluation_start: {source}")
        evaluation_seen = True
    if completed_method and (not marker_seen or not evaluation_seen):
        raise ValueError(f"completed method lacks final_evaluation_start or official HANS evaluation: {source}")


def _manifest_checkpoint(manifest: Mapping[str, Any], *, path: Path) -> str:
    result = _require_mapping(manifest.get("result"), name=f"result in {path}")
    checkpoint_hash = result.get("final_checkpoint_hash")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        raise ValueError(f"run manifest lacks final_checkpoint_hash: {path}")
    return checkpoint_hash.lower()


def _validate_method(
    run_dir: Path,
    *,
    method: str,
    seed: int,
    common: Mapping[str, str],
) -> None:
    _check_success_status(run_dir)
    manifest = _read_json(run_dir / "run_manifest.json")
    if manifest.get("method_tag") != method or manifest.get("training_seed") != seed:
        raise ValueError(f"run manifest method/seed mismatch: {run_dir}")
    for key, expected in common.items():
        if manifest.get(key) != expected:
            raise ValueError(f"run manifest {key} does not bind the root artifact: {run_dir}")
    checkpoint_hash = _manifest_checkpoint(manifest, path=run_dir / "run_manifest.json")
    rows = _read_jsonl(run_dir / "hans_predictions.jsonl")
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
    _validate_audit_events(_read_jsonl(run_dir / "data_access.jsonl"), source=run_dir, completed_method=True)


def _validate_shared_checkpoint(seed_dir: Path, *, seed: int, common: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, float]]:
    shared_dir = seed_dir / "shared_phase2"
    _check_success_status(shared_dir)
    manifest = _read_json(shared_dir / "run_manifest.json")
    if manifest.get("role") != "shared_phase2" or manifest.get("training_seed") != seed:
        raise ValueError(f"shared Phase-2 run manifest mismatch: {shared_dir}")
    for key, expected in common.items():
        if manifest.get(key) != expected:
            raise ValueError(f"shared Phase-2 {key} does not bind the root artifact: {shared_dir}")
    reference = _read_json(shared_dir / "shared_checkpoint.json")
    relative = reference.get("path_relative")
    digest = reference.get("sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise ValueError(f"invalid shared checkpoint reference: {shared_dir}")
    checkpoint_path = (shared_dir / relative).resolve()
    if shared_dir.resolve() not in checkpoint_path.parents or not checkpoint_path.is_file() or sha256_file(checkpoint_path) != digest:
        raise ValueError(f"shared checkpoint hash mismatch: {shared_dir}")
    metadata = _read_json(shared_dir / "shared_checkpoint_metadata.json")
    if metadata.get("checkpoint_sha256") != digest or Path(str(metadata.get("checkpoint_path"))).resolve() != checkpoint_path:
        raise ValueError(f"shared checkpoint metadata does not match reference: {shared_dir}")
    weights = _finite_positive_weights(metadata)
    return {"path": str(checkpoint_path), "sha256": digest}, weights


def _validate_class_prior_log(run_dir: Path, weights: Mapping[str, float]) -> None:
    log = (run_dir / "stdout.log").read_text(encoding="utf-8")
    if "class-prior reweighting ON" not in log:
        raise ValueError("class-prior branch log does not prove class-prior weighting was enabled")
    for label, weight in weights.items():
        needle = f"{label}: {weight}"
        if needle not in log:
            raise ValueError(f"class-prior branch log does not prove loading weight {label}: {weight}")


def _root_common(root: Path) -> tuple[dict[str, str], dict[str, Any]]:
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
    return (
        {
            "protocol_sha256": protocol_hash,
            "data_manifest_sha256": sha256_file(data),
            "environment_manifest_sha256": sha256_file(environment),
        },
        _read_json(environment),
    )


def validate_smoke_root(
    root: Path,
    *,
    expected_conditions: Sequence[str],
    canonical_dir: Path,
) -> dict[str, Any]:
    """Fail closed unless a Stage-2 smoke root is internally reproducible."""
    root = Path(root).resolve()
    conditions = tuple(expected_conditions)
    if not root.is_dir() or not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("smoke root and unique expected conditions are required")
    common, _environment = _root_common(root)
    commands = _read_json(root / "commands.json")
    if commands.get("expected_condition_tags") != list(conditions):
        raise ValueError("commands.json condition tags do not match validation conditions")
    _validate_audit_events(
        _read_jsonl(root / "manifests" / "data_access.jsonl"),
        source=root / "manifests" / "data_access.jsonl",
        completed_method=False,
    )
    matrix = _read_json(root / "manifests" / "run_matrix.json")
    seeds = matrix.get("training_seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("run_matrix.json must record at least one training seed")
    checks: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("run_matrix.json contains an invalid training seed")
        order = _require_mapping(matrix.get("condition_orders"), name="condition_orders").get(str(seed))
        if not isinstance(order, list) or set(order) != set(conditions) or len(order) != len(conditions):
            raise ValueError(f"run matrix condition order mismatch for seed {seed}")
        seed_dir = root / f"seed_{seed}"
        expected_checkpoint, weights = _validate_shared_checkpoint(seed_dir, seed=seed, common=common)
        for method in conditions:
            run_dir = seed_dir / method
            _validate_method(run_dir, method=method, seed=seed, common=common)
            if method != "standard_lora":
                branch = _read_json(run_dir / "run_manifest.json").get("shared_phase2_checkpoint")
                if branch != expected_checkpoint:
                    raise ValueError(f"dual branch shared checkpoint mismatch: {run_dir}")
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
    common, environment = _root_common(root)
    matrices = _read_json(root / "manifests" / "run_matrix.json")
    seeds = matrices.get("training_seeds")
    if not isinstance(seeds, list) or len(seeds) != 1 or not isinstance(seeds[0], int):
        raise ValueError("repeat comparison requires exactly one integer training seed")
    run = root / f"seed_{seeds[0]}" / "full_sr"
    _check_success_status(run)
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


def compare_a100_repeat(primary_root: Path, repeat_root: Path, tolerance: float = 0.005) -> dict[str, Any]:
    """Compare fresh A100 full_sr smoke runs without using MNLI as the gate."""
    primary_provenance, primary_hans, primary_mnli, _ = _repeat_full_sr(Path(primary_root).resolve())
    repeat_provenance, repeat_hans, repeat_mnli, _ = _repeat_full_sr(Path(repeat_root).resolve())
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
