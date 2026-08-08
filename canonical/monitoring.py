"""Frozen, non-invasive subprocess monitoring for Stage 2 jobs."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from canonical.artifacts import json_ready


@dataclass(frozen=True)
class MonitorPolicy:
    """Wall-clock thresholds for a monitored subprocess."""

    check_interval_seconds: float
    stall_seconds: float
    hard_timeout_seconds: float

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a finite positive number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be a finite positive number")


PRODUCTION_POLICY = MonitorPolicy(300, 3600, 43200)

_COMMON_EVENT_KEYS = {"event", "timestamp", "elapsed_seconds", "command", "cwd"}
_EVENT_KEYS = {
    "STARTED": ({*_COMMON_EVENT_KEYS, "policy", "fingerprints"},),
    "STATUS_CHECK": ({*_COMMON_EVENT_KEYS, "returncode", "fingerprints", "watch_errors"},),
    "PROGRESS": ({*_COMMON_EVENT_KEYS, "fingerprints"},),
    "STALL_WARNING": ({*_COMMON_EVENT_KEYS, "stall_seconds", "fingerprints"},),
    "FATAL_PATTERN": ({*_COMMON_EVENT_KEYS, "pattern", "path"},),
    "COMPLETED": ({*_COMMON_EVENT_KEYS, "returncode"},),
    "CRASHED": (
        {*_COMMON_EVENT_KEYS, "returncode"},
        {*_COMMON_EVENT_KEYS, "returncode", "failure_stage", "error_type"},
    ),
    "HARD_TIMEOUT": ({*_COMMON_EVENT_KEYS, "timeout_seconds"},),
}
_FAILED_SUCCESS_EVENTS = {"FATAL_PATTERN", "CRASHED", "HARD_TIMEOUT"}


def _reject_nonfinite(value: Any, *, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite monitor value at {label}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, label=f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, label=f"{label}[{index}]")


def _normalized_command(command: Sequence[str]) -> list[str]:
    if not isinstance(command, (list, tuple)) or not command or not all(isinstance(item, str) and item for item in command):
        raise ValueError("monitor command must be a non-empty string argv")
    normalized = list(command)
    for index in range(min(2, len(normalized))):
        normalized[index] = os.path.normcase(os.path.normpath(normalized[index]))
    return normalized


def _validate_fingerprints(value: Any, *, allow_empty: bool) -> None:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError("monitor fingerprints must be a non-empty list")
    base = {"path", "exists", "size", "mtime_ns"}
    for fingerprint in value:
        if not isinstance(fingerprint, dict) or set(fingerprint) not in (base, {*base, "error"}):
            raise ValueError("monitor fingerprint schema is invalid")
        if not isinstance(fingerprint["path"], str) or not fingerprint["path"]:
            raise ValueError("monitor fingerprint path is invalid")
        if fingerprint["exists"] not in (True, False, None):
            raise ValueError("monitor fingerprint exists value is invalid")
        for key in ("size", "mtime_ns"):
            item = fingerprint[key]
            if item is not None and (isinstance(item, bool) or not isinstance(item, int) or item < 0):
                raise ValueError(f"monitor fingerprint {key} is invalid")
        if "error" in fingerprint and (not isinstance(fingerprint["error"], str) or not fingerprint["error"]):
            raise ValueError("monitor fingerprint error is invalid")


def _validate_event_specific(record: Mapping[str, Any]) -> None:
    event = record["event"]
    if event == "STARTED":
        policy = record["policy"]
        if not isinstance(policy, dict) or policy != asdict(PRODUCTION_POLICY):
            raise ValueError("monitor STARTED policy is invalid")
        _validate_fingerprints(record["fingerprints"], allow_empty=False)
    elif event == "STATUS_CHECK":
        returncode = record["returncode"]
        if returncode is not None and (isinstance(returncode, bool) or not isinstance(returncode, int)):
            raise ValueError("monitor STATUS_CHECK returncode is invalid")
        _validate_fingerprints(record["fingerprints"], allow_empty=False)
        errors = record["watch_errors"]
        if not isinstance(errors, list) or any(
            not isinstance(item, dict) or set(item) != {"path", "error"}
            or not all(isinstance(item[key], str) and item[key] for key in ("path", "error"))
            for item in errors
        ):
            raise ValueError("monitor STATUS_CHECK watch_errors are invalid")
    elif event == "PROGRESS":
        _validate_fingerprints(record["fingerprints"], allow_empty=False)
    elif event == "STALL_WARNING":
        if record["stall_seconds"] != PRODUCTION_POLICY.stall_seconds:
            raise ValueError("monitor STALL_WARNING threshold is invalid")
        _validate_fingerprints(record["fingerprints"], allow_empty=False)
    elif event == "FATAL_PATTERN":
        if record["pattern"] not in FATAL_PATTERNS or not isinstance(record["path"], str) or not record["path"]:
            raise ValueError("monitor FATAL_PATTERN fields are invalid")
    elif event in {"COMPLETED", "CRASHED"}:
        if isinstance(record["returncode"], bool) or not isinstance(record["returncode"], int):
            raise ValueError(f"monitor {event} returncode is invalid")
        if "failure_stage" in record and (
            not isinstance(record["failure_stage"], str) or not record["failure_stage"]
            or not isinstance(record["error_type"], str) or not record["error_type"]
        ):
            raise ValueError("monitor CRASHED launch fields are invalid")
    elif event == "HARD_TIMEOUT" and record["timeout_seconds"] != PRODUCTION_POLICY.hard_timeout_seconds:
        raise ValueError("monitor HARD_TIMEOUT threshold is invalid")


def validate_monitor_jsonl(
    path: str | Path,
    *,
    expected_command: Sequence[str],
    expected_cwd: str | Path,
) -> dict[str, Any]:
    """Validate successful monitor evidence against the producer's exact schema."""
    records: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"monitor evidence is unreadable: {path}") from error
    if not lines or any(not line for line in lines):
        raise ValueError("monitor evidence is empty or contains blank records")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite monitor JSON constant: {value}")

    for line_number, line in enumerate(lines, 1):
        try:
            record = json.loads(line, parse_constant=reject_constant)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed monitor JSONL at line {line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"monitor line {line_number} must be an object")
        _reject_nonfinite(record, label=f"line[{line_number}]")
        event = record.get("event")
        schemas = _EVENT_KEYS.get(event)
        if schemas is None or set(record) not in schemas:
            raise ValueError(f"monitor event schema is invalid for {event!r}")
        timestamp = record.get("timestamp")
        if not isinstance(timestamp, str):
            raise ValueError("monitor timestamp must be a timezone-aware ISO string")
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise ValueError("monitor timestamp is invalid") from error
        if parsed_timestamp.tzinfo is None:
            raise ValueError("monitor timestamp must include a timezone")
        elapsed = record.get("elapsed_seconds")
        if isinstance(elapsed, bool) or not isinstance(elapsed, Real) or elapsed < 0:
            raise ValueError("monitor elapsed_seconds is invalid")
        if _normalized_command(record.get("command")) != _normalized_command(expected_command):
            raise ValueError("monitor command does not bind the expected child argv")
        cwd = record.get("cwd")
        if not isinstance(cwd, str) or Path(cwd).resolve() != Path(expected_cwd).resolve():
            raise ValueError("monitor cwd does not bind the expected repository")
        _validate_event_specific(record)
        records.append(record)
    if records[0]["event"] != "STARTED" or records[-1]["event"] != "COMPLETED":
        raise ValueError("monitor evidence must start with STARTED and end with COMPLETED")
    if not any(record["event"] == "STATUS_CHECK" for record in records):
        raise ValueError("monitor evidence requires at least one STATUS_CHECK")
    if any(record["event"] in _FAILED_SUCCESS_EVENTS for record in records):
        raise ValueError("successful monitor evidence contains a failure event")
    if records[-1].get("returncode") != 0:
        raise ValueError("monitor COMPLETED returncode must equal zero")
    previous_time: datetime | None = None
    previous_elapsed = -1.0
    for record in records:
        parsed = datetime.fromisoformat(record["timestamp"])
        if previous_time is not None and parsed < previous_time:
            raise ValueError("monitor timestamps are out of order")
        if float(record["elapsed_seconds"]) < previous_elapsed:
            raise ValueError("monitor elapsed_seconds are out of order")
        previous_time = parsed
        previous_elapsed = float(record["elapsed_seconds"])
    return {"schema_version": "stage2_monitor_evidence_v1", "state": "pass", "events": len(records)}


FATAL_PATTERNS = {
    "CUDA_OOM": re.compile(
        r"\bcuda\s+(?:error\s*:\s*)?out of memory\b|cudnn_status_alloc_failed|outofmemoryerror"
    ),
    "NONFINITE_LOSS": re.compile(
        r"\b(?:non[- ]?finite\s+loss|loss\s*(?:is|=|:)?\s*(?:nan|inf(?:inity)?|non[- ]?finite))\b"
    ),
    "DOWNLOAD_FAILURE": re.compile(
        r"\b(?:download|network|connection)(?:\s+\w+){0,3}\s+(?:failed|failure|error)\b|"
        r"\b(?:failed|failure|error)\s+(?:to\s+)?(?:download|connect|fetch)\b|"
        r"\bconnectionerror\s*:\s*(?:could not|unable to)?\s*connect\b|\bcould not connect\b"
    ),
    "CHECKPOINT_HASH_MISMATCH": re.compile(
        r"\b(?:checkpoint(?:[_\s-]?hash)?|sha[- ]?256)(?:\s+\w+){0,4}\s+(?:mismatch|invalid)\b|"
        r"\b(?:mismatch|invalid)\b.*\b(?:checkpoint|sha[- ]?256)\b"
    ),
    "PREDICTION_ROW_MISMATCH": re.compile(
        r"\b(?:prediction(?:[_\s-]?(?:row|count|hash))?|hans\s+metrics)(?:\s+\w+){0,6}\s+"
        r"(?:mismatch|invalid)\b|\bhans\s+metrics\s+do\s+not(?:\s+\w+){0,3}\s+match\s+"
        r"recomputed\s+predictions\b|\b(?:metrics|prediction).*(?:recomputed|recompute).*\bmismatch\b"
    ),
}


def _resolved(path: Path) -> Path:
    """Resolve a path without allowing a watch failure to stop monitoring."""
    try:
        return path.resolve(strict=False)
    except OSError:
        return path.absolute()


def _is_excluded(path: Path, excluded_path: Path | None) -> bool:
    return excluded_path is not None and _resolved(path) == excluded_path


def _error_fingerprint(path: Path, error: OSError) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": None,
        "size": None,
        "mtime_ns": None,
        "error": type(error).__name__,
    }


def _fingerprint(path: Path, *, excluded_path: Path | None = None) -> dict[str, object]:
    """Return a JSON-ready progress fingerprint for one watched path.

    For a directory, aggregate its regular files so writes below a run directory
    count as progress even when the directory metadata itself is unchanged.
    """
    path = Path(path)
    try:
        status = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False, "size": None, "mtime_ns": None}
    except (PermissionError, OSError) as error:
        return _error_fingerprint(path, error)

    if not stat.S_ISDIR(status.st_mode):
        return {
            "path": str(path),
            "exists": True,
            "size": status.st_size,
            "mtime_ns": status.st_mtime_ns,
        }

    total_size = 0
    newest_mtime_ns = status.st_mtime_ns
    try:
        for child in path.rglob("*"):
            if _is_excluded(child, excluded_path):
                continue
            try:
                child_status = child.stat()
            except (FileNotFoundError, PermissionError, OSError):
                continue
            newest_mtime_ns = max(newest_mtime_ns, child_status.st_mtime_ns)
            if stat.S_ISREG(child_status.st_mode):
                total_size += child_status.st_size
    except (PermissionError, OSError) as error:
        return _error_fingerprint(path, error)
    return {
        "path": str(path),
        "exists": True,
        "size": total_size,
        "mtime_ns": newest_mtime_ns,
    }


def _fingerprints(paths: Iterable[Path], *, excluded_path: Path | None = None) -> list[dict[str, object]]:
    return [_fingerprint(path, excluded_path=excluded_path) for path in paths]


def _watched_text_files(paths: Iterable[Path], *, excluded_path: Path | None = None) -> Iterable[Path]:
    for path in paths:
        try:
            if _is_excluded(path, excluded_path):
                continue
            if path.is_file():
                yield path
            elif path.is_dir():
                for child in path.rglob("*"):
                    if not _is_excluded(child, excluded_path) and child.is_file():
                        yield child
        except (FileNotFoundError, PermissionError, OSError):
            continue


def _fatal_matches(
    paths: Iterable[Path], *, excluded_path: Path | None = None
) -> list[tuple[str, Path]]:
    """Find fatal signatures in the last MiB of watched files."""
    matches: list[tuple[str, Path]] = []
    for path in _watched_text_files(paths, excluded_path=excluded_path):
        try:
            with path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - 1024 * 1024))
                text = handle.read().decode("utf-8", errors="replace").lower()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        for pattern_name, pattern in FATAL_PATTERNS.items():
            if pattern.search(text):
                matches.append((pattern_name, path))
    return matches


def _preflight_events(event_file: Path) -> None:
    """Create and verify the event destination before starting a child process."""
    event_file.parent.mkdir(parents=True, exist_ok=True)
    with event_file.open("a", encoding="utf-8", newline="\n") as handle:
        handle.flush()


def monitor_command(
    command: Sequence[str],
    *,
    cwd: str | Path,
    events_path: str | Path,
    watched_paths: Iterable[str | Path],
    policy: MonitorPolicy = PRODUCTION_POLICY,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
    popen_factory: Callable[..., object] = subprocess.Popen,
) -> int:
    """Run *command* while recording progress, advisory, and timeout events.

    The command is always an argv sequence (never a shell string). Fatal log
    signatures and stalls are recorded only; the sole automatic intervention is
    termination after the hard timeout.
    """
    if not command:
        raise ValueError("command must contain at least one argv element")
    if any(not isinstance(argument, str) for argument in command):
        raise TypeError("command argv elements must be strings")
    if not isinstance(policy, MonitorPolicy):
        raise TypeError("policy must be a MonitorPolicy")

    argv = list(command)
    cwd_path = Path(cwd)
    try:
        cwd_path = cwd_path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"cwd must be an existing directory: {cwd}") from error
    if not cwd_path.is_dir():
        raise ValueError(f"cwd must be an existing directory: {cwd}")

    working_directory = str(cwd_path)
    event_file = Path(events_path)
    _preflight_events(event_file)
    resolved_event_file = _resolved(event_file)
    watched = [Path(path) for path in watched_paths]
    started_at = clock()
    if not math.isfinite(started_at):
        raise ValueError("clock must return a finite number")
    initial_fingerprints = _fingerprints(watched, excluded_path=resolved_event_file)
    seen_fatal_patterns: set[str] = set()

    def emit(event: str, now: float, **details: object) -> None:
        elapsed_seconds = now - started_at
        emitted_at = wall_clock()
        if not isinstance(emitted_at, datetime) or emitted_at.tzinfo is None:
            raise ValueError("wall_clock must return a timezone-aware datetime")
        record = {
            "event": event,
            "timestamp": emitted_at.isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "command": argv,
            "cwd": working_directory,
            **details,
        }
        normalized = json_ready(record)
        line = json.dumps(normalized, ensure_ascii=False, allow_nan=False, sort_keys=True)
        event_file.parent.mkdir(parents=True, exist_ok=True)
        with event_file.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()

    try:
        process = popen_factory(argv, cwd=working_directory, shell=False)
    except OSError as error:
        emit(
            "CRASHED",
            started_at,
            returncode=127,
            failure_stage="popen",
            error_type=type(error).__name__,
        )
        return 127
    emit(
        "STARTED",
        started_at,
        policy=asdict(policy),
        fingerprints=initial_fingerprints,
    )
    previous_fingerprints = initial_fingerprints
    last_progress_at = started_at
    stall_warned = False

    while True:
        now = clock()
        if not math.isfinite(now):
            raise ValueError("clock must return a finite number")
        return_code = process.poll()
        current_fingerprints = _fingerprints(watched, excluded_path=resolved_event_file)
        if current_fingerprints != previous_fingerprints:
            if not any("error" in fingerprint for fingerprint in current_fingerprints):
                last_progress_at = now
                stall_warned = False
                emit("PROGRESS", now, fingerprints=current_fingerprints)
            previous_fingerprints = current_fingerprints

        watch_errors = [
            {"path": str(fingerprint["path"]), "error": str(fingerprint["error"])}
            for fingerprint in current_fingerprints
            if "error" in fingerprint
        ]

        for pattern_name, pattern_path in _fatal_matches(watched, excluded_path=resolved_event_file):
            if pattern_name not in seen_fatal_patterns:
                seen_fatal_patterns.add(pattern_name)
                emit("FATAL_PATTERN", now, pattern=pattern_name, path=str(pattern_path))

        emit(
            "STATUS_CHECK",
            now,
            returncode=return_code,
            fingerprints=current_fingerprints,
            watch_errors=watch_errors,
        )
        if return_code is not None:
            emit("COMPLETED" if return_code == 0 else "CRASHED", now, returncode=return_code)
            return int(return_code)

        elapsed_seconds = now - started_at
        if elapsed_seconds >= policy.hard_timeout_seconds:
            emit("HARD_TIMEOUT", now, timeout_seconds=policy.hard_timeout_seconds)
            process.terminate()
            terminate_deadline = clock() + 10
            while process.poll() is None:
                grace_now = clock()
                if grace_now >= terminate_deadline:
                    break
                sleep(min(1.0, terminate_deadline - grace_now))
                if clock() <= grace_now:
                    break
            if process.poll() is None:
                process.kill()
            return 124

        if not stall_warned and now - last_progress_at >= policy.stall_seconds:
            emit(
                "STALL_WARNING",
                now,
                stall_seconds=policy.stall_seconds,
                fingerprints=current_fingerprints,
            )
            stall_warned = True

        sleep(policy.check_interval_seconds)
