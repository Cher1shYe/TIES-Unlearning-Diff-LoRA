"""Frozen, non-invasive subprocess monitoring for Stage 2 jobs."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import stat
import subprocess
import time
from typing import Callable, Iterable, Sequence

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
        record = {
            "event": event,
            "timestamp": now,
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
