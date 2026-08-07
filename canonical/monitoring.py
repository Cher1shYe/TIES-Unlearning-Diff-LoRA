"""Frozen, non-invasive subprocess monitoring for Stage 2 jobs."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
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
    "CUDA_OOM": ("cuda out of memory", "cudnn_status_alloc_failed", "outofmemoryerror"),
    "NONFINITE_LOSS": ("non-finite loss", "nonfinite loss", "loss is nan", "loss=nan"),
    "DOWNLOAD_FAILURE": ("download failed", "failed to download", "download failure"),
    "CHECKPOINT_HASH_MISMATCH": ("checkpoint hash mismatch", "checkpoint checksum mismatch"),
    "PREDICTION_ROW_MISMATCH": ("prediction-row mismatch", "prediction row mismatch"),
}


def _fingerprint(path: Path) -> dict[str, object]:
    """Return a JSON-ready progress fingerprint for one watched path.

    For a directory, aggregate its regular files so writes below a run directory
    count as progress even when the directory metadata itself is unchanged.
    """
    path = Path(path)
    try:
        status = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False, "size": None, "mtime_ns": None}

    if not path.is_dir():
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
            try:
                child_status = child.stat()
            except (FileNotFoundError, PermissionError, OSError):
                continue
            newest_mtime_ns = max(newest_mtime_ns, child_status.st_mtime_ns)
            if child.is_file():
                total_size += child_status.st_size
    except (PermissionError, OSError):
        pass
    return {
        "path": str(path),
        "exists": True,
        "size": total_size,
        "mtime_ns": newest_mtime_ns,
    }


def _fingerprints(paths: Iterable[Path]) -> list[dict[str, object]]:
    return [_fingerprint(path) for path in paths]


def _watched_text_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        try:
            if path.is_file():
                yield path
            elif path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file():
                        yield child
        except (FileNotFoundError, PermissionError, OSError):
            continue


def _fatal_matches(paths: Iterable[Path]) -> list[tuple[str, Path]]:
    """Find fatal signatures in the last MiB of watched files."""
    matches: list[tuple[str, Path]] = []
    for path in _watched_text_files(paths):
        try:
            with path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - 1024 * 1024))
                text = handle.read().decode("utf-8", errors="replace").lower()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        for pattern_name, phrases in FATAL_PATTERNS.items():
            if any(phrase in text for phrase in phrases):
                matches.append((pattern_name, path))
    return matches


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
    working_directory = str(Path(cwd))
    event_file = Path(events_path)
    watched = [Path(path) for path in watched_paths]
    started_at = clock()
    if not math.isfinite(started_at):
        raise ValueError("clock must return a finite number")
    initial_fingerprints = _fingerprints(watched)
    seen_fatal_patterns: set[tuple[str, str]] = set()

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

    process = popen_factory(argv, cwd=working_directory, shell=False)
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
        current_fingerprints = _fingerprints(watched)
        if current_fingerprints != previous_fingerprints:
            last_progress_at = now
            stall_warned = False
            emit("PROGRESS", now, fingerprints=current_fingerprints)
            previous_fingerprints = current_fingerprints

        for pattern_name, pattern_path in _fatal_matches(watched):
            key = (pattern_name, str(pattern_path))
            if key not in seen_fatal_patterns:
                seen_fatal_patterns.add(key)
                emit("FATAL_PATTERN", now, pattern=pattern_name, path=str(pattern_path))

        emit(
            "STATUS_CHECK",
            now,
            returncode=return_code,
            fingerprints=current_fingerprints,
        )
        if return_code is not None:
            emit("COMPLETED" if return_code == 0 else "CRASHED", now, returncode=return_code)
            return int(return_code)

        elapsed_seconds = now - started_at
        if elapsed_seconds >= policy.hard_timeout_seconds:
            emit("HARD_TIMEOUT", now, timeout_seconds=policy.hard_timeout_seconds)
            process.terminate()
            terminate_deadline = clock() + 10
            while process.poll() is None and clock() < terminate_deadline:
                sleep(min(1.0, terminate_deadline - clock()))
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
