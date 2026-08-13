"""Strict, atomic artifact helpers for canonical experiment outputs."""

from dataclasses import asdict, is_dataclass
from enum import Enum
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from numbers import Integral, Real
from typing import Any, Iterable, Mapping


def json_ready(value: Any, path: str = "$") -> Any:
    """Normalize a value for strict JSON and reject non-finite numbers."""
    if is_dataclass(value):
        return json_ready(asdict(value), path)
    if isinstance(value, Enum):
        return json_ready(value.value, path)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite JSON number at {path}: {value!r}")
        return number
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {path} must be str, got {type(key).__name__}")
            result[key] = json_ready(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [json_ready(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def _atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise


def write_json(path: os.PathLike[str] | str, value: Any) -> None:
    normalized = json_ready(value)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    _atomic_text(Path(path), text)


def write_jsonl(path: os.PathLike[str] | str, records: Iterable[Mapping[str, Any]]) -> None:
    normalized = [json_ready(record, f"$[{index}]") for index, record in enumerate(records)]
    text = "".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
        for record in normalized
    )
    _atomic_text(Path(path), text)


def read_jsonl(path: os.PathLike[str] | str) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(value)
    return records


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_git_metadata(repo_root: os.PathLike[str] | str = ".") -> dict[str, Any]:
    root = str(Path(repo_root))

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
        return completed.stdout.strip()

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current") or None,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def collect_environment_metadata() -> dict[str, Any]:
    """Collect the complete runtime record used by runner and freeze probes."""
    packages = {}
    for distribution in ("torch", "transformers", "datasets", "numpy"):
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[distribution] = None
    environment = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": None,
        "cuda_driver": None,
        "gpu": None,
        "torch_gpu": None,
        "nvidia_smi_gpu": None,
        "pip_freeze": None,
    }
    try:
        import torch

        environment["cuda_runtime"] = str(torch.version.cuda) if torch.version.cuda is not None else None
        if torch.cuda.is_available():
            environment["torch_gpu"] = torch.cuda.get_device_name(0)
            environment["gpu"] = environment["torch_gpu"]
    except (ImportError, RuntimeError):
        pass
    try:
        lines = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        if lines:
            gpu, driver = (part.strip() for part in lines[0].split(",", 1))
            environment["nvidia_smi_gpu"] = gpu
            if environment["gpu"] is None:
                environment["gpu"] = gpu
            environment["cuda_driver"] = driver
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    try:
        environment["pip_freeze"] = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pass
    return environment
