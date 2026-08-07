"""Create and verify the small, clean-commit source transport for Stage 2."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from numbers import Real
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import zipfile
from typing import Any

from canonical.artifacts import sha256_file


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MEMBERS = ("source_metadata.json", "stage2_source.bundle")


def _strict_json_bytes(payload: bytes, *, source: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden in {source}: {value}")

    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {source}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object")

    def check(item: Any, path: str = "$") -> None:
        if isinstance(item, Real) and not isinstance(item, bool) and not math.isfinite(float(item)):
            raise ValueError(f"non-finite JSON number in {source} at {path}")
        if isinstance(item, dict):
            for key, nested in item.items():
                check(nested, f"{path}.{key}")
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                check(nested, f"{path}[{index}]")

    check(value)
    return value


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, check=True, text=True, capture_output=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise ValueError(f"git command failed: {' '.join(args)}: {detail.strip()}") from error
    return result.stdout.strip()


def _clean_git_metadata(repo_root: Path) -> dict[str, Any]:
    status = _git(repo_root, "status", "--porcelain")
    if status:
        raise ValueError("Stage 2 source packaging requires a clean Git working tree")
    commit = _git(repo_root, "rev-parse", "HEAD").lower()
    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("Git HEAD must be an exact 40-character hexadecimal commit")
    return {
        "commit": commit,
        "branch": _git(repo_root, "branch", "--show-current") or None,
        "dirty": False,
        "status_porcelain": [],
    }


def _bundle_verify(repo_root: Path, bundle_path: Path, commit: str) -> None:
    _git(repo_root, "bundle", "verify", str(bundle_path))
    heads = _git(repo_root, "bundle", "list-heads", str(bundle_path)).splitlines()
    if not any(line.split(maxsplit=1)[0].lower() == commit for line in heads if line):
        raise ValueError("git bundle does not contain the recorded HEAD commit")


def _deterministic_zip(output_path: Path, metadata_bytes: bytes, bundle_bytes: bytes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise ValueError(f"refusing to overwrite existing source archive: {output_path}")
    with zipfile.ZipFile(output_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in (("source_metadata.json", metadata_bytes), ("stage2_source.bundle", bundle_bytes)):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_source_package(repo_root: Path, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    """Create a deterministic zip containing exactly a verified HEAD bundle and metadata."""
    repo_root = Path(repo_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    output_path = Path(output_path).resolve()
    if not protocol_path.is_file():
        raise FileNotFoundError(f"frozen protocol not found: {protocol_path}")
    git = _clean_git_metadata(repo_root)
    protocol_sha256 = sha256_file(protocol_path)
    with tempfile.TemporaryDirectory(prefix="stage2-source-") as temporary:
        bundle = Path(temporary) / "stage2_source.bundle"
        _git(repo_root, "bundle", "create", str(bundle), "HEAD")
        _bundle_verify(repo_root, bundle, git["commit"])
        bundle_bytes = bundle.read_bytes()
    metadata = {
        "schema_version": "stage2_source_package_v1",
        "git": git,
        "protocol_sha256": protocol_sha256,
        "bundle_sha256": sha256(bundle_bytes).hexdigest(),
    }
    metadata_bytes = (json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _deterministic_zip(output_path, metadata_bytes, bundle_bytes)
    return metadata


def verify_source_package(archive_path: Path, *, repo_root: Path) -> dict[str, Any]:
    """Validate archive membership, hashes, metadata and the Git bundle without extracting unsafely."""
    archive_path = Path(archive_path).resolve()
    repo_root = Path(repo_root).resolve()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or any(not _safe_member(name) for name in names):
                raise ValueError("source archive has unsafe or duplicate member paths")
            if set(names) != set(_MEMBERS):
                raise ValueError("source archive has unexpected or missing members")
            metadata_bytes = archive.read("source_metadata.json")
            bundle_bytes = archive.read("stage2_source.bundle")
    except zipfile.BadZipFile as error:
        raise ValueError(f"invalid source archive: {archive_path}") from error
    metadata = _strict_json_bytes(metadata_bytes, source="source_metadata.json")
    if metadata.get("schema_version") != "stage2_source_package_v1":
        raise ValueError("source metadata schema is invalid")
    git = metadata.get("git")
    if not isinstance(git, dict) or git.get("dirty") is not False:
        raise ValueError("source metadata must bind a clean Git tree")
    commit = git.get("commit")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("source metadata commit must be an exact 40-character hexadecimal value")
    for field, actual in (("protocol_sha256", metadata.get("protocol_sha256")), ("bundle_sha256", metadata.get("bundle_sha256"))):
        if not isinstance(actual, str) or _SHA256_RE.fullmatch(actual) is None:
            raise ValueError(f"source metadata {field} must be a SHA-256 hexadecimal value")
    if sha256(bundle_bytes).hexdigest() != metadata["bundle_sha256"]:
        raise ValueError("source bundle SHA-256 does not match source metadata")
    with tempfile.TemporaryDirectory(prefix="stage2-source-verify-") as temporary:
        bundle = Path(temporary) / "stage2_source.bundle"
        bundle.write_bytes(bundle_bytes)
        _bundle_verify(repo_root, bundle, commit)
    return metadata
