"""Build a deterministic, parentless, source-only Stage-2 execution snapshot."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from numbers import Real
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import zipfile
from typing import Any

from canonical.artifacts import sha256_file, write_json


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MEMBERS = ("source_metadata.json", "stage2_source.bundle")
_ALLOWED_DIRS = {"canonical", "configs", "data", "models", "tests", "training", "utils"}
_EXCLUDED_PARTS = {"ties_results", "ties_unlearn_results", ".venv-stage2", ".uv-cache", "__pycache__", ".stage2_monitor", ".git", ".worktrees", "out"}
_ALLOWED_DOC_FILES = {"docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md"}
_ALLOWED_NOTEBOOK_FILES = {"notebooks/stage2_colab_a100_smoke.ipynb"}
_ROOT_FILES = {
    ".gitignore", "LICENSE", "README.md", "requirements.txt",
    "finish_baselines.py", "finish_sensitivity.py", "freeze_stage2_environment.py",
    "main.py", "monitor_stage2_job.py", "package_stage2_evidence.py",
    "package_stage2_source.py", "plot_mr4_rank_controls.py", "plot_sensitivity.py",
    "run_ablations.py", "run_baselines.py", "run_canonical.py", "run_multiseed.py",
    "run_sensitivity.py", "run_stage2_smoke.py", "validate_stage2_smoke.py",
    "verify_stage2_evidence.py",
}
_EXCLUSIONS_METADATA = sorted(_EXCLUDED_PARTS | {"*.zip", "model weights", "runtime evidence"})


def _strict_json_bytes(payload: bytes, *, source: str) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {source}: {value}")
    try:
        result = json.loads(payload.decode("utf-8"), parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {source}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError(f"{source} must contain an object")
    def finite(value: Any) -> None:
        if isinstance(value, Real) and not isinstance(value, bool) and not math.isfinite(float(value)):
            raise ValueError(f"non-finite JSON number in {source}")
        if isinstance(value, dict):
            for item in value.values(): finite(item)
        elif isinstance(value, list):
            for item in value: finite(item)
    finite(result)
    return result


def _safe_path(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name and all(part not in ("", ".") for part in path.parts)


def _allowed_source(name: str) -> bool:
    if not _safe_path(name):
        return False
    path = PurePosixPath(name)
    if any(part.casefold() in {item.casefold() for item in _EXCLUDED_PARTS} for part in path.parts):
        return False
    if path.suffix.casefold() in {".zip", ".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".pyc"}:
        return False
    return name in _ALLOWED_DOC_FILES or name in _ALLOWED_NOTEBOOK_FILES or (len(path.parts) > 1 and path.parts[0] in _ALLOWED_DIRS) or (len(path.parts) == 1 and name in _ROOT_FILES)


def _git(root: Path, *args: str, input_bytes: bytes | None = None, env: dict[str, str] | None = None) -> bytes:
    try:
        result = subprocess.run(["git", *args], cwd=root, check=True, input=input_bytes, capture_output=True, env=env)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"")
        raise ValueError(f"git command failed: {' '.join(args)}: {detail.decode(errors='replace').strip()}") from error
    return result.stdout


def _git_text(root: Path, *args: str) -> str:
    return _git(root, *args).decode("utf-8").strip()


def _clean_git_metadata(repo_root: Path) -> dict[str, Any]:
    status = _git_text(repo_root, "status", "--porcelain")
    if status:
        raise ValueError("Stage 2 source packaging requires a clean Git working tree")
    commit = _git_text(repo_root, "rev-parse", "HEAD").lower()
    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("Git HEAD must be an exact 40-character hexadecimal commit")
    return {"origin_commit": commit, "branch": _git_text(repo_root, "branch", "--show-current") or None, "dirty": False, "status_porcelain": []}


def _tracked_entries(repo_root: Path) -> list[dict[str, str]]:
    raw = _git(repo_root, "ls-tree", "-r", "-z", "HEAD")
    entries: list[dict[str, str]] = []
    for record in raw.split(b"\0"):
        if not record: continue
        header, raw_name = record.split(b"\t", 1)
        mode, kind, blob = header.decode("ascii").split()
        name = raw_name.decode("utf-8")
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"source snapshot rejects symlink/submodule/special mode: {name}")
        if _allowed_source(name):
            payload = _git(repo_root, "cat-file", "blob", blob)
            entries.append({"path": name, "mode": mode, "blob_sha1": blob, "content_sha256": sha256(payload).hexdigest()})
    if not entries:
        raise ValueError("source allowlist selected no tracked files")
    return sorted(entries, key=lambda item: item["path"].encode("utf-8"))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _create_execution_bundle(origin: Path, entries: list[dict[str, str]], destination: Path) -> str:
    _git(destination, "init", "-q")
    for entry in entries:
        payload = _git(origin, "cat-file", "blob", entry["blob_sha1"])
        written = _git(destination, "hash-object", "-w", "--stdin", input_bytes=payload).decode().strip()
        if written != entry["blob_sha1"]:
            raise ValueError(f"blob identity changed for {entry['path']}")
        _git(destination, "update-index", "--add", "--cacheinfo", f"{entry['mode']},{written},{entry['path']}")
    tree = _git_text(destination, "write-tree")
    fixed_env = dict(os.environ)
    fixed_env.update({"GIT_AUTHOR_NAME":"Stage2 Source Snapshot","GIT_AUTHOR_EMAIL":"stage2@example.invalid","GIT_COMMITTER_NAME":"Stage2 Source Snapshot","GIT_COMMITTER_EMAIL":"stage2@example.invalid","GIT_AUTHOR_DATE":"2000-01-01T00:00:00+00:00","GIT_COMMITTER_DATE":"2000-01-01T00:00:00+00:00"})
    commit = _git(destination, "commit-tree", tree, input_bytes=b"Stage 2 source-only execution snapshot\n", env=fixed_env).decode().strip()
    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("execution commit is invalid")
    _git(destination, "update-ref", "refs/heads/stage2-execution", commit)
    _git(destination, "bundle", "create", str(destination / "stage2_source.bundle"), "refs/heads/stage2-execution")
    return commit


def _zip(output: Path, metadata: bytes, bundle: bytes) -> None:
    if output.exists(): raise ValueError(f"refusing to overwrite existing source archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in (("source_metadata.json", metadata), ("stage2_source.bundle", bundle)):
            info = zipfile.ZipInfo(name, (1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o100644 << 16
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_source_package(repo_root: Path, protocol_path: Path, output_path: Path, *, expectations_output_path: Path | None = None) -> dict[str, Any]:
    repo_root, protocol_path, output_path = Path(repo_root).resolve(), Path(protocol_path).resolve(), Path(output_path).resolve()
    if not protocol_path.is_file(): raise FileNotFoundError(f"frozen protocol not found: {protocol_path}")
    if output_path.exists(): raise ValueError(f"refusing to overwrite existing source archive: {output_path}")
    if expectations_output_path is not None:
        expectations_output_path = Path(expectations_output_path).resolve()
        if expectations_output_path == output_path:
            raise ValueError("source archive and expectations output must be different paths")
        if expectations_output_path.exists(): raise ValueError(f"refusing to overwrite existing expectations: {expectations_output_path}")
    git = _clean_git_metadata(repo_root)
    entries = _tracked_entries(repo_root)
    try:
        protocol_relative = protocol_path.relative_to(repo_root).as_posix()
    except ValueError as error:
        raise ValueError("frozen protocol must be a tracked allowlisted source file") from error
    protocol_entries = [entry for entry in entries if entry["path"] == protocol_relative]
    if not _allowed_source(protocol_relative) or len(protocol_entries) != 1:
        raise ValueError("frozen protocol must be a tracked allowlisted source file at clean HEAD")
    # Bind the exact blob transported in the bundle, independent of checkout
    # newline conversion on the origin workstation.
    protocol_hash = protocol_entries[0]["content_sha256"]
    manifest_hash = sha256(_canonical_bytes(entries)).hexdigest()
    with tempfile.TemporaryDirectory(prefix="stage2-source-v2-") as temporary:
        temp = Path(temporary)
        execution_commit = _create_execution_bundle(repo_root, entries, temp)
        bundle = (temp / "stage2_source.bundle").read_bytes()
    git["execution_commit"] = execution_commit
    metadata = {"schema_version":"stage2_source_package_v2","git":git,"protocol_path":protocol_relative,"protocol_sha256":protocol_hash,"bundle_sha256":sha256(bundle).hexdigest(),"source_manifest":entries,"source_manifest_sha256":manifest_hash,"exclusions":_EXCLUSIONS_METADATA}
    metadata_bytes = json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _zip(output_path, metadata_bytes, bundle)
    if expectations_output_path is not None:
        write_json(expectations_output_path, {"schema_version":"stage2_source_expectations_v1","archive_sha256":sha256_file(output_path),"origin_commit":git["origin_commit"],"execution_commit":execution_commit,"source_manifest_sha256":manifest_hash})
    return metadata


def verify_source_package(archive_path: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    archive_path = Path(archive_path).resolve()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)) or any(not _safe_path(name) for name in names) or set(names)!=set(_MEMBERS): raise ValueError("source archive has unsafe, duplicate, unexpected, or missing members")
            metadata=_strict_json_bytes(archive.read("source_metadata.json"),source="source_metadata.json"); bundle=archive.read("stage2_source.bundle")
    except zipfile.BadZipFile as error: raise ValueError("invalid source archive") from error
    expected_metadata_keys={"schema_version","git","protocol_path","protocol_sha256","bundle_sha256","source_manifest","source_manifest_sha256","exclusions"}
    if set(metadata)!=expected_metadata_keys or metadata.get("schema_version")!="stage2_source_package_v2" or sha256(bundle).hexdigest()!=metadata.get("bundle_sha256") or metadata.get("exclusions") != _EXCLUSIONS_METADATA: raise ValueError("source metadata/bundle is invalid")
    git=metadata.get("git"); entries=metadata.get("source_manifest")
    if not isinstance(git,dict) or set(git)!={"origin_commit","execution_commit","branch","dirty","status_porcelain"} or git.get("dirty") is not False or git.get("status_porcelain") != [] or not (git.get("branch") is None or isinstance(git.get("branch"),str)) or any(_COMMIT_RE.fullmatch(str(git.get(key,""))) is None for key in ("origin_commit","execution_commit")) or not isinstance(entries,list): raise ValueError("source metadata provenance is invalid")
    if not isinstance(metadata.get("protocol_path"),str) or not _allowed_source(metadata["protocol_path"]) or _SHA256_RE.fullmatch(str(metadata.get("protocol_sha256",""))) is None: raise ValueError("source metadata protocol provenance is invalid")
    if sha256(_canonical_bytes(entries)).hexdigest()!=metadata.get("source_manifest_sha256"): raise ValueError("source manifest checksum mismatch")
    if any(not isinstance(entry,dict) or set(entry)!={"path","mode","blob_sha1","content_sha256"} or not isinstance(entry.get("path"),str) or not _allowed_source(entry["path"]) or entry["mode"] not in {"100644","100755"} or _COMMIT_RE.fullmatch(str(entry.get("blob_sha1",""))) is None or _SHA256_RE.fullmatch(str(entry.get("content_sha256",""))) is None for entry in entries): raise ValueError("source manifest entry is unsafe")
    if entries != sorted(entries,key=lambda item:item["path"].encode("utf-8")) or len({entry["path"] for entry in entries}) != len(entries): raise ValueError("source manifest entries must be unique and canonically ordered")
    if not any(entry["path"]==metadata["protocol_path"] and entry["content_sha256"]==metadata["protocol_sha256"] for entry in entries): raise ValueError("source manifest does not bind the frozen protocol")
    with tempfile.TemporaryDirectory(prefix="stage2-source-verify-") as temporary:
        temp=Path(temporary); _git(temp,"init","-q"); bundle_path=temp/"stage2_source.bundle"; bundle_path.write_bytes(bundle)
        _git(temp,"bundle","verify",str(bundle_path))
        heads=_git_text(temp,"bundle","list-heads",str(bundle_path)).splitlines()
        if heads != [f"{git['execution_commit']} refs/heads/stage2-execution"]: raise ValueError("source bundle does not expose exactly the recorded execution ref")
        _git(temp,"fetch",str(bundle_path),f"{git['execution_commit']}:refs/heads/execution")
        commit_text=_git_text(temp,"cat-file","-p",git["execution_commit"])
        if any(line.startswith("parent ") for line in commit_text.splitlines()): raise ValueError("execution commit must be parentless")
        raw=_git(temp,"ls-tree","-r","-z",git["execution_commit"]); actual=[]
        for record in raw.split(b"\0"):
            if not record: continue
            header,name=record.split(b"\t",1); mode,kind,blob=header.decode().split(); path=name.decode()
            if kind!="blob": raise ValueError("execution tree contains non-blob")
            actual.append({"path":path,"mode":mode,"blob_sha1":blob,"content_sha256":sha256(_git(temp,"cat-file","blob",blob)).hexdigest()})
        if actual!=entries: raise ValueError("execution tree does not exactly match source manifest")
    return metadata
