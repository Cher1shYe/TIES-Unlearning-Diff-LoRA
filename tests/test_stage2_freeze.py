import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.artifacts import sha256_file, write_json
from canonical.data import build_hans_selection_integrity
from canonical.freeze import (
    _commands,
    _strict_data_manifest,
    _strict_environment,
    _write_inventory,
    build_evidence_archive,
    build_freeze_bundle,
    verify_freeze_bundle,
)
from canonical.source_package import _allowed_source, _tracked_entries, build_source_package, verify_source_package
from canonical.stage2_validation import compare_a100_repeat, validate_smoke_root
from tests.test_stage2_validation import (
    TEST_HANS_ANCHORS,
    TEST_STAGE2_PROFILE,
)


def _digest(value):
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


_FREEZE_SPLIT_PAYLOAD = {
    "schema_version": "hans_split_v1",
    "hans_split_seed": 42,
    "build_pair_ids": ["ex0"],
    "dev_pair_ids": ["ex1"],
    "small_strata": [],
}
TEST_FREEZE_HANS_ANCHORS = {
    "schema_version": "hans_official_semantic_anchors_v2",
    "split_checksum": _digest(_FREEZE_SPLIT_PAYLOAD),
    "partitions": {
        "build": {
            "count": 1,
            "source_pair_ids_sha256": _digest(["ex0"]),
            "qualified_ids_sha256": _digest(["hans_train::ex0"]),
            "content_sha256_ordered_checksum": _digest(["1" * 64]),
            "source_id_content_joint_checksum": _digest([["hans_train::ex0", "1" * 64]]),
        },
        "dev": {
            "count": 1,
            "source_pair_ids_sha256": _digest(["ex1"]),
            "qualified_ids_sha256": _digest(["hans_train::ex1"]),
            "content_sha256_ordered_checksum": _digest(["2" * 64]),
            "source_id_content_joint_checksum": _digest([["hans_train::ex1", "2" * 64]]),
        },
        "evaluation": {
            "count": 2,
            "source_pair_ids_sha256": _digest(["ex0", "ex1"]),
            "qualified_ids_sha256": _digest(["hans_evaluation::ex0", "hans_evaluation::ex1"]),
            "content_sha256_ordered_checksum": _digest(["3" * 64, "4" * 64]),
            "source_id_content_joint_checksum": _digest(
                [["hans_evaluation::ex0", "3" * 64], ["hans_evaluation::ex1", "4" * 64]]
            ),
        },
    },
    "selection_full": {
        "count": 2,
        "selected_source_pair_ids_sha256": _digest(["ex0", "ex1"]),
        "selected_artifact_ids_sha256": _digest(["hans_evaluation::ex0", "hans_evaluation::ex1"]),
        "source_to_artifact_mapping_sha256": _digest(
            [["ex0", "hans_evaluation::ex0"], ["ex1", "hans_evaluation::ex1"]]
        ),
    },
}


class CanonicalManifestBackend:
    def __init__(self, config):
        self.config = config

    def initialize_manifests(self, output_dir, _protocol_path):
        manifests = Path(output_dir) / "manifests"
        provenance = {
            ("mnli", "train"): ("nyu-mll/glue:mnli", "train", ["idx", "row_id", "id", "uid"], []),
            ("mnli", "validation_matched"): ("nyu-mll/glue:mnli", "validation_matched", ["idx", "row_id", "id", "uid"], []),
            ("hans", "build"): ("tommccoy1/hans", "build", ["pairID"], []),
            ("hans", "dev"): ("tommccoy1/hans", "dev", ["pairID"], []),
            ("hans", "evaluation"): ("tommccoy1/hans", "evaluation", ["pairID"], ["gold_label", "heuristic", "subcase"]),
            ("ood", "esnli"): ("e-SNLI", "test", ["pairID", "uid", "id", "idx"], []),
            ("ood", "anli"): ("facebook/anli", "test", ["pairID", "uid", "id", "idx"], []),
            ("ood", "snli_hard"): ("snli_1.0_test_hard", "test", ["pairID", "uid", "id", "idx"], []),
            ("ood", "wanli"): ("alisawuffles/WANLI", "test", ["pairID", "uid", "id", "idx"], []),
        }
        def entry(group, name, count):
            if group == "hans":
                namespace = "hans_evaluation" if name == "evaluation" else "hans_train"
                offset = 1 if name == "dev" else 0
                values = [f"{namespace}::ex{index + offset}" for index in range(count)]
            else:
                values = [f"{name}-{index}" for index in range(count)]
            payload = json.dumps(values, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            import hashlib
            digest = hashlib.sha256(payload).hexdigest()
            source, split, preferred, strata = provenance[(group, name)]
            selected_limit = count if group == "mnli" else None
            return {"source": source, "split": split, "id_strategy": "preferred_field_or_content_sha256", "preferred_id_fields": preferred, "strata_fields": strata, "selection_seed": 42, "selected_limit": selected_limit, "full_count": count, "selected_count": count, "full_ids": values, "selected_ids": values, "full_ids_sha256": digest, "selected_ids_sha256": digest}
        hans_entries = {
            name: entry("hans", name, 2 if name == "evaluation" else 1)
            for name in ("build", "dev", "evaluation")
        }
        content_hashes = {
            "build": ["1" * 64],
            "dev": ["2" * 64],
            "evaluation": ["3" * 64, "4" * 64],
        }
        partitions = {}
        for name, hashes in content_hashes.items():
            ordered = json.dumps(hashes, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            joint = json.dumps(list(zip(hans_entries[name]["full_ids"], hashes)), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            import hashlib
            partitions[name] = {
                "count": len(hashes),
                "content_sha256": hashes,
                "content_sha256_ordered_checksum": hashlib.sha256(ordered).hexdigest(),
                "source_id_content_joint_checksum": hashlib.sha256(joint).hexdigest(),
                "duplicate_content_count": 0,
            }
        hans_entries["content_integrity"] = {
            "schema_version": "hans_content_integrity_v1",
            "algorithm": "sha256_canonical_json_utf8_v1",
            "fields": ["gold_label", "premise", "hypothesis", "heuristic", "subcase"],
            "excludes_pair_id": True,
            "partitions": partitions,
            "overlap_counts": {"build_dev": 0, "build_evaluation": 0, "dev_evaluation": 0},
        }
        hans_entries["split_integrity"] = {
            "schema_version": "hans_split_integrity_v1",
            "seed": 42,
            "split_algorithm": "source_local_id_sort_numpy_default_rng_per_stratum_v1",
            "checksum_algorithm": "sha256_canonical_json_utf8_v1",
            "build_count": 1,
            "dev_count": 1,
            "build_source_pair_ids": ["ex0"],
            "dev_source_pair_ids": ["ex1"],
            "small_strata": [],
            "split_checksum": _digest(_FREEZE_SPLIT_PAYLOAD),
        }
        hans_entries["selection_integrity"] = build_hans_selection_integrity(
            [
                {"pairID": f"ex{index}", "canonical_pair_id": f"hans_evaluation::ex{index}"}
                for index in range(2)
            ],
            hans_entries["evaluation"]["selected_ids"],
            limit=None,
            seed=42,
        )
        write_json(
            manifests / "data_manifest.json",
            {
                "schema_version": "canonical_data_manifest_v4",
                "scope": "canonical_v1",
                "data_seed": 42,
                "hans_split_seed": 42,
                "mnli": {"train": entry("mnli", "train", 100000), "validation_matched": entry("mnli", "validation_matched", 5000)},
                "hans": hans_entries,
                "ood": {name: entry("ood", name, 1) for name in ("esnli", "anli", "snli_hard", "wanli")},
            },
        )


class ZeroArgumentCanonicalManifestBackend(CanonicalManifestBackend):
    def __init__(self):
        self.config = type("Config", (), {"mnli_train_size": 100_000})()


class Stage2FreezeTest(unittest.TestCase):
    def setUp(self):
        self._contract_patches = (
            patch("canonical.freeze.HANS_OFFICIAL_ANCHORS_V2", TEST_FREEZE_HANS_ANCHORS),
            patch("canonical.stage2_validation.HANS_OFFICIAL_ANCHORS_V2", TEST_HANS_ANCHORS),
            patch("canonical.stage2_validation._STAGE2_DATA_PROFILE", TEST_STAGE2_PROFILE),
        )
        for contract_patch in self._contract_patches:
            contract_patch.start()
            self.addCleanup(contract_patch.stop)

    def _git(self, repo, *args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)

    def _source_repo(self, directory):
        repo = Path(directory) / "repo"
        repo.mkdir()
        self._git(repo, "init")
        self._git(repo, "config", "user.email", "fixture@example.test")
        self._git(repo, "config", "user.name", "Fixture")
        (repo / "tracked.txt").write_text("source\n", encoding="utf-8")
        (repo / ".gitignore").write_text("ties_results/\n", encoding="utf-8")
        protocol = repo / "docs" / "paper_rebuild" / "FROZEN_EXPERIMENT_PROTOCOL.md"
        protocol.parent.mkdir(parents=True)
        protocol.write_bytes(b"# frozen\n")
        self._git(repo, "add", ".")
        self._git(repo, "commit", "-m", "fixture")
        return repo, protocol

    def _execution_inputs(self, directory):
        origin, origin_protocol = self._source_repo(directory)
        archive = Path(directory) / "stage2_source.zip"
        expectations = Path(directory) / "stage2_source_expectations.json"
        metadata = build_source_package(origin, origin_protocol, archive, expectations_output_path=expectations)
        unpack = Path(directory) / "source_unpack"
        with zipfile.ZipFile(archive) as source:
            source.extractall(unpack)
        repo = Path(directory) / "execution_repo"
        subprocess.run(["git", "-c", "core.autocrlf=false", "clone", str(unpack / "stage2_source.bundle"), str(repo)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "config", "core.autocrlf", "false"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", metadata["git"]["execution_commit"]], check=True, capture_output=True)
        return repo, repo / "docs" / "paper_rebuild" / "FROZEN_EXPERIMENT_PROTOCOL.md", archive, expectations

    def _a100_smoke(self, directory):
        smoke = Path(directory) / "ties_results" / "stage2_smoke" / "colab_a100_run1"
        manifests = smoke / "manifests"
        write_json(
            manifests / "environment_manifest.json",
            {
                "schema_version": "canonical_environment_manifest_v1",
                "gpu": "NVIDIA A100-SXM4-40GB",
                "torch_gpu": "NVIDIA A100-SXM4-40GB",
                "nvidia_smi_gpu": "NVIDIA A100-SXM4-40GB",
                "python": "3.12.0",
                "python_implementation": "CPython",
                "platform": "Linux-test",
                "cuda_runtime": "12.8",
                "cuda_driver": "555.1",
                "packages": {"torch": "2.11.0", "transformers": "5.0", "datasets": "4.0", "numpy": "2.0"},
                "pip_freeze": ["datasets==4.0", "numpy==2.0", "torch==2.11.0", "transformers==5.0"],
            },
        )
        git_repo = Path(directory) / "execution_repo"
        if not git_repo.exists():
            git_repo = Path(directory) / "repo"
        commit = self._git(git_repo, "rev-parse", "HEAD").stdout.strip()
        for root, tags in ((smoke, ("shared_phase2", "standard_lora", "full_sr", "class_prior_reweight")), (smoke.parent / "colab_a100_repeat_full_sr", ("shared_phase2", "full_sr"))):
            copy_environment = root / "manifests" / "environment_manifest.json"
            copy_environment.parent.mkdir(parents=True, exist_ok=True)
            copy_environment.write_bytes((manifests / "environment_manifest.json").read_bytes())
            mode = "primary" if root == smoke else "repeat_full_sr"
            argv = [sys.executable, str((git_repo / "run_stage2_smoke.py").resolve()), "--mode", mode, "--environment", "colab_a100", "--protocol", "docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md", "--output-dir", f"ties_results/stage2_smoke/{root.name}", "--fresh"]
            write_json(root / "commands.json", {"schema_version": "stage2_smoke_commands_v1", "mode": mode, "environment": "colab_a100", "argv": argv, "expected_condition_tags": list(tags[1:]), "profile_name": "stage2_smoke_v1", "gpu_name": "NVIDIA A100-SXM4-40GB", "started_at": "2026-08-08T00:00:00+00:00"})
            for tag in tags:
                write_json(root / "seed_42" / tag / "run_manifest.json", {"git": {"commit": commit, "branch": None, "dirty": False, "status_porcelain": []}, "command": argv})
        write_json(
            smoke / "stage2_validation.json",
            {
                "schema_version": "stage2_validation_v1",
                "state": "pass",
                "repeat_comparison": self._validated_a100(smoke, smoke.parent / "colab_a100_repeat_full_sr"),
            },
        )
        (smoke / "protocol_snapshot").mkdir()
        (smoke / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md").write_bytes(b"# frozen\n")
        return smoke

    def _validated_a100(self, primary, _repeat, **_kwargs):
        return {"schema_version": "stage2_a100_repeat_comparison_v1", "state": "pass"}

    def _production_smoke_pair(self, repo):
        from tests.test_stage2_validation import _create_smoke_root, _repeat_root, _rehash_status
        base = Path(repo) / "ties_results" / "stage2_smoke"
        generated_primary = _create_smoke_root(base / "generated-primary")
        generated_repeat = _repeat_root(base / "generated-repeat")
        primary = base / "colab_a100_run1"
        repeat = base / "colab_a100_repeat_full_sr"
        generated_primary.rename(primary)
        generated_repeat.rename(repeat)
        commit = self._git(repo, "rev-parse", "HEAD").stdout.strip()
        environment = {
            "schema_version": "canonical_environment_manifest_v1", "gpu": "NVIDIA A100-SXM4-40GB",
            "torch_gpu": "NVIDIA A100-SXM4-40GB", "nvidia_smi_gpu": "NVIDIA A100-SXM4-40GB",
            "python": "3.12.0", "python_implementation": "CPython", "platform": "Linux-test", "cuda_runtime": "12.8", "cuda_driver": "555.1",
            "packages": {"torch": "2.11.0", "transformers": "5.0", "datasets": "4.0", "numpy": "2.0"},
            "pip_freeze": ["datasets==4.0", "numpy==2.0", "torch==2.11.0", "transformers==5.0"],
        }
        for root, mode, tags in ((primary, "primary", ["standard_lora", "full_sr", "class_prior_reweight"]), (repeat, "repeat_full_sr", ["full_sr"])):
            protocol_snapshot = root / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md"
            protocol_snapshot.write_bytes((Path(repo) / "docs" / "paper_rebuild" / "FROZEN_EXPERIMENT_PROTOCOL.md").read_bytes())
            protocol_hash = sha256_file(protocol_snapshot)
            (root / "protocol_snapshot" / "protocol_sha256.txt").write_text(protocol_hash + "\n", encoding="utf-8")
            write_json(root / "manifests" / "environment_manifest.json", environment)
            env_hash = sha256_file(root / "manifests" / "environment_manifest.json")
            argv = [sys.executable,str((Path(repo) / "run_stage2_smoke.py").resolve()),"--mode",mode,"--environment","colab_a100","--protocol","docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md","--output-dir",f"ties_results/stage2_smoke/{root.name}","--fresh"]
            write_json(root / "commands.json", {"schema_version":"stage2_smoke_commands_v1","mode":mode,"environment":"colab_a100","argv":argv,"expected_condition_tags":tags,"profile_name":"stage2_smoke_v1","gpu_name":"NVIDIA A100-SXM4-40GB","started_at":"2026-08-08T00:00:00+00:00"})
            checkpoint = root / "seed_42" / "shared_phase2" / "checkpoints" / "shared.pt"
            metadata_path = root / "seed_42" / "shared_phase2" / "shared_checkpoint_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["checkpoint_path"] = str(checkpoint)
            write_json(metadata_path, metadata)
            checkpoint_ref = {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
            for manifest_path in (root / "seed_42").glob("*/run_manifest.json"):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["git"]["commit"] = commit
                manifest["git"] = {"commit": commit, "branch": None, "dirty": False, "status_porcelain": []}
                manifest["command"] = argv
                manifest["protocol_sha256"] = protocol_hash
                manifest["environment_manifest_sha256"] = env_hash
                if manifest_path.parent.name not in {"shared_phase2", "standard_lora"}:
                    manifest["shared_phase2_checkpoint"] = checkpoint_ref
                write_json(manifest_path, manifest)
                _rehash_status(manifest_path.parent)
        canonical_dir = Path(repo) / "ties_results" / "canonical_v1"
        comparison = compare_a100_repeat(primary, repeat, canonical_dir=canonical_dir)
        primary_report = validate_smoke_root(primary, expected_conditions=("standard_lora", "full_sr", "class_prior_reweight"), canonical_dir=canonical_dir)
        primary_report["repeat_comparison"] = comparison
        write_json(primary / "stage2_validation.json", primary_report)
        from canonical.stage2_validation import render_validation_markdown
        (primary / "stage2_validation.md").write_text(render_validation_markdown(primary_report), encoding="utf-8")
        repeat_report = validate_smoke_root(repeat, expected_conditions=("full_sr",), canonical_dir=canonical_dir)
        write_json(repeat / "stage2_validation.json", repeat_report)
        (repeat / "stage2_validation.md").write_text(render_validation_markdown(repeat_report), encoding="utf-8")
        return primary, repeat

    @staticmethod
    def _fake_gpu_probe():
        return {"nvidia_smi": "NVIDIA A100-SXM4-40GB", "torch_gpu": "NVIDIA A100-SXM4-40GB", "torch_cuda": "12.8"}

    @staticmethod
    def _environment_probe():
        return {
            "schema_version": "canonical_environment_manifest_v1", "gpu": "NVIDIA A100-SXM4-40GB",
            "torch_gpu": "NVIDIA A100-SXM4-40GB", "nvidia_smi_gpu": "NVIDIA A100-SXM4-40GB",
            "python": "3.12.0", "python_implementation": "CPython", "platform": "Linux-test",
            "cuda_runtime": "12.8", "cuda_driver": "555.1",
            "packages": {"torch": "2.11.0", "transformers": "5.0", "datasets": "4.0", "numpy": "2.0"},
            "pip_freeze": ["datasets==4.0", "numpy==2.0", "torch==2.11.0", "transformers==5.0"],
        }

    @staticmethod
    def _write_monitor(path, command, cwd):
        from canonical.monitoring import PRODUCTION_POLICY, monitor_command

        class CompletedProcess:
            def poll(self):
                return 0

        Path(path).unlink(missing_ok=True)
        monitor_command(
            command, cwd=cwd, events_path=path,
            watched_paths=[Path(cwd) / "ties_results" / "stage2_smoke"],
            policy=PRODUCTION_POLICY, clock=lambda: 0.0, sleep=lambda _seconds: None,
            popen_factory=lambda *args, **kwargs: CompletedProcess(),
        )

    @staticmethod
    def _rewrite_evidence(source, destination, transform):
        with zipfile.ZipFile(source) as archive:
            members = {info.filename: archive.read(info.filename) for info in archive.infolist()}
        transform(members)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)

    def test_source_package_binds_clean_commit_protocol_and_verified_git_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol = self._source_repo(tmp)
            archive = Path(tmp) / "stage2_source.zip"

            expectations = Path(tmp) / "stage2_source_expectations.json"
            metadata = build_source_package(repo, protocol, archive, expectations_output_path=expectations)

            self.assertFalse(metadata["git"]["dirty"])
            self.assertRegex(metadata["git"]["origin_commit"], r"^[0-9a-f]{40}$")
            self.assertRegex(metadata["git"]["execution_commit"], r"^[0-9a-f]{40}$")
            self.assertNotEqual(metadata["git"]["origin_commit"], metadata["git"]["execution_commit"])
            self.assertEqual("stage2_source_package_v2", metadata["schema_version"])
            self.assertEqual(64, len(metadata["protocol_sha256"]))
            self.assertEqual(64, len(metadata["bundle_sha256"]))
            self.assertEqual(metadata, verify_source_package(archive, repo_root=repo))
            sidecar = json.loads(expectations.read_text(encoding="utf-8"))
            self.assertEqual(metadata["git"]["execution_commit"], sidecar["execution_commit"])
            self.assertEqual(sha256_file(archive), sidecar["archive_sha256"])
            with zipfile.ZipFile(archive) as source:
                self.assertEqual(["source_metadata.json", "stage2_source.bundle"], sorted(source.namelist()))

    def test_source_package_rejects_dirty_repositories_and_unsafe_archive_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol = self._source_repo(tmp)
            (repo / "dirty.txt").write_text("no\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "clean"):
                build_source_package(repo, protocol, Path(tmp) / "dirty.zip")

            unsafe = Path(tmp) / "unsafe.zip"
            with zipfile.ZipFile(unsafe, "w") as archive:
                archive.writestr("../source_metadata.json", "{}")
                archive.writestr("stage2_source.bundle", b"not-a-bundle")
            with self.assertRaisesRegex(ValueError, "unsafe|unexpected"):
                verify_source_package(unsafe, repo_root=repo)

    def test_source_snapshot_is_deterministic_parentless_and_excludes_runtime_and_origin_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol = self._source_repo(tmp)
            runtime = repo / "ties_results" / "stage2_smoke" / "leak.json"
            runtime.parent.mkdir(parents=True)
            runtime.write_text("{}\n", encoding="utf-8")
            self._git(repo, "add", "-f", "ties_results/stage2_smoke/leak.json")
            self._git(repo, "commit", "-m", "tracked runtime must still be excluded")
            first, second = Path(tmp) / "first.zip", Path(tmp) / "second.zip"
            meta1 = build_source_package(repo, protocol, first)
            meta2 = build_source_package(repo, protocol, second)

            self.assertEqual(meta1["git"]["execution_commit"], meta2["git"]["execution_commit"])
            self.assertEqual(sha256_file(first), sha256_file(second))
            self.assertTrue(all(not entry["path"].startswith("ties_results/") for entry in meta1["source_manifest"]))
            unpack = Path(tmp) / "unpack"
            with zipfile.ZipFile(first) as source:
                source.extractall(unpack)
            clone = Path(tmp) / "clone"
            subprocess.run(["git", "clone", str(unpack / "stage2_source.bundle"), str(clone)], check=True, capture_output=True)
            execution = meta1["git"]["execution_commit"]
            parents = self._git(clone, "rev-list", "--parents", "-n", "1", execution).stdout.split()
            self.assertEqual([execution], parents)
            history_probe = subprocess.run(["git", "-C", str(clone), "cat-file", "-e", meta1["git"]["origin_commit"]], capture_output=True)
            self.assertNotEqual(0, history_probe.returncode)

    def test_source_snapshot_rejects_tracked_symlink_and_submodule_modes(self):
        for special_mode in ("120000", "160000"):
            with self.subTest(mode=special_mode), tempfile.TemporaryDirectory() as tmp:
                repo, protocol = self._source_repo(tmp)
                if special_mode == "120000":
                    blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=repo, input=b"target", check=True, capture_output=True).stdout.decode().strip()
                else:
                    blob = self._git(repo, "rev-parse", "HEAD").stdout.strip()
                self._git(repo, "update-index", "--add", "--cacheinfo", f"{special_mode},{blob},canonical/special")
                self._git(repo, "commit", "-m", f"special mode {special_mode}")
                with self.assertRaisesRegex(ValueError, "symlink/submodule"):
                    _tracked_entries(repo)

    def test_source_allowlist_excludes_private_notes_reports_and_nonruntime_docs(self):
        forbidden = (
            "MR4_REVISION_NOTES.md",
            "docs/paper_rebuild/PAPER_REBUILD_MASTER_CHECKLIST.md",
            "docs/paper_rebuild/STAGE1_CANONICAL_INFRASTRUCTURE_REPORT.md",
            "docs/superpowers/plans/2026-08-08-stage2-smoke-environment-freeze.md",
            "docs/superpowers/specs/2026-08-08-stage2-smoke-environment-freeze-design.md",
            "manuscript/private_notes.md",
        )
        self.assertTrue(_allowed_source("README.md"))
        self.assertTrue(_allowed_source("docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md"))
        self.assertTrue(_allowed_source("notebooks/stage2_colab_a100_smoke.ipynb"))
        self.assertTrue(_allowed_source("run_stage2_smoke.py"))
        self.assertTrue(all(not _allowed_source(path) for path in forbidden))
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol = self._source_repo(tmp)
            for relative in forbidden:
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("private\n", encoding="utf-8")
            runtime = repo / "canonical" / "runtime.py"
            runtime.parent.mkdir(parents=True)
            runtime.write_text("VALUE = 1\n", encoding="utf-8")
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-m", "allowlist fixture")
            metadata = build_source_package(
                repo, protocol, Path(tmp) / "source.zip",
                expectations_output_path=Path(tmp) / "expectations.json",
            )
            paths = {entry["path"] for entry in metadata["source_manifest"]}
            self.assertIn("canonical/runtime.py", paths)
            self.assertTrue(paths.isdisjoint(forbidden))

    def test_current_source_snapshot_clone_contains_notebook_excludes_private_plan_and_runs_full_suite(self):
        if os.environ.get("STAGE2_PACKAGED_CLONE_CHILD") == "1":
            self.skipTest("packaged clone child does not recursively package itself")
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "current-origin"
            snapshot.mkdir()
            listed = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                cwd=ROOT, check=True, capture_output=True,
            ).stdout.split(b"\0")
            for raw in listed:
                if not raw:
                    continue
                relative = raw.decode("utf-8")
                source = ROOT / relative
                if not source.is_file():
                    continue
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            self._git(snapshot, "init")
            self._git(snapshot, "config", "user.email", "fixture@example.test")
            self._git(snapshot, "config", "user.name", "Fixture")
            self._git(snapshot, "add", ".")
            self._git(snapshot, "commit", "-m", "current source integration")
            archive = Path(tmp) / "source.zip"
            expectations = Path(tmp) / "expectations.json"
            metadata = build_source_package(
                snapshot,
                snapshot / "docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md",
                archive,
                expectations_output_path=expectations,
            )
            unpack = Path(tmp) / "unpack"
            with zipfile.ZipFile(archive) as source_zip:
                source_zip.extractall(unpack)
            clone = Path(tmp) / "execution"
            subprocess.run(["git", "clone", str(unpack / "stage2_source.bundle"), str(clone)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(clone), "checkout", "--detach", metadata["git"]["execution_commit"]], check=True, capture_output=True)
            self.assertTrue((clone / "notebooks/stage2_colab_a100_smoke.ipynb").is_file())
            self.assertFalse((clone / "docs/superpowers/plans/2026-08-08-stage2-smoke-environment-freeze.md").exists())
            self.assertFalse((clone / "docs/superpowers/specs/2026-08-08-stage2-smoke-environment-freeze-design.md").exists())
            child_environment = dict(os.environ)
            child_environment["STAGE2_PACKAGED_CLONE_CHILD"] = "1"
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                cwd=clone, env=child_environment, text=True, capture_output=True, timeout=240,
            )
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_source_snapshot_preflights_archive_and_expectations_overwrite_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol = self._source_repo(tmp)
            archive = Path(tmp) / "stage2_source.zip"
            expectations = Path(tmp) / "stage2_source_expectations.json"
            expectations.write_text("do not overwrite\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expectations"):
                build_source_package(repo, protocol, archive, expectations_output_path=expectations)

            self.assertFalse(archive.exists())
            self.assertEqual("do not overwrite\n", expectations.read_text(encoding="utf-8"))

    def test_strict_data_environment_and_commands_recompute_semantics_not_just_inventory_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = CanonicalManifestBackend(type("Config", (), {"mnli_train_size": 100_000})())
            backend.initialize_manifests(root / "canonical", root / "protocol.md")
            data_path = root / "canonical" / "manifests" / "data_manifest.json"
            data = json.loads(data_path.read_text(encoding="utf-8"))
            _strict_data_manifest(data)
            data["mnli"]["train"]["full_ids_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                _strict_data_manifest(data)

            repo, _protocol, _archive, _expectations = self._execution_inputs(tmp)
            smoke = self._a100_smoke(tmp)
            environment = json.loads((smoke / "manifests" / "environment_manifest.json").read_text(encoding="utf-8"))
            _strict_environment(environment)
            environment["packages"]["numpy"] = "9.9"
            with self.assertRaisesRegex(ValueError, "pip freeze"):
                _strict_environment(environment)
            environment = json.loads((smoke / "manifests" / "environment_manifest.json").read_text(encoding="utf-8"))
            environment["nvidia_smi_gpu"] = "NVIDIA L4"
            with self.assertRaisesRegex(ValueError, "GPU"):
                _strict_environment(environment)
            command = _commands(smoke, smoke / "commands.json", mode="primary", gpu="NVIDIA A100-SXM4-40GB")
            self.assertEqual("primary", command["mode"])
            command["argv"].remove("--fresh")
            write_json(smoke / "commands.json", command)
            with self.assertRaisesRegex(ValueError, "argv"):
                _commands(smoke, smoke / "commands.json", mode="primary", gpu="NVIDIA A100-SXM4-40GB")

            command["argv"] = ["python", "run_stage2_smoke.py", "--mode", "--environment", "primary", "colab_a100", "--output-dir", str(smoke), "--fresh"]
            write_json(smoke / "commands.json", command)
            with self.assertRaisesRegex(ValueError, "argv"):
                _commands(smoke, smoke / "commands.json", mode="primary", gpu="NVIDIA A100-SXM4-40GB")

            data = json.loads(data_path.read_text(encoding="utf-8"))
            data["hans"]["build"]["full_ids"] = [1]
            data["hans"]["build"]["selected_ids"] = [1]
            import hashlib
            numeric_hash = hashlib.sha256(b"[1]").hexdigest()
            data["hans"]["build"]["full_ids_sha256"] = numeric_hash
            data["hans"]["build"]["selected_ids_sha256"] = numeric_hash
            with self.assertRaisesRegex(ValueError, "identity|ID"):
                _strict_data_manifest(data)

            data = json.loads(data_path.read_text(encoding="utf-8"))
            train = data["mnli"]["train"]
            train["full_ids"].append("unexpected-100001")
            train["selected_ids"].append("unexpected-100001")
            train["full_count"] = train["selected_count"] = 100001
            import hashlib
            checksum = hashlib.sha256(json.dumps(train["full_ids"], ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            train["full_ids_sha256"] = train["selected_ids_sha256"] = checksum
            train["selected_limit"] = 100001
            with self.assertRaisesRegex(ValueError, "count|provenance"):
                _strict_data_manifest(data)

            data = json.loads(data_path.read_text(encoding="utf-8"))
            empty = data["ood"]["wanli"]
            empty["full_ids"] = empty["selected_ids"] = []
            empty["full_count"] = empty["selected_count"] = 0
            empty_hash = hashlib.sha256(b"[]").hexdigest()
            empty["full_ids_sha256"] = empty["selected_ids_sha256"] = empty_hash
            with self.assertRaisesRegex(ValueError, "nonempty|count"):
                _strict_data_manifest(data)

            data = json.loads(data_path.read_text(encoding="utf-8"))
            data["hans"]["dev"]["full_ids"] = list(data["hans"]["build"]["full_ids"])
            data["hans"]["dev"]["selected_ids"] = list(data["hans"]["build"]["full_ids"])
            data["hans"]["dev"]["full_ids_sha256"] = data["hans"]["build"]["full_ids_sha256"]
            data["hans"]["dev"]["selected_ids_sha256"] = data["hans"]["build"]["full_ids_sha256"]
            with self.assertRaisesRegex(ValueError, "disjoint"):
                _strict_data_manifest(data)

            for invalid_id in (
                "ex0",
                "hans_train::ex2",
                "hans_evaluation::hans_evaluation::ex0",
                "hans_evaluation::ex00",
                "hans_evaluation::ex-1",
                "hans_evaluation::ex1::extra",
            ):
                with self.subTest(invalid_id=invalid_id):
                    data = json.loads(data_path.read_text(encoding="utf-8"))
                    evaluation = data["hans"]["evaluation"]
                    invalid_ids = [invalid_id, evaluation["full_ids"][1]]
                    evaluation["full_ids"] = evaluation["selected_ids"] = invalid_ids
                    digest = hashlib.sha256(
                        json.dumps(
                            invalid_ids,
                            ensure_ascii=False,
                            allow_nan=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    evaluation["full_ids_sha256"] = evaluation["selected_ids_sha256"] = digest
                    with self.assertRaisesRegex(ValueError, "HANS manifest"):
                        _strict_data_manifest(data)

    def test_notebook_and_cli_contracts_are_fail_fast_and_sidecar_bound(self):
        notebook = (ROOT / "notebooks" / "stage2_colab_a100_smoke.ipynb").read_text(encoding="utf-8")
        parsed_notebook = json.loads(notebook)
        for index, cell in enumerate(parsed_notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
        self.assertNotIn("!python", notebook)
        self.assertNotIn("from canonical.freeze import", notebook)
        self.assertNotIn("subprocess.check_output", notebook)
        self.assertIn("package_stage2_evidence.py", notebook)
        self.assertIn("verify_stage2_evidence.py", notebook)
        self.assertNotIn("'docs','models','notebooks'", notebook)
        self.assertIn("source_manifest_sha256", notebook)
        self.assertIn("--source-expectations", notebook)
        self.assertIn("parents!=[head]", notebook)
        self.assertIn("actual!=manifest", notebook)
        self.assertIn("core.autocrlf=false", notebook)
        self.assertIn("torch.cuda.is_available", notebook)
        self.assertIn("sys.executable,str((repo/'monitor_stage2_job.py').resolve())", notebook)
        self.assertIn("sys.executable,str((repo/'run_stage2_smoke.py').resolve())", notebook)
        self.assertNotIn("'--','python','run_stage2_smoke.py'", notebook)
        self.assertGreaterEqual(notebook.count("check=True"), 12)
        ordered = [
            "colab_a100_run1.events.jsonl", "--conditions','standard_lora",
            "colab_a100_repeat_full_sr.events.jsonl", "--root',repeat,'--conditions','full_sr", "--compare-repeat',repeat",
            "freeze_stage2_environment.py','--protocol", "freeze_stage2_environment.py','--verify-only",
            "package_stage2_evidence.py','--repo-root", "verify_stage2_evidence.py','--archive",
        ]
        positions = [notebook.index(value) for value in ordered]
        self.assertEqual(positions, sorted(positions))
        result = subprocess.run(
            [sys.executable, "freeze_stage2_environment.py", "--verify-only", "--output-dir", "missing", "--repo-root", "."],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("only --output-dir", result.stderr)

    def test_freeze_bundle_is_canonical_targeted_but_outside_canonical_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol, archive, expectations = self._execution_inputs(tmp)
            smoke = self._a100_smoke(tmp)
            freeze = Path(tmp) / "ties_results" / "stage2_smoke" / "freeze_bundle"
            canonical = Path(tmp) / "ties_results" / "canonical_v1"

            with patch("canonical.freeze.compare_a100_repeat", self._validated_a100):
                result = build_freeze_bundle(
                    protocol, smoke, freeze, repo,
                    source_archive_path=archive, expectations_path=expectations,
                    commands_path=smoke / "commands.json",
                    backend_factory=CanonicalManifestBackend, environment_probe=self._environment_probe,
                )

            self.assertEqual("canonical_v1", result["target_schema"])
            self.assertFalse(canonical.exists())
            manifest = json.loads((freeze / "manifests" / "data_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("canonical_v1", manifest["scope"])
            self.assertEqual("pass", verify_freeze_bundle(freeze)["state"])

    def test_freeze_accepts_a_zero_argument_fixture_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol, archive, expectations = self._execution_inputs(tmp)
            smoke = self._a100_smoke(tmp)
            with patch("canonical.freeze.compare_a100_repeat", self._validated_a100):
                result = build_freeze_bundle(
                    protocol, smoke, Path(tmp) / "freeze", repo,
                    source_archive_path=archive, expectations_path=expectations,
                    commands_path=smoke / "commands.json",
                    backend_factory=ZeroArgumentCanonicalManifestBackend, environment_probe=self._environment_probe,
                )
            self.assertEqual("pass", result["state"])

    def test_freeze_inventory_is_last_nonrecursive_and_detects_extra_hash_unsafe_and_nonfinite_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol, archive, expectations = self._execution_inputs(tmp)
            smoke = self._a100_smoke(tmp)
            freeze = Path(tmp) / "freeze"
            with patch("canonical.freeze.compare_a100_repeat", self._validated_a100):
                build_freeze_bundle(protocol, smoke, freeze, repo, source_archive_path=archive, expectations_path=expectations,
                                    commands_path=smoke / "commands.json", backend_factory=CanonicalManifestBackend, environment_probe=self._environment_probe)
            inventory = json.loads((freeze / "checksum_inventory.json").read_text(encoding="utf-8"))
            self.assertNotIn("checksum_inventory.json", inventory["files"])
            for relative, expected in inventory["files"].items():
                self.assertEqual(expected, sha256_file(freeze / relative))

            (freeze / "unexpected.txt").write_text("extra\n", encoding="utf-8")
            _write_inventory(freeze)
            with self.assertRaisesRegex(ValueError, "extra"):
                verify_freeze_bundle(freeze)
            (freeze / "unexpected.txt").unlink()
            _write_inventory(freeze)
            protocol_snapshot = freeze / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md"
            protocol_snapshot.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_freeze_bundle(freeze)

    def test_verify_freeze_rejects_semantic_tamper_even_after_inventory_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol, archive, expectations = self._execution_inputs(tmp)
            smoke = self._a100_smoke(tmp)
            freeze = Path(tmp) / "freeze"
            with patch("canonical.freeze.compare_a100_repeat", self._validated_a100):
                build_freeze_bundle(protocol, smoke, freeze, repo, source_archive_path=archive,
                                    expectations_path=expectations, commands_path=smoke / "commands.json",
                                    backend_factory=CanonicalManifestBackend, environment_probe=self._environment_probe)
            data_path = freeze / "manifests" / "data_manifest.json"
            original_data = data_path.read_bytes()
            data = json.loads(original_data)
            data["data_seed"] = 7
            write_json(data_path, data)
            _write_inventory(freeze)
            with self.assertRaisesRegex(ValueError, "seeds"):
                verify_freeze_bundle(freeze)

            data_path.write_bytes(original_data)
            data = json.loads(original_data)
            content = data["hans"]["content_integrity"]
            content["partitions"]["evaluation"]["source_id_content_joint_checksum"] = "0" * 64
            write_json(data_path, data)
            _write_inventory(freeze)
            with self.assertRaisesRegex(ValueError, "HANS content"):
                verify_freeze_bundle(freeze)

            data_path.write_bytes(original_data)
            for invalid_id in (
                "ex0",
                "hans_train::ex2",
                "hans_evaluation::hans_evaluation::ex0",
                "hans_evaluation::ex00",
                "hans_evaluation::ex-1",
                "hans_evaluation::ex1::extra",
            ):
                with self.subTest(invalid_id=invalid_id):
                    data = json.loads(original_data)
                    entry = data["hans"]["evaluation"]
                    invalid_ids = [invalid_id, entry["full_ids"][1]]
                    entry["full_ids"] = entry["selected_ids"] = invalid_ids
                    import hashlib
                    digest = hashlib.sha256(
                        json.dumps(
                            invalid_ids,
                            ensure_ascii=False,
                            allow_nan=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    entry["full_ids_sha256"] = entry["selected_ids_sha256"] = digest
                    write_json(data_path, data)
                    _write_inventory(freeze)
                    with self.assertRaisesRegex(ValueError, "HANS manifest"):
                        verify_freeze_bundle(freeze)

            data_path.write_bytes(original_data)
            _write_inventory(freeze)
            frozen_expectations = freeze / "source_expectations.json"
            values = json.loads(frozen_expectations.read_text(encoding="utf-8"))
            values["execution_commit"] = "0" * 40
            write_json(frozen_expectations, values)
            _write_inventory(freeze)
            with self.assertRaisesRegex(ValueError, "expectations|provenance"):
                verify_freeze_bundle(freeze)

    def test_verify_freeze_rejects_rehashed_hans_anchor_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol, archive, expectations = self._execution_inputs(tmp)
            smoke = self._a100_smoke(tmp)
            freeze = Path(tmp) / "freeze-anchor"
            with patch("canonical.freeze.compare_a100_repeat", self._validated_a100):
                build_freeze_bundle(
                    protocol, smoke, freeze, repo,
                    source_archive_path=archive,
                    expectations_path=expectations,
                    commands_path=smoke / "commands.json",
                    backend_factory=CanonicalManifestBackend,
                    environment_probe=self._environment_probe,
                )
            data_path = freeze / "manifests" / "data_manifest.json"
            original = data_path.read_bytes()

            data = json.loads(original)
            hans = data["hans"]
            for key in (
                "full_ids", "selected_ids", "full_ids_sha256", "selected_ids_sha256"
            ):
                hans["build"][key], hans["dev"][key] = hans["dev"][key], hans["build"][key]
            hans["content_integrity"]["partitions"]["build"], hans["content_integrity"]["partitions"]["dev"] = (
                hans["content_integrity"]["partitions"]["dev"],
                hans["content_integrity"]["partitions"]["build"],
            )
            split = hans["split_integrity"]
            split["build_source_pair_ids"], split["dev_source_pair_ids"] = (
                split["dev_source_pair_ids"], split["build_source_pair_ids"]
            )
            split["split_checksum"] = _digest(
                {
                    "schema_version": "hans_split_v1",
                    "hans_split_seed": 42,
                    "build_pair_ids": split["build_source_pair_ids"],
                    "dev_pair_ids": split["dev_source_pair_ids"],
                    "small_strata": [],
                }
            )
            write_json(data_path, data)
            _write_inventory(freeze)
            with self.assertRaisesRegex(ValueError, "official.*split.*anchor"):
                verify_freeze_bundle(freeze)

            data_path.write_bytes(original)
            data = json.loads(original)
            entry = data["hans"]["content_integrity"]["partitions"]["evaluation"]
            entry["content_sha256"] = list(reversed(entry["content_sha256"]))
            entry["content_sha256_ordered_checksum"] = _digest(entry["content_sha256"])
            entry["source_id_content_joint_checksum"] = _digest(
                list(zip(data["hans"]["evaluation"]["full_ids"], entry["content_sha256"]))
            )
            write_json(data_path, data)
            _write_inventory(freeze)
            with self.assertRaisesRegex(ValueError, "official.*content.*anchor"):
                verify_freeze_bundle(freeze)

            data_path.write_bytes(original)
            data = json.loads(original)
            selection = data["hans"]["selection_integrity"]
            selection["cap"] = 2
            selection["selected_order"] = "ranked_cap_order"
            payload = {
                key: value
                for key, value in selection.items()
                if key != "integrity_checksum"
            }
            selection["integrity_checksum"] = _digest(
                [[key, payload[key]] for key in sorted(payload)]
            )
            write_json(data_path, data)
            _write_inventory(freeze)
            with self.assertRaisesRegex(ValueError, "HANS selection"):
                verify_freeze_bundle(freeze)

    def test_freeze_refuses_unvalidated_a100_evidence_canonical_paths_and_nonempty_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol, archive, expectations = self._execution_inputs(tmp)
            smoke = self._a100_smoke(tmp)
            write_json(smoke / "stage2_validation.json", {"state": "fail"})
            with self.assertRaisesRegex(ValueError, "validated"):
                build_freeze_bundle(protocol, smoke, Path(tmp) / "freeze", repo, source_archive_path=archive, expectations_path=expectations,
                                    commands_path=smoke / "commands.json", backend_factory=CanonicalManifestBackend, environment_probe=self._environment_probe)
            with self.assertRaisesRegex(ValueError, "canonical_v1"):
                build_freeze_bundle(protocol, smoke, Path(tmp) / "canonical_v1" / "freeze", repo,
                                    source_archive_path=archive, expectations_path=expectations, commands_path=smoke / "commands.json",
                                    backend_factory=CanonicalManifestBackend)
            write_json(
                smoke / "stage2_validation.json",
                {"state": "pass", "repeat_comparison": self._validated_a100(smoke, smoke.parent / "colab_a100_repeat_full_sr")},
            )
            occupied = Path(tmp) / "occupied"
            occupied.mkdir()
            (occupied / "old.txt").write_text("old\n", encoding="utf-8")
            with patch("canonical.freeze.compare_a100_repeat", self._validated_a100):
                with self.assertRaisesRegex(ValueError, "new or empty"):
                    build_freeze_bundle(protocol, smoke, occupied, repo, source_archive_path=archive, expectations_path=expectations,
                                        commands_path=smoke / "commands.json", backend_factory=CanonicalManifestBackend, environment_probe=self._environment_probe)

    def test_freeze_binds_requested_protocol_and_commands_to_validated_a100_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol, archive, expectations = self._execution_inputs(tmp)
            smoke = self._a100_smoke(tmp)
            other_protocol = Path(tmp) / "other_protocol.md"
            other_protocol.write_text("# other\n", encoding="utf-8")
            with patch("canonical.freeze.compare_a100_repeat", self._validated_a100):
                with self.assertRaisesRegex(ValueError, "protocol"):
                    build_freeze_bundle(other_protocol, smoke, Path(tmp) / "bad-protocol", repo,
                                        source_archive_path=archive, expectations_path=expectations, commands_path=smoke / "commands.json",
                                        backend_factory=CanonicalManifestBackend, environment_probe=self._environment_probe)
                copied_commands = Path(tmp) / "copied_commands.json"
                copied_commands.write_bytes((smoke / "commands.json").read_bytes())
                with self.assertRaisesRegex(ValueError, "commands"):
                    build_freeze_bundle(protocol, smoke, Path(tmp) / "bad-commands", repo,
                                        source_archive_path=archive, expectations_path=expectations, commands_path=copied_commands,
                                        backend_factory=CanonicalManifestBackend, environment_probe=self._environment_probe)
                bad_expectations = Path(tmp) / "bad-expectations.json"
                values = json.loads(expectations.read_text(encoding="utf-8"))
                values["archive_sha256"] = "0" * 64
                write_json(bad_expectations, values)
                with self.assertRaisesRegex(ValueError, "expectations"):
                    build_freeze_bundle(protocol, smoke, Path(tmp) / "bad-expectations", repo,
                                        source_archive_path=archive, expectations_path=bad_expectations,
                                        commands_path=smoke / "commands.json", backend_factory=CanonicalManifestBackend,
                                        environment_probe=self._environment_probe)
                with self.assertRaisesRegex(ValueError, "live environment"):
                    build_freeze_bundle(protocol, smoke, Path(tmp) / "bad-probe", repo,
                                        source_archive_path=archive, expectations_path=expectations,
                                        commands_path=smoke / "commands.json", backend_factory=CanonicalManifestBackend,
                                        environment_probe=lambda: {**self._environment_probe(), "gpu": "NVIDIA L4"})
                origin_repo = Path(tmp) / "repo"
                with self.assertRaisesRegex(ValueError, "source archive commit"):
                    build_freeze_bundle(protocol, smoke, Path(tmp) / "wrong-current-head", origin_repo,
                                        source_archive_path=archive, expectations_path=expectations,
                                        commands_path=smoke / "commands.json", backend_factory=CanonicalManifestBackend,
                                        environment_probe=self._environment_probe)

    def test_freeze_rejects_fully_rehashed_stored_environment_spoof_against_live_probe(self):
        from tests.test_stage2_validation import _rehash_status

        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol, archive, expectations = self._execution_inputs(tmp)
            primary, repeat = self._production_smoke_pair(repo)
            repeat_environment_path = repeat / "manifests" / "environment_manifest.json"
            original_repeat_environment = repeat_environment_path.read_bytes()
            semantic_environment = json.loads(original_repeat_environment)
            repeat_environment_path.write_text(json.dumps(semantic_environment, separators=(",", ":")) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "repeat environment"):
                build_freeze_bundle(
                    protocol, primary, Path(tmp) / "byte-mismatch-freeze", repo,
                    source_archive_path=archive, expectations_path=expectations,
                    commands_path=primary / "commands.json", repeat_root=repeat,
                    repeat_commands_path=repeat / "commands.json",
                    backend_factory=CanonicalManifestBackend, environment_probe=self._environment_probe,
                )
            repeat_environment_path.write_bytes(original_repeat_environment)
            spoof = {
                "schema_version": "canonical_environment_manifest_v1", "gpu": "NVIDIA A100-SXM4-40GB",
                "torch_gpu": "NVIDIA A100-SXM4-40GB", "nvidia_smi_gpu": "NVIDIA A100-SXM4-40GB",
                "python": "3.12.9", "python_implementation": "CPython", "platform": "Linux-spoof",
                "cuda_runtime": "99.0", "cuda_driver": "999.0",
                "packages": {"torch": "2.11.0", "transformers": "9.0", "datasets": "9.0", "numpy": "9.0"},
                "pip_freeze": ["datasets==9.0", "numpy==9.0", "torch==2.11.0", "transformers==9.0"],
            }
            for root in (primary, repeat):
                environment_path = root / "manifests" / "environment_manifest.json"
                write_json(environment_path, spoof)
                environment_hash = sha256_file(environment_path)
                for manifest_path in (root / "seed_42").glob("*/run_manifest.json"):
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["environment_manifest_sha256"] = environment_hash
                    write_json(manifest_path, manifest)
                    _rehash_status(manifest_path.parent)
            with self.assertRaisesRegex(ValueError, "live environment"):
                build_freeze_bundle(
                    protocol, primary, Path(tmp) / "spoof-freeze", repo,
                    source_archive_path=archive, expectations_path=expectations,
                    commands_path=primary / "commands.json", repeat_root=repeat,
                    repeat_commands_path=repeat / "commands.json",
                    backend_factory=CanonicalManifestBackend, environment_probe=self._environment_probe,
                )

    def test_freeze_rejects_all_six_rehashed_manifest_command_and_dirty_spoofs(self):
        from tests.test_stage2_validation import _rehash_status

        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol, archive, expectations = self._execution_inputs(tmp)
            primary, repeat = self._production_smoke_pair(repo)
            manifests = list((primary / "seed_42").glob("*/run_manifest.json")) + list((repeat / "seed_42").glob("*/run_manifest.json"))
            self.assertEqual(6, len(manifests))
            for manifest_path in manifests:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["command"] = ["different_program"]
                write_json(manifest_path, manifest)
                _rehash_status(manifest_path.parent)
            with self.assertRaisesRegex(ValueError, "command"):
                build_freeze_bundle(
                    protocol, primary, Path(tmp) / "command-freeze", repo,
                    source_archive_path=archive, expectations_path=expectations,
                    commands_path=primary / "commands.json", repeat_root=repeat,
                    repeat_commands_path=repeat / "commands.json",
                    backend_factory=CanonicalManifestBackend, environment_probe=self._environment_probe,
                )

            for root in (primary, repeat):
                argv = json.loads((root / "commands.json").read_text(encoding="utf-8"))["argv"]
                for manifest_path in (root / "seed_42").glob("*/run_manifest.json"):
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["command"] = argv
                    manifest["git"]["dirty"] = True
                    write_json(manifest_path, manifest)
                    _rehash_status(manifest_path.parent)
            with self.assertRaisesRegex(ValueError, "clean|dirty"):
                build_freeze_bundle(
                    protocol, primary, Path(tmp) / "dirty-freeze", repo,
                    source_archive_path=archive, expectations_path=expectations,
                    commands_path=primary / "commands.json", repeat_root=repeat,
                    repeat_commands_path=repeat / "commands.json",
                    backend_factory=CanonicalManifestBackend, environment_probe=self._environment_probe,
                )
    def test_dependency_light_cli_help(self):
        for script in ("package_stage2_source.py", "freeze_stage2_environment.py", "package_stage2_evidence.py", "verify_stage2_evidence.py"):
            result = subprocess.run([sys.executable, script, "--help"], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(0, result.returncode, result.stderr)

    def test_evidence_archive_requires_external_expectations_before_reading_runtime_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "ties_results" / "stage2_smoke" / "colab_a100_run1"
            repeat = root / "ties_results" / "stage2_smoke" / "colab_a100_repeat_full_sr"
            freeze = root / "ties_results" / "stage2_smoke" / "freeze_bundle"
            for directory in (primary / "manifests", repeat / "manifests", freeze):
                directory.mkdir(parents=True, exist_ok=True)
            for path in (
                primary / "commands.json", primary / "stage2_validation.json",
                primary / "manifests" / "environment_manifest.json", primary / "metrics.json",
                repeat / "commands.json", repeat / "manifests" / "environment_manifest.json",
                freeze / "checksum_inventory.json",
            ):
                path.write_text("{}\n", encoding="utf-8")
            monitors = root / "ties_results" / ".stage2_monitor"
            monitors.mkdir(parents=True)
            for name in ("colab_a100_run1.events.jsonl", "colab_a100_repeat_full_sr.events.jsonl"):
                (monitors / name).write_text("{}\n", encoding="utf-8")
            (primary / "checkpoint.pt").write_bytes(b"weights")
            archive = root / "evidence.zip"

            with self.assertRaisesRegex(ValueError, "expectations"):
                build_evidence_archive(root, archive)
            self.assertFalse(archive.exists())

    def test_evidence_archive_is_derived_from_validated_outputs_and_exact_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol, source_archive, expectations = self._execution_inputs(tmp)
            primary, repeat = self._production_smoke_pair(repo)
            freeze = repo / "ties_results" / "stage2_smoke" / "freeze_bundle"
            build_freeze_bundle(
                protocol, primary, freeze, repo,
                source_archive_path=source_archive, expectations_path=expectations,
                commands_path=primary / "commands.json", repeat_root=repeat,
                repeat_commands_path=repeat / "commands.json",
                backend_factory=CanonicalManifestBackend, environment_probe=self._environment_probe,
            )
            monitors = repo / "ties_results" / ".stage2_monitor"
            monitors.mkdir(parents=True)
            self._write_monitor(monitors / "colab_a100_run1.events.jsonl", json.loads((primary / "commands.json").read_text(encoding="utf-8"))["argv"], repo)
            self._write_monitor(monitors / "colab_a100_repeat_full_sr.events.jsonl", json.loads((repeat / "commands.json").read_text(encoding="utf-8"))["argv"], repo)
            (primary / "source_transport.zip").write_bytes(b"not source evidence")

            evidence = Path(tmp) / "stage2_a100_evidence.zip"
            result = build_evidence_archive(repo, evidence, expectations_path=expectations)
            self.assertEqual("pass", result["state"])
            with zipfile.ZipFile(evidence) as archive:
                names = archive.namelist()
                inventory_name = "ties_results/stage2_smoke/stage2_evidence_inventory.json"
                inventory = json.loads(archive.read(inventory_name))
            self.assertEqual(set(names), set(inventory["files"]) | {inventory_name})
            self.assertEqual("stage2_evidence_inventory_v2", inventory["schema_version"])
            self.assertEqual(2, len(inventory["omitted_weights"]))
            self.assertTrue(any(name.endswith("source_expectations.json") for name in names))
            self.assertFalse(any(Path(name).suffix.casefold() in {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".zip"} for name in names))

            extra = primary / "unexpected.json"
            extra.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                build_evidence_archive(repo, Path(tmp) / "extra.zip", expectations_path=expectations)

            extra.unlink()
            for index, suffix in enumerate((".pt", ".pth", ".ckpt", ".bin", ".safetensors")):
                untracked_weight = primary / f"untracked{suffix}"
                untracked_weight.write_bytes(b"untracked weight")
                try:
                    with self.assertRaisesRegex(ValueError, "untracked model weight"):
                        build_evidence_archive(repo, Path(tmp) / f"weight-{index}.zip", expectations_path=expectations)
                finally:
                    untracked_weight.unlink()
            missing_members = (
                primary / "seed_42" / "full_sr" / "metrics.json",
                primary / "seed_42" / "full_sr" / "hans_predictions.jsonl",
                primary / "seed_42" / "full_sr" / "stdout.log",
                primary / "seed_42" / "full_sr" / "config.json",
                primary / "stage2_validation.md",
                monitors / "colab_a100_repeat_full_sr.events.jsonl",
                freeze / "source_origin_commit.txt",
            )
            for index, member in enumerate(missing_members):
                with self.subTest(missing=member.name):
                    payload = member.read_bytes()
                    member.unlink()
                    try:
                        with self.assertRaises((ValueError, FileNotFoundError)):
                            build_evidence_archive(repo, Path(tmp) / f"missing-{index}.zip", expectations_path=expectations)
                    finally:
                        member.write_bytes(payload)

            bad_expectations = Path(tmp) / "bad-evidence-expectations.json"
            sidecar = json.loads(expectations.read_text(encoding="utf-8"))
            sidecar["origin_commit"] = "0" * 40
            write_json(bad_expectations, sidecar)
            with self.assertRaisesRegex(ValueError, "expectations mismatch"):
                build_evidence_archive(repo, Path(tmp) / "expectations-mismatch.zip", expectations_path=bad_expectations)

    def test_evidence_transport_verifies_and_extracts_fresh_without_model_weights(self):
        from canonical.evidence_transport import verify_evidence_archive

        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol, source_archive, expectations = self._execution_inputs(tmp)
            primary, repeat = self._production_smoke_pair(repo)
            freeze = repo / "ties_results" / "stage2_smoke" / "freeze_bundle"
            build_freeze_bundle(
                protocol, primary, freeze, repo,
                source_archive_path=source_archive, expectations_path=expectations,
                commands_path=primary / "commands.json", repeat_root=repeat,
                repeat_commands_path=repeat / "commands.json",
                backend_factory=CanonicalManifestBackend, environment_probe=self._environment_probe,
            )
            monitors = repo / "ties_results" / ".stage2_monitor"
            self._write_monitor(monitors / "colab_a100_run1.events.jsonl", json.loads((primary / "commands.json").read_text(encoding="utf-8"))["argv"], repo)
            self._write_monitor(monitors / "colab_a100_repeat_full_sr.events.jsonl", json.loads((repeat / "commands.json").read_text(encoding="utf-8"))["argv"], repo)
            archive = Path(tmp) / "evidence.zip"
            build_evidence_archive(repo, archive, expectations_path=expectations)

            extracted = Path(tmp) / "fresh-extract"
            result = verify_evidence_archive(archive, extract_dir=extracted)

            self.assertEqual("pass", result["state"])
            self.assertTrue((extracted / "ties_results" / "stage2_smoke" / "stage2_evidence_inventory.json").is_file())
            self.assertFalse(any(path.suffix.casefold() in {".pt", ".pth", ".ckpt", ".bin", ".safetensors"} for path in extracted.rglob("*")))
            with self.assertRaisesRegex(ValueError, "overwrite"):
                verify_evidence_archive(archive, extract_dir=extracted)

    def test_evidence_transport_rejects_semantic_tamper_after_inventory_rebuild(self):
        from canonical.evidence_transport import INVENTORY_MEMBER, verify_evidence_archive
        from hashlib import sha256

        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol, source_archive, expectations = self._execution_inputs(tmp)
            primary, repeat = self._production_smoke_pair(repo)
            freeze = repo / "ties_results" / "stage2_smoke" / "freeze_bundle"
            build_freeze_bundle(
                protocol, primary, freeze, repo,
                source_archive_path=source_archive, expectations_path=expectations,
                commands_path=primary / "commands.json", repeat_root=repeat,
                repeat_commands_path=repeat / "commands.json",
                backend_factory=CanonicalManifestBackend, environment_probe=self._environment_probe,
            )
            monitors = repo / "ties_results" / ".stage2_monitor"
            self._write_monitor(monitors / "colab_a100_run1.events.jsonl", json.loads((primary / "commands.json").read_text(encoding="utf-8"))["argv"], repo)
            self._write_monitor(monitors / "colab_a100_repeat_full_sr.events.jsonl", json.loads((repeat / "commands.json").read_text(encoding="utf-8"))["argv"], repo)
            source = Path(tmp) / "evidence.zip"
            build_evidence_archive(repo, source, expectations_path=expectations)

            def inventory(members):
                return json.loads(members[INVENTORY_MEMBER])

            def save(members, value):
                members[INVENTORY_MEMBER] = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()

            def replace_inventoried(members, name, payload):
                members[name] = payload
                value = inventory(members); value["files"][name] = sha256(payload).hexdigest(); save(members, value)

            def add_extra(members):
                name = "ties_results/stage2_smoke/colab_a100_run1/unexpected.json"
                members[name] = b"{}\n"
                value = inventory(members); value["files"][name] = sha256(members[name]).hexdigest(); save(members, value)

            def remove_monitor(members):
                name = "ties_results/.stage2_monitor/colab_a100_repeat_full_sr.events.jsonl"
                del members[name]
                value = inventory(members); del value["files"][name]; save(members, value)

            def include_weight(members):
                value = inventory(members); entry = value["omitted_weights"][0]
                members[entry["path"]] = b"transported weight"
                value["files"][entry["path"]] = sha256(members[entry["path"]]).hexdigest(); save(members, value)

            def spoof_omitted(members):
                value = inventory(members); value["omitted_weights"][0]["sha256"] = "0" * 64; save(members, value)

            def spoof_expectations(members):
                name = "ties_results/stage2_smoke/source_expectations.json"
                value = json.loads(members[name]); value["origin_commit"] = "0" * 40
                members[name] = (json.dumps(value, sort_keys=True) + "\n").encode()
                inv = inventory(members); inv["files"][name] = sha256(members[name]).hexdigest(); save(members, inv)

            def add_freeze_extra(members):
                name = "ties_results/stage2_smoke/freeze_bundle/unexpected.txt"
                members[name] = b"extra\n"
                value = inventory(members); value["files"][name] = sha256(members[name]).hexdigest(); save(members, value)

            def drop_standard_metrics_and_status_key(members):
                metrics = "ties_results/stage2_smoke/colab_a100_run1/seed_42/standard_lora/metrics.json"
                status_name = "ties_results/stage2_smoke/colab_a100_run1/seed_42/standard_lora/status.json"
                status = json.loads(members[status_name]); status["output_hashes"].pop("metrics.json")
                del members[metrics]
                value = inventory(members); del value["files"][metrics]
                members[status_name] = (json.dumps(status, sort_keys=True) + "\n").encode()
                value["files"][status_name] = sha256(members[status_name]).hexdigest(); save(members, value)

            def invalidate_hans_and_rehash(members):
                predictions = "ties_results/stage2_smoke/colab_a100_run1/seed_42/full_sr/hans_predictions.jsonl"
                status_name = "ties_results/stage2_smoke/colab_a100_run1/seed_42/full_sr/status.json"
                rows = [json.loads(line) for line in members[predictions].decode().splitlines()]
                rows[0]["gold_label"] = "invalid-label"
                payload = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
                status = json.loads(members[status_name]); status["output_hashes"]["hans_predictions.jsonl"] = sha256(payload).hexdigest()
                members[predictions] = payload
                members[status_name] = (json.dumps(status, sort_keys=True) + "\n").encode()
                value = inventory(members); value["files"][predictions] = sha256(payload).hexdigest(); value["files"][status_name] = sha256(members[status_name]).hexdigest(); save(members, value)

            def spoof_branch_shared_hash_and_rehash(members):
                manifest_name = "ties_results/stage2_smoke/colab_a100_run1/seed_42/full_sr/run_manifest.json"
                status_name = "ties_results/stage2_smoke/colab_a100_run1/seed_42/full_sr/status.json"
                manifest = json.loads(members[manifest_name]); manifest["shared_phase2_checkpoint"]["sha256"] = "0" * 64
                members[manifest_name] = (json.dumps(manifest, sort_keys=True) + "\n").encode()
                status = json.loads(members[status_name]); status["output_hashes"]["run_manifest.json"] = sha256(members[manifest_name]).hexdigest()
                members[status_name] = (json.dumps(status, sort_keys=True) + "\n").encode()
                value = inventory(members); value["files"][manifest_name] = sha256(members[manifest_name]).hexdigest(); value["files"][status_name] = sha256(members[status_name]).hexdigest(); save(members, value)

            def fabricate_omitted_method_weight(members):
                status_name = "ties_results/stage2_smoke/colab_a100_run1/seed_42/full_sr/status.json"
                fabricated = "ties_results/stage2_smoke/colab_a100_run1/seed_42/full_sr/fabricated.pt"
                digest = "f" * 64
                status = json.loads(members[status_name]); status["output_hashes"]["fabricated.pt"] = digest
                members[status_name] = (json.dumps(status, sort_keys=True) + "\n").encode()
                value = inventory(members); value["files"][status_name] = sha256(members[status_name]).hexdigest()
                value["omitted_weights"].append({"path": fabricated, "sha256": digest, "reason": "model_weight_excluded"})
                value["omitted_weights"].sort(key=lambda item: item["path"]); save(members, value)

            def tamper_stored_validation_semantics(members):
                name = "ties_results/stage2_smoke/colab_a100_repeat_full_sr/stage2_validation.json"
                report = json.loads(members[name]); report["checks"]["hans_recomputation"]["state"] = "tampered"
                replace_inventoried(members, name, (json.dumps(report, sort_keys=True) + "\n").encode())

            transforms = (
                add_extra, remove_monitor, include_weight, spoof_omitted, spoof_expectations, add_freeze_extra,
                drop_standard_metrics_and_status_key, invalidate_hans_and_rehash,
                spoof_branch_shared_hash_and_rehash, fabricate_omitted_method_weight,
                tamper_stored_validation_semantics,
            )
            for index, transform in enumerate(transforms):
                with self.subTest(transform=transform.__name__):
                    tampered = Path(tmp) / f"tampered-{index}.zip"
                    self._rewrite_evidence(source, tampered, transform)
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        verify_evidence_archive(tampered)

            symlink_archive = Path(tmp) / "symlink.zip"
            with zipfile.ZipFile(source) as archive, zipfile.ZipFile(symlink_archive, "w") as target:
                for info in archive.infolist():
                    target.writestr(info.filename, archive.read(info.filename))
                link = zipfile.ZipInfo("ties_results/stage2_smoke/link")
                link.create_system = 3
                link.external_attr = (0o120777 << 16)
                target.writestr(link, "target")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                verify_evidence_archive(symlink_archive)

    def test_monitor_transport_schema_rejects_failure_policy_and_command_spoofs(self):
        from canonical.evidence_transport import validate_monitor_evidence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = [sys.executable, str((ROOT / "run_stage2_smoke.py").resolve()), "--mode", "primary"]
            monitor = root / "monitor.jsonl"
            self._write_monitor(monitor, command, root)
            validate_monitor_evidence(monitor, expected_command=command, expected_cwd=root)

            cases = {
                "empty": "",
                "malformed": "{bad json}\n",
                "crashed": json.dumps({"event":"CRASHED","timestamp":0,"elapsed_seconds":0,"command":command,"cwd":str(root),"returncode":1}) + "\n",
                "hard-timeout": json.dumps({"event":"HARD_TIMEOUT","timestamp":0,"elapsed_seconds":0,"command":command,"cwd":str(root)}) + "\n",
                "fatal": json.dumps({"event":"FATAL_PATTERN","timestamp":0,"elapsed_seconds":0,"command":command,"cwd":str(root)}) + "\n",
            }
            for name, payload in cases.items():
                with self.subTest(name=name):
                    monitor.write_text(payload, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        validate_monitor_evidence(monitor, expected_command=command, expected_cwd=root)

            self._write_monitor(monitor, command, root)
            records = [json.loads(line) for line in monitor.read_text(encoding="utf-8").splitlines()]
            records[0]["policy"]["stall_seconds"] = 1
            monitor.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "policy"):
                validate_monitor_evidence(monitor, expected_command=command, expected_cwd=root)

            self._write_monitor(monitor, ["different_program"], root)
            with self.assertRaisesRegex(ValueError, "command"):
                validate_monitor_evidence(monitor, expected_command=command, expected_cwd=root)


if __name__ == "__main__":
    unittest.main()
