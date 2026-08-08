import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.artifacts import sha256_file, write_json
from canonical.freeze import (
    _commands,
    _strict_data_manifest,
    _strict_environment,
    _write_inventory,
    build_evidence_archive,
    build_freeze_bundle,
    verify_freeze_bundle,
)
from canonical.source_package import _tracked_entries, build_source_package, verify_source_package
from canonical.stage2_validation import compare_a100_repeat


class CanonicalManifestBackend:
    def __init__(self, config):
        self.config = config

    def initialize_manifests(self, output_dir, _protocol_path):
        manifests = Path(output_dir) / "manifests"
        def entry(name, count):
            values = [f"{name}-{index}" for index in range(count)]
            payload = json.dumps(values, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            import hashlib
            digest = hashlib.sha256(payload).hexdigest()
            selected_limit = count if name.startswith("mnli-") else None
            return {"source": name, "split": "test", "id_strategy": "preferred_field_or_content_sha256", "preferred_id_fields": ["id"], "strata_fields": [], "selection_seed": 42, "selected_limit": selected_limit, "full_count": count, "selected_count": count, "full_ids": values, "selected_ids": values, "full_ids_sha256": digest, "selected_ids_sha256": digest}
        write_json(
            manifests / "data_manifest.json",
            {
                "schema_version": "canonical_data_manifest_v2",
                "scope": "canonical_v1",
                "data_seed": 42,
                "hans_split_seed": 42,
                "mnli": {"train": entry("mnli-train", 100000), "validation_matched": entry("mnli-validation", 5000)},
                "hans": {name: entry(f"hans-{name}", 1) for name in ("build", "dev", "evaluation")},
                "ood": {name: entry(f"ood-{name}", 1) for name in ("esnli", "anli", "snli_hard", "wanli")},
            },
        )


class ZeroArgumentCanonicalManifestBackend(CanonicalManifestBackend):
    def __init__(self):
        self.config = type("Config", (), {"mnli_train_size": 100_000})()


class Stage2FreezeTest(unittest.TestCase):
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
        protocol = repo / "FROZEN_EXPERIMENT_PROTOCOL.md"
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
        return repo, repo / "FROZEN_EXPERIMENT_PROTOCOL.md", archive, expectations

    def _a100_smoke(self, directory):
        smoke = Path(directory) / "ties_results" / "stage2_smoke" / "colab_a100_run1"
        manifests = smoke / "manifests"
        write_json(
            manifests / "environment_manifest.json",
            {
                "schema_version": "canonical_environment_manifest_v1",
                "gpu": "NVIDIA A100-SXM4-40GB",
                "python": "3.12.0",
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
            write_json(root / "commands.json", {"schema_version": "stage2_smoke_commands_v1", "mode": mode, "environment": "colab_a100", "argv": ["python", "run_stage2_smoke.py", "--mode", mode, "--environment", "colab_a100", "--protocol", "FROZEN_EXPERIMENT_PROTOCOL.md", "--output-dir", f"ties_results/stage2_smoke/{root.name}", "--fresh"], "expected_condition_tags": list(tags[1:]), "profile_name": "stage2_smoke_v1", "gpu_name": "NVIDIA A100-SXM4-40GB", "started_at": "2026-08-08T00:00:00+00:00"})
            for tag in tags:
                write_json(root / "seed_42" / tag / "run_manifest.json", {"git": {"commit": commit}})
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
            "python": "3.12.0", "platform": "Linux-test", "cuda_runtime": "12.8", "cuda_driver": "555.1",
            "packages": {"torch": "2.11.0", "transformers": "5.0", "datasets": "4.0", "numpy": "2.0"},
            "pip_freeze": ["datasets==4.0", "numpy==2.0", "torch==2.11.0", "transformers==5.0"],
        }
        for root, mode, tags in ((primary, "primary", ["standard_lora", "full_sr", "class_prior_reweight"]), (repeat, "repeat_full_sr", ["full_sr"])):
            protocol_snapshot = root / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md"
            protocol_snapshot.write_bytes((Path(repo) / "FROZEN_EXPERIMENT_PROTOCOL.md").read_bytes())
            protocol_hash = sha256_file(protocol_snapshot)
            (root / "protocol_snapshot" / "protocol_sha256.txt").write_text(protocol_hash + "\n", encoding="utf-8")
            write_json(root / "manifests" / "environment_manifest.json", environment)
            env_hash = sha256_file(root / "manifests" / "environment_manifest.json")
            write_json(root / "commands.json", {"schema_version":"stage2_smoke_commands_v1","mode":mode,"environment":"colab_a100","argv":["python","run_stage2_smoke.py","--mode",mode,"--environment","colab_a100","--protocol","FROZEN_EXPERIMENT_PROTOCOL.md","--output-dir",f"ties_results/stage2_smoke/{root.name}","--fresh"],"expected_condition_tags":tags,"profile_name":"stage2_smoke_v1","gpu_name":"NVIDIA A100-SXM4-40GB","started_at":"2026-08-08T00:00:00+00:00"})
            checkpoint = root / "seed_42" / "shared_phase2" / "checkpoints" / "shared.pt"
            metadata_path = root / "seed_42" / "shared_phase2" / "shared_checkpoint_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["checkpoint_path"] = str(checkpoint)
            write_json(metadata_path, metadata)
            checkpoint_ref = {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
            for manifest_path in (root / "seed_42").glob("*/run_manifest.json"):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["git"]["commit"] = commit
                manifest["protocol_sha256"] = protocol_hash
                manifest["environment_manifest_sha256"] = env_hash
                if manifest_path.parent.name not in {"shared_phase2", "standard_lora"}:
                    manifest["shared_phase2_checkpoint"] = checkpoint_ref
                write_json(manifest_path, manifest)
                _rehash_status(manifest_path.parent)
        comparison = compare_a100_repeat(primary, repeat, canonical_dir=Path(repo) / "ties_results" / "canonical_v1")
        write_json(primary / "stage2_validation.json", {"schema_version":"stage2_validation_v1","root":str(primary),"state":"pass","checks":{},"repeat_comparison":comparison})
        (primary / "stage2_validation.md").write_text("# Stage 2 Smoke Validation\n\nState: `pass`\n", encoding="utf-8")
        return primary, repeat

    @staticmethod
    def _fake_gpu_probe():
        return {"nvidia_smi": "NVIDIA A100-SXM4-40GB", "torch_gpu": "NVIDIA A100-SXM4-40GB", "torch_cuda": "12.8"}

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

    def test_notebook_and_cli_contracts_are_fail_fast_and_sidecar_bound(self):
        notebook = (ROOT / "notebooks" / "stage2_colab_a100_smoke.ipynb").read_text(encoding="utf-8")
        self.assertNotIn("!python", notebook)
        self.assertNotIn("from canonical.freeze import", notebook)
        self.assertNotIn("subprocess.check_output", notebook)
        self.assertIn("package_stage2_evidence.py", notebook)
        self.assertIn("source_manifest_sha256", notebook)
        self.assertIn("--source-expectations", notebook)
        self.assertIn("parents!=[head]", notebook)
        self.assertIn("actual!=manifest", notebook)
        self.assertIn("core.autocrlf=false", notebook)
        self.assertIn("torch.cuda.is_available", notebook)
        self.assertGreaterEqual(notebook.count("check=True"), 12)
        ordered = [
            "colab_a100_run1.events.jsonl", "--conditions','standard_lora",
            "colab_a100_repeat_full_sr.events.jsonl", "--compare-repeat',repeat",
            "freeze_stage2_environment.py','--protocol", "freeze_stage2_environment.py','--verify-only",
            "package_stage2_evidence.py",
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
                    backend_factory=CanonicalManifestBackend, gpu_probe=self._fake_gpu_probe,
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
                    backend_factory=ZeroArgumentCanonicalManifestBackend, gpu_probe=self._fake_gpu_probe,
                )
            self.assertEqual("pass", result["state"])

    def test_freeze_inventory_is_last_nonrecursive_and_detects_extra_hash_unsafe_and_nonfinite_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol, archive, expectations = self._execution_inputs(tmp)
            smoke = self._a100_smoke(tmp)
            freeze = Path(tmp) / "freeze"
            with patch("canonical.freeze.compare_a100_repeat", self._validated_a100):
                build_freeze_bundle(protocol, smoke, freeze, repo, source_archive_path=archive, expectations_path=expectations,
                                    commands_path=smoke / "commands.json", backend_factory=CanonicalManifestBackend, gpu_probe=self._fake_gpu_probe)
            inventory = json.loads((freeze / "checksum_inventory.json").read_text(encoding="utf-8"))
            self.assertNotIn("checksum_inventory.json", inventory["files"])
            for relative, expected in inventory["files"].items():
                self.assertEqual(expected, sha256_file(freeze / relative))

            (freeze / "unexpected.txt").write_text("extra\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "extra"):
                verify_freeze_bundle(freeze)
            (freeze / "unexpected.txt").unlink()
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
                                    backend_factory=CanonicalManifestBackend, gpu_probe=self._fake_gpu_probe)
            data_path = freeze / "manifests" / "data_manifest.json"
            original_data = data_path.read_bytes()
            data = json.loads(original_data)
            data["data_seed"] = 7
            write_json(data_path, data)
            _write_inventory(freeze)
            with self.assertRaisesRegex(ValueError, "seeds"):
                verify_freeze_bundle(freeze)

            data_path.write_bytes(original_data)
            frozen_expectations = freeze / "source_expectations.json"
            values = json.loads(frozen_expectations.read_text(encoding="utf-8"))
            values["execution_commit"] = "0" * 40
            write_json(frozen_expectations, values)
            _write_inventory(freeze)
            with self.assertRaisesRegex(ValueError, "expectations|provenance"):
                verify_freeze_bundle(freeze)

    def test_freeze_refuses_unvalidated_a100_evidence_canonical_paths_and_nonempty_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol, archive, expectations = self._execution_inputs(tmp)
            smoke = self._a100_smoke(tmp)
            write_json(smoke / "stage2_validation.json", {"state": "fail"})
            with self.assertRaisesRegex(ValueError, "validated"):
                build_freeze_bundle(protocol, smoke, Path(tmp) / "freeze", repo, source_archive_path=archive, expectations_path=expectations,
                                    commands_path=smoke / "commands.json", backend_factory=CanonicalManifestBackend, gpu_probe=self._fake_gpu_probe)
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
                                        commands_path=smoke / "commands.json", backend_factory=CanonicalManifestBackend, gpu_probe=self._fake_gpu_probe)

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
                                        backend_factory=CanonicalManifestBackend, gpu_probe=self._fake_gpu_probe)
                copied_commands = Path(tmp) / "copied_commands.json"
                copied_commands.write_bytes((smoke / "commands.json").read_bytes())
                with self.assertRaisesRegex(ValueError, "commands"):
                    build_freeze_bundle(protocol, smoke, Path(tmp) / "bad-commands", repo,
                                        source_archive_path=archive, expectations_path=expectations, commands_path=copied_commands,
                                        backend_factory=CanonicalManifestBackend, gpu_probe=self._fake_gpu_probe)
                bad_expectations = Path(tmp) / "bad-expectations.json"
                values = json.loads(expectations.read_text(encoding="utf-8"))
                values["archive_sha256"] = "0" * 64
                write_json(bad_expectations, values)
                with self.assertRaisesRegex(ValueError, "expectations"):
                    build_freeze_bundle(protocol, smoke, Path(tmp) / "bad-expectations", repo,
                                        source_archive_path=archive, expectations_path=bad_expectations,
                                        commands_path=smoke / "commands.json", backend_factory=CanonicalManifestBackend,
                                        gpu_probe=self._fake_gpu_probe)
                with self.assertRaisesRegex(ValueError, "live A100"):
                    build_freeze_bundle(protocol, smoke, Path(tmp) / "bad-probe", repo,
                                        source_archive_path=archive, expectations_path=expectations,
                                        commands_path=smoke / "commands.json", backend_factory=CanonicalManifestBackend,
                                        gpu_probe=lambda: {"nvidia_smi":"NVIDIA L4", "torch_gpu":"NVIDIA L4", "torch_cuda":"12.8"})
                origin_repo = Path(tmp) / "repo"
                with self.assertRaisesRegex(ValueError, "source archive commit"):
                    build_freeze_bundle(protocol, smoke, Path(tmp) / "wrong-current-head", origin_repo,
                                        source_archive_path=archive, expectations_path=expectations,
                                        commands_path=smoke / "commands.json", backend_factory=CanonicalManifestBackend,
                                        gpu_probe=self._fake_gpu_probe)

    def test_dependency_light_cli_help(self):
        for script in ("package_stage2_source.py", "freeze_stage2_environment.py", "package_stage2_evidence.py"):
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
                backend_factory=CanonicalManifestBackend, gpu_probe=self._fake_gpu_probe,
            )
            monitors = repo / "ties_results" / ".stage2_monitor"
            monitors.mkdir(parents=True)
            for name in ("colab_a100_run1.events.jsonl", "colab_a100_repeat_full_sr.events.jsonl"):
                (monitors / name).write_text('{"state":"pass"}\n', encoding="utf-8")
            for suffix in (".pt", ".pth", ".ckpt", ".bin", ".safetensors"):
                (primary / f"forbidden{suffix}").write_bytes(b"model weights")
            (primary / "source_transport.zip").write_bytes(b"not source evidence")

            evidence = Path(tmp) / "stage2_a100_evidence.zip"
            result = build_evidence_archive(repo, evidence, expectations_path=expectations)
            self.assertEqual("pass", result["state"])
            with zipfile.ZipFile(evidence) as archive:
                names = archive.namelist()
                inventory = json.loads(archive.read("evidence_checksum_inventory.json"))
            self.assertEqual(set(names), set(inventory["files"]) | {"evidence_checksum_inventory.json"})
            self.assertTrue(any(name.endswith("source_expectations.json") for name in names))
            self.assertFalse(any(Path(name).suffix.casefold() in {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".zip"} for name in names))

            extra = primary / "unexpected.json"
            extra.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                build_evidence_archive(repo, Path(tmp) / "extra.zip", expectations_path=expectations)

            extra.unlink()
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


if __name__ == "__main__":
    unittest.main()
