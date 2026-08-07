import json
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
from canonical.freeze import build_freeze_bundle, verify_freeze_bundle
from canonical.source_package import build_source_package, verify_source_package


class CanonicalManifestBackend:
    def __init__(self, config):
        self.config = config

    def initialize_manifests(self, output_dir, _protocol_path):
        manifests = Path(output_dir) / "manifests"
        write_json(
            manifests / "data_manifest.json",
            {
                "schema_version": "canonical_data_manifest_v2",
                "scope": "canonical_v1",
                "data_seed": 42,
                "mnli": {"train": {"selected_count": self.config.mnli_train_size}},
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
        protocol = repo / "FROZEN_EXPERIMENT_PROTOCOL.md"
        protocol.write_text("# frozen\n", encoding="utf-8")
        self._git(repo, "add", ".")
        self._git(repo, "commit", "-m", "fixture")
        return repo, protocol

    def _a100_smoke(self, directory):
        smoke = Path(directory) / "ties_results" / "stage2_smoke" / "colab_a100_run1"
        manifests = smoke / "manifests"
        write_json(
            manifests / "environment_manifest.json",
            {
                "schema_version": "canonical_environment_manifest_v1",
                "gpu": "NVIDIA A100-SXM4-40GB",
                "python": "3.12.0",
                "pip_freeze": ["datasets==4.0", "torch==2.11.0", "transformers==5.0"],
            },
        )
        write_json(
            smoke / "commands.json",
            {"schema_version": "stage2_smoke_commands_v1", "argv": ["python", "run_stage2_smoke.py"]},
        )
        write_json(
            smoke / "stage2_validation.json",
            {
                "schema_version": "stage2_validation_v1",
                "state": "pass",
                "repeat_comparison": {"state": "pass"},
            },
        )
        (smoke / "protocol_snapshot").mkdir()
        (smoke / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md").write_text(
            "# frozen\n", encoding="utf-8"
        )
        return smoke

    def _validated_a100(self, primary, _repeat):
        return {"schema_version": "stage2_a100_repeat_comparison_v1", "state": "pass"}

    def test_source_package_binds_clean_commit_protocol_and_verified_git_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol = self._source_repo(tmp)
            archive = Path(tmp) / "stage2_source.zip"

            metadata = build_source_package(repo, protocol, archive)

            self.assertFalse(metadata["git"]["dirty"])
            self.assertRegex(metadata["git"]["commit"], r"^[0-9a-f]{40}$")
            self.assertEqual(64, len(metadata["protocol_sha256"]))
            self.assertEqual(64, len(metadata["bundle_sha256"]))
            self.assertEqual(metadata, verify_source_package(archive, repo_root=repo))
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

    def test_freeze_bundle_is_canonical_targeted_but_outside_canonical_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol = self._source_repo(tmp)
            archive = Path(tmp) / "stage2_source.zip"
            build_source_package(repo, protocol, archive)
            smoke = self._a100_smoke(tmp)
            freeze = Path(tmp) / "ties_results" / "stage2_smoke" / "freeze_bundle"
            canonical = Path(tmp) / "ties_results" / "canonical_v1"

            with patch("canonical.freeze.compare_a100_repeat", self._validated_a100):
                result = build_freeze_bundle(
                    protocol, smoke, freeze, repo,
                    source_archive_path=archive,
                    commands_path=smoke / "commands.json",
                    backend_factory=CanonicalManifestBackend,
                )

            self.assertEqual("canonical_v1", result["target_schema"])
            self.assertFalse(canonical.exists())
            manifest = json.loads((freeze / "manifests" / "data_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("canonical_v1", manifest["scope"])
            self.assertEqual("pass", verify_freeze_bundle(freeze)["state"])

    def test_freeze_accepts_a_zero_argument_fixture_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol = self._source_repo(tmp)
            archive = Path(tmp) / "stage2_source.zip"
            build_source_package(repo, protocol, archive)
            smoke = self._a100_smoke(tmp)
            with patch("canonical.freeze.compare_a100_repeat", self._validated_a100):
                result = build_freeze_bundle(
                    protocol, smoke, Path(tmp) / "freeze", repo,
                    source_archive_path=archive,
                    commands_path=smoke / "commands.json",
                    backend_factory=ZeroArgumentCanonicalManifestBackend,
                )
            self.assertEqual("pass", result["state"])

    def test_freeze_inventory_is_last_nonrecursive_and_detects_extra_hash_unsafe_and_nonfinite_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol = self._source_repo(tmp)
            archive = Path(tmp) / "stage2_source.zip"
            build_source_package(repo, protocol, archive)
            smoke = self._a100_smoke(tmp)
            freeze = Path(tmp) / "freeze"
            with patch("canonical.freeze.compare_a100_repeat", self._validated_a100):
                build_freeze_bundle(protocol, smoke, freeze, repo, source_archive_path=archive,
                                    commands_path=smoke / "commands.json", backend_factory=CanonicalManifestBackend)
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

    def test_freeze_refuses_unvalidated_a100_evidence_canonical_paths_and_nonempty_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol = self._source_repo(tmp)
            archive = Path(tmp) / "stage2_source.zip"
            build_source_package(repo, protocol, archive)
            smoke = self._a100_smoke(tmp)
            write_json(smoke / "stage2_validation.json", {"state": "fail"})
            with self.assertRaisesRegex(ValueError, "validated"):
                build_freeze_bundle(protocol, smoke, Path(tmp) / "freeze", repo, source_archive_path=archive,
                                    commands_path=smoke / "commands.json", backend_factory=CanonicalManifestBackend)
            with self.assertRaisesRegex(ValueError, "canonical_v1"):
                build_freeze_bundle(protocol, smoke, Path(tmp) / "canonical_v1" / "freeze", repo,
                                    source_archive_path=archive, commands_path=smoke / "commands.json",
                                    backend_factory=CanonicalManifestBackend)
            write_json(
                smoke / "stage2_validation.json",
                {"state": "pass", "repeat_comparison": {"state": "pass"}},
            )
            occupied = Path(tmp) / "occupied"
            occupied.mkdir()
            (occupied / "old.txt").write_text("old\n", encoding="utf-8")
            with patch("canonical.freeze.compare_a100_repeat", self._validated_a100):
                with self.assertRaisesRegex(ValueError, "new or empty"):
                    build_freeze_bundle(protocol, smoke, occupied, repo, source_archive_path=archive,
                                        commands_path=smoke / "commands.json", backend_factory=CanonicalManifestBackend)

    def test_freeze_binds_requested_protocol_and_commands_to_validated_a100_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, protocol = self._source_repo(tmp)
            archive = Path(tmp) / "stage2_source.zip"
            build_source_package(repo, protocol, archive)
            smoke = self._a100_smoke(tmp)
            other_protocol = Path(tmp) / "other_protocol.md"
            other_protocol.write_text("# other\n", encoding="utf-8")
            with patch("canonical.freeze.compare_a100_repeat", self._validated_a100):
                with self.assertRaisesRegex(ValueError, "protocol"):
                    build_freeze_bundle(other_protocol, smoke, Path(tmp) / "bad-protocol", repo,
                                        source_archive_path=archive, commands_path=smoke / "commands.json",
                                        backend_factory=CanonicalManifestBackend)
                copied_commands = Path(tmp) / "copied_commands.json"
                copied_commands.write_bytes((smoke / "commands.json").read_bytes())
                with self.assertRaisesRegex(ValueError, "commands"):
                    build_freeze_bundle(protocol, smoke, Path(tmp) / "bad-commands", repo,
                                        source_archive_path=archive, commands_path=copied_commands,
                                        backend_factory=CanonicalManifestBackend)

    def test_dependency_light_cli_help(self):
        for script in ("package_stage2_source.py", "freeze_stage2_environment.py"):
            result = subprocess.run([sys.executable, script, "--help"], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
