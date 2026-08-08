import json
import math
from datetime import datetime
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import monitor_stage2_job


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import canonical.monitoring as monitoring
from canonical.monitoring import MonitorPolicy, PRODUCTION_POLICY, monitor_command


def _task_section(plan: str, heading: str, next_heading: str) -> str:
    start = plan.index(f"### {heading}")
    end = plan.index(f"### {next_heading}", start)
    return plan[start:end]


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class FakeProcess:
    def __init__(self, *, exit_after_checks, exit_code=0, ignores_terminate=False):
        self.exit_after_checks = exit_after_checks
        self.exit_code = exit_code
        self.ignores_terminate = ignores_terminate
        self.checks = 0
        self.terminated = False
        self.killed = False
        self.argv = None
        self.cwd = None
        self.shell = None

    def poll(self):
        self.checks += 1
        if self.killed:
            return -9
        if self.terminated and not self.ignores_terminate:
            return -15
        if self.exit_after_checks is not None and self.checks >= self.exit_after_checks:
            return self.exit_code
        return None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class FrozenGraceProcess(FakeProcess):
    """Fails quickly if a monitor loops forever while its clock is frozen."""

    def poll(self):
        if self.checks >= 20:
            raise AssertionError("hard-timeout grace loop was not bounded")
        return super().poll()


class HalfSpeedClock(FakeClock):
    def sleep(self, seconds):
        self.value += min(seconds, 0.5)


class Stage2MonitoringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.log = self.root / "train.log"
        self.events = self.root / "monitor_events.jsonl"
        self.command = ["python", "run_stage2_smoke.py", "--mode", "primary"]

    def _records(self):
        return [json.loads(line) for line in self.events.read_text(encoding="utf-8").splitlines()]

    def _run(self, process, *, policy, watched_paths=None):
        clock = FakeClock()

        def factory(argv, *, cwd, shell):
            process.argv, process.cwd, process.shell = argv, cwd, shell
            return process

        result = monitor_command(
            self.command,
            cwd=self.root,
            events_path=self.events,
            watched_paths=watched_paths if watched_paths is not None else [self.log],
            policy=policy,
            clock=clock,
            sleep=clock.sleep,
            popen_factory=factory,
        )
        return result, clock

    def test_production_policy_is_frozen(self):
        self.assertEqual(PRODUCTION_POLICY, MonitorPolicy(300, 3600, 43200))
        process = FakeProcess(exit_after_checks=1)
        result, _ = self._run(process, policy=PRODUCTION_POLICY)
        self.assertEqual(0, result)
        self.assertEqual(
            {"check_interval_seconds": 300, "stall_seconds": 3600, "hard_timeout_seconds": 43200},
            self._records()[0]["policy"],
        )

    def test_production_monitor_validator_accepts_real_primary_and_repeat_command_records(self):
        from canonical.monitoring import validate_monitor_jsonl
        from run_stage2_smoke import _command_record

        for mode, tags in (
            ("primary", ("standard_lora", "full_sr", "class_prior_reweight")),
            ("repeat_full_sr", ("full_sr",)),
        ):
            with self.subTest(mode=mode):
                events = self.root / f"{mode}.jsonl"
                output = self.root / mode
                argv = [
                    sys.executable,
                    str((ROOT / "run_stage2_smoke.py").resolve()),
                    "--mode", mode,
                    "--environment", "colab_a100",
                    "--protocol", str((ROOT / "docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md").resolve()),
                    "--output-dir", str(output.resolve()),
                    "--fresh",
                ]
                command_record = _command_record(
                    mode=mode, environment="colab_a100", argv=argv,
                    condition_tags=tags, gpu_name="NVIDIA A100-SXM4-40GB",
                )
                process = FakeProcess(exit_after_checks=1)
                result = monitor_command(
                    command_record["argv"], cwd=ROOT, events_path=events,
                    watched_paths=[output], policy=PRODUCTION_POLICY,
                    clock=FakeClock(), sleep=lambda _seconds: None,
                    popen_factory=lambda *args, **kwargs: process,
                )
                self.assertEqual(0, result)
                report = validate_monitor_jsonl(
                    events, expected_command=command_record["argv"], expected_cwd=ROOT,
                )
                self.assertEqual("pass", report["state"])
                records = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(["STARTED", "STATUS_CHECK", "COMPLETED"], [record["event"] for record in records])
                self.assertTrue(all("T" in record["timestamp"] for record in records))

    def test_production_monitor_validator_rejects_missing_status_wrong_fields_nonfinite_and_command_mismatch(self):
        from canonical.monitoring import validate_monitor_jsonl

        process = FakeProcess(exit_after_checks=1)
        self._run(process, policy=PRODUCTION_POLICY)
        original = self.events.read_text(encoding="utf-8")
        records = [json.loads(line) for line in original.splitlines()]

        self.events.write_text("\n".join(json.dumps(record) for record in (records[0], records[-1])) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "STATUS_CHECK"):
            validate_monitor_jsonl(self.events, expected_command=self.command, expected_cwd=self.root)

        wrong = [dict(record) for record in records]
        wrong[1]["changes"] = wrong[1].pop("fingerprints")
        self.events.write_text("\n".join(json.dumps(record) for record in wrong) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_monitor_jsonl(self.events, expected_command=self.command, expected_cwd=self.root)

        self.events.write_text(original.replace('"elapsed_seconds": 0.0', '"elapsed_seconds": NaN', 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_monitor_jsonl(self.events, expected_command=self.command, expected_cwd=self.root)

        self.events.write_text(original, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "command"):
            validate_monitor_jsonl(self.events, expected_command=["different"], expected_cwd=self.root)

    def test_completion_mirrors_child_return_code_and_never_uses_shell(self):
        process = FakeProcess(exit_after_checks=1, exit_code=7)

        result, _ = self._run(process, policy=MonitorPolicy(1, 20, 30))

        self.assertEqual(result, 7)
        self.assertEqual(process.argv, self.command)
        self.assertEqual(process.cwd, str(self.root))
        self.assertFalse(process.shell)
        self.assertEqual(self._records()[-1]["event"], "CRASHED")

    def test_stall_is_advisory_and_does_not_terminate(self):
        process = FakeProcess(exit_after_checks=4, exit_code=0)

        result, _ = self._run(process, policy=MonitorPolicy(1, 2, 10))

        self.assertEqual(result, 0)
        self.assertIn("STALL_WARNING", [record["event"] for record in self._records()])
        self.assertFalse(process.terminated)
        self.assertFalse(process.killed)

    def test_hard_timeout_terminates_then_kills_if_needed(self):
        process = FakeProcess(exit_after_checks=None, ignores_terminate=True)

        result, clock = self._run(process, policy=MonitorPolicy(1, 20, 3))

        self.assertEqual(result, 124)
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertGreaterEqual(clock.value, 13)
        self.assertEqual(self._records()[-1]["event"], "HARD_TIMEOUT")

    def test_hard_timeout_kills_with_a_frozen_clock_and_noop_sleep(self):
        process = FrozenGraceProcess(exit_after_checks=None, ignores_terminate=True)
        clock_values = iter([0.0, *([1.0] * 100)])

        result = monitor_command(
            self.command,
            cwd=self.root,
            events_path=self.events,
            watched_paths=[self.log],
            policy=MonitorPolicy(1, 20, 1),
            clock=lambda: next(clock_values),
            sleep=lambda seconds: None,
            popen_factory=lambda *args, **kwargs: process,
        )

        self.assertEqual(result, 124)
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertLess(process.checks, 20)

    def test_hard_timeout_waits_full_ten_seconds_when_clock_advances_half_speed(self):
        process = FakeProcess(exit_after_checks=None, ignores_terminate=True)
        clock = HalfSpeedClock()

        result = monitor_command(
            self.command,
            cwd=self.root,
            events_path=self.events,
            watched_paths=[self.log],
            policy=MonitorPolicy(1, 20, 1),
            clock=clock,
            sleep=clock.sleep,
            popen_factory=lambda *args, **kwargs: process,
        )

        self.assertEqual(result, 124)
        self.assertTrue(process.killed)
        self.assertGreaterEqual(clock.value, 11)
        self.assertLessEqual(clock.value, 11.5)

    def test_changed_file_fingerprint_records_progress(self):
        self.log.write_text("epoch 1\n", encoding="utf-8")
        process = FakeProcess(exit_after_checks=3)
        clock = FakeClock()

        def sleep(seconds):
            clock.sleep(seconds)
            if clock.value == 1:
                self.log.write_text("epoch 2 complete\n", encoding="utf-8")

        result = monitor_command(
            self.command,
            cwd=self.root,
            events_path=self.events,
            watched_paths=[self.log],
            policy=MonitorPolicy(1, 10, 20),
            clock=clock,
            sleep=sleep,
            popen_factory=lambda *args, **kwargs: process,
        )

        self.assertEqual(result, 0)
        progress = [record for record in self._records() if record["event"] == "PROGRESS"]
        self.assertEqual(len(progress), 1)
        self.assertTrue(progress[0]["fingerprints"][0]["exists"])
        self.assertIn("mtime_ns", progress[0]["fingerprints"][0])

    def test_fatal_log_pattern_is_recorded_but_does_not_change_command_or_stop_process(self):
        self.log.write_text("RuntimeError: CUDA out of memory\n", encoding="utf-8")
        process = FakeProcess(exit_after_checks=2, exit_code=9)

        result, _ = self._run(process, policy=MonitorPolicy(1, 20, 30))

        records = self._records()
        fatal = next(record for record in records if record["event"] == "FATAL_PATTERN")
        self.assertEqual(result, 9)
        self.assertEqual(fatal["pattern"], "CUDA_OOM")
        self.assertFalse(process.terminated)
        self.assertFalse(process.killed)
        self.assertEqual(process.argv, self.command)

    def test_fatal_patterns_cover_all_reviewed_variants_once_without_stopping_child(self):
        self.log.write_text(
            "\n".join(
                [
                    "RuntimeError: CUDA error: out of memory",
                    "training loss: NaN",
                    "requests.exceptions.ConnectionError: could not connect",
                    "checkpoint_hash SHA-256 mismatch",
                    "HANS metrics do not exactly match recomputed predictions",
                ]
            ),
            encoding="utf-8",
        )
        process = FakeProcess(exit_after_checks=2, exit_code=9)

        result, _ = self._run(process, policy=MonitorPolicy(1, 20, 30))

        fatal_events = [record for record in self._records() if record["event"] == "FATAL_PATTERN"]
        self.assertEqual(result, 9)
        self.assertEqual(
            {"CUDA_OOM", "NONFINITE_LOSS", "DOWNLOAD_FAILURE", "CHECKPOINT_HASH_MISMATCH", "PREDICTION_ROW_MISMATCH"},
            {record["pattern"] for record in fatal_events},
        )
        self.assertEqual(5, len(fatal_events))
        self.assertFalse(process.terminated)
        self.assertFalse(process.killed)

    def test_documented_directory_watch_excludes_monitor_events_from_progress(self):
        process = FakeProcess(exit_after_checks=5)

        result, _ = self._run(
            process,
            policy=MonitorPolicy(1, 2, 10),
            watched_paths=[self.root],
        )

        names = [record["event"] for record in self._records()]
        self.assertEqual(result, 0)
        self.assertIn("STALL_WARNING", names)
        self.assertNotIn("PROGRESS", names)

    def test_invalid_cwd_is_rejected_before_starting_a_child(self):
        calls = []

        with self.assertRaisesRegex(ValueError, "cwd"):
            monitor_command(
                self.command,
                cwd=self.root / "missing",
                events_path=self.events,
                watched_paths=[self.log],
                popen_factory=lambda *args, **kwargs: calls.append((args, kwargs)),
            )

        self.assertEqual(calls, [])
        self.assertFalse(self.events.exists())

    def test_event_preflight_failure_does_not_start_a_child(self):
        blocked_parent = self.root / "not-a-directory"
        blocked_parent.write_text("file", encoding="utf-8")
        calls = []

        with self.assertRaises(OSError):
            monitor_command(
                self.command,
                cwd=self.root,
                events_path=blocked_parent / "events.jsonl",
                watched_paths=[self.log],
                popen_factory=lambda *args, **kwargs: calls.append((args, kwargs)),
            )

        self.assertEqual(calls, [])

    def test_popen_oserror_is_recorded_as_crash_without_orphan_child(self):
        def missing_executable(*args, **kwargs):
            raise FileNotFoundError("missing executable")

        result = monitor_command(
            self.command,
            cwd=self.root,
            events_path=self.events,
            watched_paths=[self.log],
            popen_factory=missing_executable,
        )

        records = self._records()
        self.assertEqual(result, 127)
        self.assertEqual(records[-1]["event"], "CRASHED")
        self.assertEqual(records[-1]["returncode"], 127)
        self.assertEqual(records[-1]["failure_stage"], "popen")

    def test_unreadable_watched_path_is_reported_in_status_check_without_new_event_type(self):
        blocked = self.root / "blocked.log"
        original_stat = Path.stat

        def blocked_stat(path, *args, **kwargs):
            if path == blocked:
                raise PermissionError("denied")
            return original_stat(path, *args, **kwargs)

        with patch("canonical.monitoring.Path.stat", new=blocked_stat):
            result, _ = self._run(
                FakeProcess(exit_after_checks=1),
                policy=MonitorPolicy(1, 20, 30),
                watched_paths=[blocked],
            )

        records = self._records()
        status = next(record for record in records if record["event"] == "STATUS_CHECK")
        self.assertEqual(result, 0)
        self.assertEqual(status["watch_errors"], [{"path": str(blocked), "error": "PermissionError"}])
        self.assertNotIn("WATCH_ERROR", [record["event"] for record in records])

    def test_unreadable_watched_directory_glob_is_reported_in_status_check(self):
        original_rglob = Path.rglob

        def blocked_rglob(path, pattern):
            if path == self.root:
                raise PermissionError("denied")
            return original_rglob(path, pattern)

        with patch("canonical.monitoring.Path.rglob", new=blocked_rglob):
            result, _ = self._run(
                FakeProcess(exit_after_checks=1),
                policy=MonitorPolicy(1, 20, 30),
                watched_paths=[self.root],
            )

        status = next(record for record in self._records() if record["event"] == "STATUS_CHECK")
        self.assertEqual(result, 0)
        self.assertEqual(status["watch_errors"], [{"path": str(self.root), "error": "PermissionError"}])

    def test_event_jsonl_has_common_strict_schema_and_no_nonfinite_numbers(self):
        process = FakeProcess(exit_after_checks=1)

        result, _ = self._run(process, policy=MonitorPolicy(1, 20, 30))

        self.assertEqual(result, 0)
        records = self._records()
        self.assertEqual(["STARTED", "STATUS_CHECK", "COMPLETED"], [record["event"] for record in records])
        for record in records:
            self.assertEqual(record["command"], self.command)
            self.assertEqual(record["cwd"], str(self.root))
            self.assertIsInstance(record["timestamp"], str)
            self.assertIsNotNone(datetime.fromisoformat(record["timestamp"]).tzinfo)
            self.assertIsInstance(record["elapsed_seconds"], (int, float))
            self.assertTrue(math.isfinite(record["elapsed_seconds"]))

    def test_monitor_preflight_keeps_documented_fresh_output_root_absent(self):
        output_root = self.root / "ties_results" / "stage2_smoke" / "local_rtx5080"
        evidence = self.root / "ties_results" / ".stage2_monitor" / "local_rtx5080.events.jsonl"

        def child_fresh_gate(*args, **kwargs):
            self.assertFalse(output_root.exists(), "monitor preflight polluted child --fresh output root")
            return FakeProcess(exit_after_checks=1)

        result = monitor_command(
            [
                "python",
                "run_stage2_smoke.py",
                "--mode",
                "primary",
                "--output-dir",
                str(output_root),
                "--fresh",
            ],
            cwd=self.root,
            events_path=evidence,
            watched_paths=[output_root],
            policy=MonitorPolicy(1, 20, 30),
            clock=FakeClock(),
            sleep=lambda seconds: None,
            popen_factory=child_fresh_gate,
        )

        self.assertEqual(result, 0)
        self.assertTrue(evidence.is_file())
        self.assertFalse(output_root.exists())

    def test_cli_help_needs_no_ml_dependencies(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "monitor_stage2_job.py"), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--events", result.stdout)
        self.assertIn("--watch", result.stdout)

    def test_cli_removes_separator_from_child_argv(self):
        received = {}
        original_monitor = monitor_stage2_job.monitor_command
        self.addCleanup(setattr, monitor_stage2_job, "monitor_command", original_monitor)

        def capture(command, **kwargs):
            received["command"] = command
            received["kwargs"] = kwargs
            return 0

        monitor_stage2_job.monitor_command = capture
        result = monitor_stage2_job.main(
            ["--events", "events.jsonl", "--watch", "run", "--", "python", "run.py"]
        )

        self.assertEqual(result, 0)
        self.assertEqual(received["command"], ["python", "run.py"])

    def test_cli_documents_sibling_evidence_for_fresh_output_command(self):
        received = {}
        original_monitor = monitor_stage2_job.monitor_command
        self.addCleanup(setattr, monitor_stage2_job, "monitor_command", original_monitor)
        output_root = Path("ties_results/stage2_smoke/local_rtx5080")
        evidence = Path("ties_results/.stage2_monitor/local_rtx5080.events.jsonl")

        def capture(command, **kwargs):
            received["command"] = command
            received["kwargs"] = kwargs
            return 0

        monitor_stage2_job.monitor_command = capture
        result = monitor_stage2_job.main(
            [
                "--events",
                str(evidence),
                "--watch",
                str(output_root),
                "--",
                "python",
                "run_stage2_smoke.py",
                "--output-dir",
                str(output_root),
                "--fresh",
            ]
        )

        self.assertEqual(result, 0)
        self.assertEqual(received["kwargs"]["events_path"], evidence)
        self.assertEqual(received["kwargs"]["watched_paths"], [output_root])
        self.assertNotIn(str(evidence), received["command"])

    def test_task7_runtime_ignore_and_evidence_archive_are_scoped_to_task7(self):
        from canonical.stage2_contract import EVIDENCE_ARCHIVE_NAME, MONITOR_PATHS, NOTEBOOK_PATH

        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("ties_results/.stage2_monitor/", gitignore)
        self.assertEqual("notebooks/stage2_colab_a100_smoke.ipynb", NOTEBOOK_PATH)
        self.assertEqual("stage2_a100_evidence.zip", EVIDENCE_ARCHIVE_NAME)
        self.assertEqual(2, len(MONITOR_PATHS))
        self.assertTrue(all(path.startswith("ties_results/.stage2_monitor/") for path in MONITOR_PATHS.values()))

    def test_task9_import_contract_names_both_sibling_a100_event_files(self):
        from canonical.stage2_contract import EVIDENCE_MEMBER_ROOT, MONITOR_PATHS, SAFE_EXTRACT_ROOT

        self.assertEqual("ties_results", EVIDENCE_MEMBER_ROOT)
        self.assertEqual(".", SAFE_EXTRACT_ROOT)
        self.assertEqual("ties_results/.stage2_monitor/colab_a100_run1.events.jsonl", MONITOR_PATHS["primary"])
        self.assertEqual("ties_results/.stage2_monitor/colab_a100_repeat_full_sr.events.jsonl", MONITOR_PATHS["repeat_full_sr"])

    def test_task_handoff_section_extractor_rejects_sibling_words_in_another_task(self):
        fallback_plan = """
### Task 7: source package only
No runtime evidence is retained here.
### Task 8: local runtime
### Task 9: import
ties_results/.stage2_monitor/ evidence archive
### Task 10: report
"""
        task7 = _task_section(fallback_plan, "Task 7:", "Task 8:")
        task9 = _task_section(fallback_plan, "Task 9:", "Task 10:")

        def require_task7_contract(section):
            self.assertIn("ties_results/.stage2_monitor/", section)
            self.assertIn("evidence archive", section)

        with self.assertRaises(AssertionError):
            require_task7_contract(task7)
        self.assertIn("ties_results/.stage2_monitor/", task9)

    def test_gitignore_effectively_ignores_sibling_monitor_evidence(self):
        candidate = "ties_results/.stage2_monitor/local_rtx5080.events.jsonl"
        checked = subprocess.run(
            ["git", "check-ignore", "-q", candidate], cwd=ROOT, check=False
        )
        if checked.returncode == 128:
            rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            self.assertIn("ties_results/.stage2_monitor/", rules)
        else:
            self.assertEqual(checked.returncode, 0)


if __name__ == "__main__":
    unittest.main()
