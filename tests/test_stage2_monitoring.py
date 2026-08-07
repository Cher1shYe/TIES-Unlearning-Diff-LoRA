import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import monitor_stage2_job


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.monitoring import MonitorPolicy, PRODUCTION_POLICY, monitor_command


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

    def test_event_jsonl_has_common_strict_schema_and_no_nonfinite_numbers(self):
        process = FakeProcess(exit_after_checks=1)

        result, _ = self._run(process, policy=MonitorPolicy(1, 20, 30))

        self.assertEqual(result, 0)
        records = self._records()
        self.assertEqual(["STARTED", "STATUS_CHECK", "COMPLETED"], [record["event"] for record in records])
        for record in records:
            self.assertEqual(record["command"], self.command)
            self.assertEqual(record["cwd"], str(self.root))
            self.assertIsInstance(record["timestamp"], (int, float))
            self.assertIsInstance(record["elapsed_seconds"], (int, float))
            self.assertTrue(math.isfinite(record["timestamp"]))
            self.assertTrue(math.isfinite(record["elapsed_seconds"]))

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


if __name__ == "__main__":
    unittest.main()
