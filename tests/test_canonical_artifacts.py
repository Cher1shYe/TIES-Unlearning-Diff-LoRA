import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.artifacts import read_jsonl, sha256_file, write_json, write_jsonl


class CanonicalArtifactContractTest(unittest.TestCase):
    def test_strict_json_rejects_nested_non_finite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            for bad in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(value=bad):
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        write_json(path, {"nested": [1.0, {"bad": bad}]})
                    self.assertFalse(path.exists())

    def test_strict_json_preserves_null_and_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            write_json(path, {"missing": None, "label": "非蕴含"})

            parsed = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(parsed["missing"])
            self.assertEqual("非蕴含", parsed["label"])
            self.assertIn('"missing": null', path.read_text(encoding="utf-8"))

    def test_jsonl_round_trip_is_one_object_per_line(self):
        records = [{"pair_id": "p-1", "value": 1.0}, {"pair_id": "p-2", "value": None}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            write_jsonl(path, records)

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(2, len(lines))
            self.assertEqual(records, read_jsonl(path))

    def test_jsonl_rejects_non_finite_without_committing_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            with self.assertRaisesRegex(ValueError, "non-finite"):
                write_jsonl(path, [{"ok": 1.0}, {"bad": math.nan}])
            self.assertFalse(path.exists())

    def test_sha256_changes_with_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.bin"
            path.write_bytes(b"first")
            first = sha256_file(path)
            path.write_bytes(b"second")
            second = sha256_file(path)

            self.assertEqual(64, len(first))
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
