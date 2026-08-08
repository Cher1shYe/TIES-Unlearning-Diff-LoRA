from copy import deepcopy
import json
from hashlib import sha256
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.artifacts import sha256_file, write_json, write_jsonl
from canonical.data import build_hans_selection_integrity, hans_manifest_identity_summary
from canonical.hans import aggregate_hans_predictions
from canonical.runner import _METHOD_OUTPUTS, _SHARED_OUTPUTS
from canonical.stage2_validation import (
    _validate_manifest_identity_audit,
    compare_a100_repeat as _production_compare_a100_repeat,
    compare_metric_values,
    validate_smoke_root as _production_validate_smoke_root,
)


PRIMARY_CONDITIONS = ("standard_lora", "full_sr", "class_prior_reweight")
HASH_A = "a" * 64
HASH_B = "b" * 64


def _predictions(method, checkpoint_hash):
    return [
        {
            "pair_id": "hans_evaluation::ex0",
            "gold_label": "entailment",
            "predicted_label": "entailment",
            "entailment_probability": 0.9,
            "heuristic": "lexical_overlap",
            "subcase": "entailment_case",
            "training_seed": 42,
            "method_tag": method,
            "checkpoint_hash": checkpoint_hash,
        },
        {
            "pair_id": "hans_evaluation::ex1",
            "gold_label": "non-entailment",
            "predicted_label": "non-entailment",
            "entailment_probability": 0.1,
            "heuristic": "subsequence",
            "subcase": "non_entailment_case",
            "training_seed": 42,
            "method_tag": method,
            "checkpoint_hash": checkpoint_hash,
        },
    ]


def _identity_entry(
    full_ids,
    *,
    selected_ids=None,
    source,
    split,
    preferred_id_fields,
    strata_fields=(),
    selected_limit=None,
):
    values = list(full_ids)
    selected = list(values if selected_ids is None else selected_ids)
    full_digest = sha256(
        json.dumps(
            values,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    selected_digest = sha256(
        json.dumps(
            selected,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "source": source,
        "split": split,
        "id_strategy": "preferred_field_or_content_sha256",
        "preferred_id_fields": list(preferred_id_fields),
        "strata_fields": list(strata_fields),
        "selection_seed": 42,
        "selected_limit": selected_limit,
        "full_count": len(values),
        "selected_count": len(selected),
        "full_ids": values,
        "selected_ids": selected,
        "full_ids_sha256": full_digest,
        "selected_ids_sha256": selected_digest,
    }


def _hans_content_integrity(hans_entries):
    content = {
        "build": ["1" * 64],
        "dev": ["2" * 64],
        "evaluation": ["3" * 64, "4" * 64],
    }
    partitions = {}
    for name, hashes in content.items():
        ordered_checksum = sha256(
            json.dumps(
                hashes,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        joint = list(zip(hans_entries[name]["full_ids"], hashes))
        joint_checksum = sha256(
            json.dumps(
                joint,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        partitions[name] = {
            "count": len(hashes),
            "content_sha256": hashes,
            "content_sha256_ordered_checksum": ordered_checksum,
            "source_id_content_joint_checksum": joint_checksum,
            "duplicate_content_count": 0,
        }
    return {
        "schema_version": "hans_content_integrity_v1",
        "algorithm": "sha256_canonical_json_utf8_v1",
        "fields": ["gold_label", "premise", "hypothesis", "heuristic", "subcase"],
        "excludes_pair_id": True,
        "partitions": partitions,
        "overlap_counts": {
            "build_dev": 0,
            "build_evaluation": 0,
            "dev_evaluation": 0,
        },
    }


def _fixture_hans_manifest():
    hans = {
        "build": _identity_entry(
            ["hans_train::ex0"], source="tommccoy1/hans", split="build",
            preferred_id_fields=("pairID",),
        ),
        "dev": _identity_entry(
            ["hans_train::ex1"], source="tommccoy1/hans", split="dev",
            preferred_id_fields=("pairID",),
        ),
        "evaluation": _identity_entry(
            ["hans_evaluation::ex0", "hans_evaluation::ex1"],
            source="tommccoy1/hans", split="evaluation",
            preferred_id_fields=("pairID",),
            strata_fields=("gold_label", "heuristic", "subcase"),
            selected_limit=2,
        ),
    }
    split_payload = {
        "schema_version": "hans_split_v1",
        "hans_split_seed": 42,
        "build_pair_ids": ["ex0"],
        "dev_pair_ids": ["ex1"],
        "small_strata": [],
    }
    hans["split_integrity"] = {
        "schema_version": "hans_split_integrity_v1",
        "seed": 42,
        "split_algorithm": "source_local_id_sort_numpy_default_rng_per_stratum_v1",
        "checksum_algorithm": "sha256_canonical_json_utf8_v1",
        "build_count": 1,
        "dev_count": 1,
        "build_source_pair_ids": ["ex0"],
        "dev_source_pair_ids": ["ex1"],
        "small_strata": [],
        "split_checksum": sha256(
            json.dumps(
                split_payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    hans["content_integrity"] = _hans_content_integrity(hans)
    selected_records = [
        {"pairID": "ex0", "canonical_pair_id": "hans_evaluation::ex0"},
        {"pairID": "ex1", "canonical_pair_id": "hans_evaluation::ex1"},
    ]
    hans["selection_integrity"] = build_hans_selection_integrity(
        selected_records,
        hans["evaluation"]["selected_ids"],
        limit=2,
        seed=42,
    )
    return hans


TEST_HANS_MANIFEST = _fixture_hans_manifest()
TEST_HANS_ANCHORS = {
    "schema_version": "hans_official_semantic_anchors_v1",
    "split_checksum": TEST_HANS_MANIFEST["split_integrity"]["split_checksum"],
    "partitions": {
        name: {
            "count": TEST_HANS_MANIFEST[name]["full_count"],
            "source_pair_ids_sha256": sha256(
                json.dumps(
                    TEST_HANS_MANIFEST["split_integrity"].get(
                        f"{name}_source_pair_ids", []
                    ),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "qualified_ids_sha256": TEST_HANS_MANIFEST[name]["full_ids_sha256"],
            "content_sha256_ordered_checksum": TEST_HANS_MANIFEST["content_integrity"]["partitions"][name]["content_sha256_ordered_checksum"],
            "source_id_content_joint_checksum": TEST_HANS_MANIFEST["content_integrity"]["partitions"][name]["source_id_content_joint_checksum"],
        }
        for name in ("build", "dev", "evaluation")
    },
    "selection_2": {
        "count": 2,
        "selected_source_pair_ids_sha256": TEST_HANS_MANIFEST["selection_integrity"]["selected_source_pair_ids_sha256"],
        "selected_artifact_ids_sha256": TEST_HANS_MANIFEST["selection_integrity"]["selected_artifact_ids_sha256"],
        "source_to_artifact_mapping_sha256": TEST_HANS_MANIFEST["selection_integrity"]["source_to_artifact_mapping_sha256"],
    },
}
TEST_STAGE2_PROFILE = {
    ("mnli", "train"): ("nyu-mll/glue:mnli", "train", ["idx", "row_id", "id", "uid"], [], 4, 2, 2),
    ("mnli", "validation_matched"): ("nyu-mll/glue:mnli", "validation_matched", ["idx", "row_id", "id", "uid"], [], 4, 2, 2),
    ("hans", "build"): ("tommccoy1/hans", "build", ["pairID"], [], 1, 1, None),
    ("hans", "dev"): ("tommccoy1/hans", "dev", ["pairID"], [], 1, 1, None),
    ("hans", "evaluation"): ("tommccoy1/hans", "evaluation", ["pairID"], ["gold_label", "heuristic", "subcase"], 2, 2, 2),
    ("ood", "esnli"): ("e-SNLI", "test", ["pairID", "uid", "id", "idx"], [], 2, 1, 1),
    ("ood", "anli"): ("facebook/anli", "test", ["pairID", "uid", "id", "idx"], [], 2, 1, 1),
    ("ood", "snli_hard"): ("snli_1.0_test_hard", "test", ["pairID", "uid", "id", "idx"], [], 2, 1, 1),
    ("ood", "wanli"): ("alisawuffles/WANLI", "test", ["pairID", "uid", "id", "idx"], [], 2, 1, 1),
}


def _controlled_contract():
    return (
        patch("canonical.stage2_validation._STAGE2_DATA_PROFILE", TEST_STAGE2_PROFILE),
        patch("canonical.stage2_validation.HANS_OFFICIAL_ANCHORS_V1", TEST_HANS_ANCHORS),
    )


def validate_smoke_root(*args, **kwargs):
    profile, anchors = _controlled_contract()
    with profile, anchors:
        return _production_validate_smoke_root(*args, **kwargs)


def compare_a100_repeat(*args, **kwargs):
    profile, anchors = _controlled_contract()
    with profile, anchors:
        return _production_compare_a100_repeat(*args, **kwargs)


def _write_success_status(directory, relative_outputs):
    write_json(
        directory / "status.json",
        {
            "schema_version": "canonical_status_v1",
            "state": "success",
            "started_at": "2026-08-08T00:00:00+00:00",
            "finished_at": "2026-08-08T00:00:01+00:00",
            "wall_time_seconds": 1.0,
            "exit_status": 0,
            "peak_gpu_memory_bytes": None,
            "error_type": None,
            "error": None,
            "output_hashes": {
                relative: sha256_file(directory / relative) for relative in relative_outputs
            },
        },
    )


def _rehash_status(directory):
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    _write_success_status(directory, list(status["output_hashes"]))


def _rebind_data_manifest_hash(root):
    data_hash = sha256_file(Path(root) / "manifests" / "data_manifest.json")
    for manifest_path in (Path(root) / "seed_42").glob("*/run_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["data_manifest_sha256"] = data_hash
        write_json(manifest_path, manifest)
        _rehash_status(manifest_path.parent)


def _rebind_manifest_identity_summary(root):
    root = Path(root)
    manifest = json.loads(
        (root / "manifests" / "data_manifest.json").read_text(encoding="utf-8")
    )
    audit = root / "manifests" / "data_access.jsonl"
    events = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    events[1] = {
        **{
            key: events[1][key]
            for key in ("sequence", "timestamp", "event", "dataset", "split", "purpose")
        },
        **hans_manifest_identity_summary(manifest["hans"]),
    }
    write_jsonl(audit, events)


def _set_status_hashes(directory, relative_paths):
    _write_success_status(directory, relative_paths)


def _repeat_root(base, **kwargs):
    kwargs.setdefault("mode", "repeat_full_sr")
    kwargs.setdefault("conditions", ("full_sr",))
    return _create_smoke_root(base, **kwargs)


def _manifest(method, checkpoint=None, *, protocol_hash=HASH_A, data_hash=HASH_A, environment_hash=HASH_B):
    return {
        "schema_version": "canonical_run_manifest_v1",
        "role": "standard_lora" if method == "standard_lora" else "dual_adapter_branch",
        "method_tag": method,
        "data_seed": 42,
        "hans_split_seed": 42,
        "training_seed": 42,
        "protocol_sha256": protocol_hash,
        "data_manifest_sha256": data_hash,
        "environment_manifest_sha256": environment_hash,
        "git": {"commit": "c" * 40, "dirty": False},
        "command": ["python", "run_stage2_smoke.py"],
        "shared_phase2_checkpoint": checkpoint,
        "result": {"final_checkpoint_hash": HASH_B},
    }


def _create_smoke_root(
    base,
    *,
    environment="NVIDIA A100-SXM4-40GB",
    mnli_accuracy=0.8,
    mode="primary",
    conditions=PRIMARY_CONDITIONS,
    smoke_environment="colab_a100",
):
    root = Path(base) / "smoke"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    hans_entries = deepcopy(TEST_HANS_MANIFEST)
    mnli = {
        "train": _identity_entry(
            [f"train-{index}" for index in range(4)],
            selected_ids=["train-0", "train-1"],
            source="nyu-mll/glue:mnli", split="train",
            preferred_id_fields=("idx", "row_id", "id", "uid"), selected_limit=2,
        ),
        "validation_matched": _identity_entry(
            [f"validation-{index}" for index in range(4)],
            selected_ids=["validation-0", "validation-1"],
            source="nyu-mll/glue:mnli", split="validation_matched",
            preferred_id_fields=("idx", "row_id", "id", "uid"), selected_limit=2,
        ),
    }
    ood = {
        name: _identity_entry(
            [f"{name}-{index}" for index in range(2)],
            selected_ids=[f"{name}-0"],
            source=source, split="test",
            preferred_id_fields=("pairID", "uid", "id", "idx"), selected_limit=1,
        )
        for name, source in {
            "esnli": "e-SNLI", "anli": "facebook/anli",
            "snli_hard": "snli_1.0_test_hard", "wanli": "alisawuffles/WANLI",
        }.items()
    }
    write_json(
        manifests / "data_manifest.json",
        {
            "schema_version": "canonical_data_manifest_v4",
            "scope": "stage2_smoke",
            "data_seed": 42,
            "hans_split_seed": 42,
            "mnli": mnli,
            "hans": hans_entries,
            "ood": ood,
        },
    )
    write_json(
        manifests / "environment_manifest.json",
        {"schema_version": "environment_manifest_v1", "gpu": environment, "python": "3.11"},
    )
    write_json(
        manifests / "run_matrix.json",
        {
            "schema_version": "stage2_smoke_matrix_v1",
            "training_seeds": [42],
            "condition_orders": {"42": list(conditions)},
        },
    )
    data_hash = sha256_file(manifests / "data_manifest.json")
    environment_hash = sha256_file(manifests / "environment_manifest.json")
    write_json(
        root / "commands.json",
        {
            "schema_version": "stage2_smoke_commands_v1",
            "mode": mode,
            "environment": smoke_environment,
            "argv": ["python", "run_stage2_smoke.py"],
            "expected_condition_tags": list(conditions),
            "profile_name": "stage2_smoke_v1",
            "gpu_name": environment,
            "started_at": "2026-08-08T00:00:00+00:00",
        },
    )
    (root / "protocol_snapshot").mkdir()
    (root / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md").write_text(
        "# frozen\n", encoding="utf-8"
    )
    protocol_hash = sha256_file(root / "protocol_snapshot" / "FROZEN_EXPERIMENT_PROTOCOL.md")
    (root / "protocol_snapshot" / "protocol_sha256.txt").write_text(protocol_hash + "\n", encoding="utf-8")
    write_jsonl(
        manifests / "data_access.jsonl",
        [
            {"sequence": 0, "timestamp": "2026-08-08T00:00:00+00:00", "event": "dataset_access", "dataset": "hans", "split": "evaluation", "purpose": "manifest_identity_only"},
            {
                "sequence": 1,
                "timestamp": "2026-08-08T00:00:01+00:00",
                "event": "manifest_identity_summary",
                "dataset": "hans",
                "split": "evaluation",
                "purpose": "manifest_identity_only",
                **hans_manifest_identity_summary(hans_entries),
            },
        ],
    )

    seed = root / "seed_42"
    shared = seed / "shared_phase2"
    checkpoint = shared / "checkpoints" / "shared.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"shared checkpoint")
    checkpoint_hash = sha256_file(checkpoint)
    write_json(
        shared / "config.json",
        {"data_seed": 42, "hans_split_seed": 42, "training_seed": 42, "output_dir": str(shared)},
    )
    shared_manifest = _manifest(
        None, protocol_hash=protocol_hash, data_hash=data_hash, environment_hash=environment_hash
    )
    shared_manifest["role"] = "shared_phase2"
    shared_manifest["method_tag"] = None
    shared_manifest["result"] = {"checkpoint_sha256": checkpoint_hash}
    write_json(shared / "run_manifest.json", shared_manifest)
    write_json(shared / "shared_checkpoint.json", {"path_relative": "checkpoints/shared.pt", "sha256": checkpoint_hash})
    weights = {"0": 0.1, "1": 0.2, "2": 0.3}
    write_json(
        shared / "shared_checkpoint_metadata.json",
        {"checkpoint_role": "canonical_shared_phase2", "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hash, "class_prior_weights": weights},
    )
    write_jsonl(shared / "data_access.jsonl", [])
    (shared / "stdout.log").write_text("shared\n", encoding="utf-8")
    (shared / "stderr.log").write_text("", encoding="utf-8")
    _write_success_status(
        shared,
        [*_SHARED_OUTPUTS, "checkpoints/shared.pt"],
    )

    shared_ref = {"path": str(checkpoint), "sha256": checkpoint_hash}
    for method in conditions:
        run = seed / method
        run.mkdir(parents=True)
        write_json(
            run / "config.json",
            {
                "data_seed": 42,
                "hans_split_seed": 42,
                "training_seed": 42,
                "method_tag": method,
                "output_dir": str(run),
            },
        )
        manifest = _manifest(
            method,
            None if method == "standard_lora" else shared_ref,
            protocol_hash=protocol_hash,
            data_hash=data_hash,
            environment_hash=environment_hash,
        )
        write_json(run / "run_manifest.json", manifest)
        rows = _predictions(method, HASH_B)
        metrics = {
            "final": {
                "mnli": {"mnli_accuracy": mnli_accuracy},
                "hans": aggregate_hans_predictions(rows),
                "esnli": {"esnli_accuracy": 0.7},
                "anli": {"anli_accuracy": 0.6},
                "snli_hard": {"snli_hard_accuracy": 0.5},
                "wanli": {"wanli_accuracy": 0.4},
            }
        }
        write_json(run / "metrics.json", metrics)
        write_json(run / "selected_layers.json", {})
        write_jsonl(
            run / "data_access.jsonl",
            [
                {"sequence": 0, "timestamp": "2026-08-08T00:00:00+00:00", "event": "final_evaluation_start", "dataset": "hans", "split": None, "purpose": "boundary"},
                {"sequence": 1, "timestamp": "2026-08-08T00:00:01+00:00", "event": "dataset_access", "dataset": "hans", "split": "evaluation", "purpose": "final"},
            ],
        )
        write_jsonl(run / "hans_predictions.jsonl", rows)
        log = "class-prior reweighting ON: {0: 0.1, 1: 0.2, 2: 0.3}\n" if method == "class_prior_reweight" else "run\n"
        (run / "stdout.log").write_text(log, encoding="utf-8")
        (run / "stderr.log").write_text("", encoding="utf-8")
        _write_success_status(
            run,
            _METHOD_OUTPUTS,
        )
    return root


class Stage2ValidationTest(unittest.TestCase):
    def test_v4_manifest_without_record_computed_source_integrity_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "v4-source-bypass")

            with self.assertRaisesRegex(ValueError, "v5|source.integrity|data manifest"):
                validate_smoke_root(
                    root,
                    expected_conditions=PRIMARY_CONDITIONS,
                    canonical_dir=Path(tmp) / "canonical_v1",
                )

    def test_manifest_identity_audit_requires_source_integrity_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "missing-source-summary")
            audit = root / "manifests" / "data_access.jsonl"
            events = [
                json.loads(line)
                for line in audit.read_text(encoding="utf-8").splitlines()
            ]
            expected_summary = hans_manifest_identity_summary(TEST_HANS_MANIFEST)

            with self.assertRaisesRegex(ValueError, "source.integrity|manifest identity"):
                _validate_manifest_identity_audit(
                    events,
                    source=audit,
                    expected_summary=expected_summary,
                )

    def test_two_id_root_labelled_stage2_smoke_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "two-id-profile")

            with self.assertRaisesRegex(ValueError, "schema|scope|profile|count"):
                _production_validate_smoke_root(
                    root,
                    expected_conditions=PRIMARY_CONDITIONS,
                    canonical_dir=Path(tmp) / "canonical_v1",
                )

    def test_manifest_identity_summary_must_equal_the_data_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "stale-summary")
            audit = root / "manifests" / "data_access.jsonl"
            events = [
                json.loads(line)
                for line in audit.read_text(encoding="utf-8").splitlines()
            ]
            events[1]["identity_counts"] = {
                "build": 999,
                "dev": 999,
                "evaluation": 999,
            }
            events[1]["identity_checksums"] = {
                "build": "9" * 64,
                "dev": "9" * 64,
                "evaluation": "9" * 64,
            }
            expected_summary = {
                "identity_counts": {"build": 1, "dev": 1, "evaluation": 2},
                "identity_checksums": {
                    "build": HASH_A,
                    "dev": HASH_A,
                    "evaluation": HASH_A,
                },
                "split_integrity_summary": {"split_checksum": HASH_A},
                "content_integrity_summary": {"evaluation": HASH_A},
                "selection_integrity_summary": {"selection": HASH_A},
            }
            events[1]["split_integrity_summary"] = expected_summary["split_integrity_summary"]
            events[1]["content_integrity_summary"] = expected_summary["content_integrity_summary"]
            events[1]["selection_integrity_summary"] = expected_summary["selection_integrity_summary"]

            with self.assertRaisesRegex(ValueError, "manifest identity.*summary"):
                _validate_manifest_identity_audit(
                    events,
                    source=audit,
                    expected_summary=expected_summary,
                )

    def test_rehashed_content_reorder_cannot_replace_official_semantic_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "rehashed-content")
            path = root / "manifests" / "data_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            entry = manifest["hans"]["content_integrity"]["partitions"]["evaluation"]
            entry["content_sha256"] = list(reversed(entry["content_sha256"]))
            entry["content_sha256_ordered_checksum"] = sha256(
                json.dumps(
                    entry["content_sha256"],
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            pairs = list(
                zip(
                    manifest["hans"]["evaluation"]["full_ids"],
                    entry["content_sha256"],
                )
            )
            entry["source_id_content_joint_checksum"] = sha256(
                json.dumps(
                    pairs,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            write_json(path, manifest)
            _rebind_data_manifest_hash(root)

            with self.assertRaisesRegex(ValueError, "official.*content|semantic anchor"):
                validate_smoke_root(
                    root,
                    expected_conditions=PRIMARY_CONDITIONS,
                    canonical_dir=Path(tmp) / "canonical_v1",
                )

    def test_rehashed_split_swap_cannot_replace_official_semantic_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "rehashed-split")
            path = root / "manifests" / "data_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            hans = manifest["hans"]
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
            checksum_payload = {
                "schema_version": "hans_split_v1",
                "hans_split_seed": split["seed"],
                "build_pair_ids": split["build_source_pair_ids"],
                "dev_pair_ids": split["dev_source_pair_ids"],
                "small_strata": [],
            }
            split["split_checksum"] = sha256(
                json.dumps(
                    checksum_payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            write_json(path, manifest)
            _rebind_manifest_identity_summary(root)
            _rebind_data_manifest_hash(root)

            with self.assertRaisesRegex(ValueError, "official.*split.*anchor"):
                validate_smoke_root(
                    root,
                    expected_conditions=PRIMARY_CONDITIONS,
                    canonical_dir=Path(tmp) / "canonical_v1",
                )

    def test_rehashed_selection_mapping_cannot_replace_official_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "rehashed-selection")
            path = root / "manifests" / "data_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            hans = manifest["hans"]
            evaluation = hans["evaluation"]
            evaluation["selected_ids"] = list(reversed(evaluation["selected_ids"]))
            evaluation["selected_ids_sha256"] = sha256(
                json.dumps(
                    evaluation["selected_ids"],
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            selection = hans["selection_integrity"]
            selection["selected_source_pair_ids"] = list(
                reversed(selection["selected_source_pair_ids"])
            )
            mapping = list(
                zip(selection["selected_source_pair_ids"], evaluation["selected_ids"])
            )
            selection["selected_source_pair_ids_sha256"] = sha256(
                json.dumps(
                    selection["selected_source_pair_ids"],
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            selection["selected_artifact_ids_sha256"] = evaluation[
                "selected_ids_sha256"
            ]
            selection["source_to_artifact_mapping_sha256"] = sha256(
                json.dumps(
                    mapping,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            selection_payload = {
                key: value
                for key, value in selection.items()
                if key != "integrity_checksum"
            }
            selection["integrity_checksum"] = sha256(
                json.dumps(
                    [[key, selection_payload[key]] for key in sorted(selection_payload)],
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            write_json(path, manifest)
            _rebind_manifest_identity_summary(root)
            _rebind_data_manifest_hash(root)

            with self.assertRaisesRegex(ValueError, "official.*selection.*anchor"):
                validate_smoke_root(
                    root,
                    expected_conditions=PRIMARY_CONDITIONS,
                    canonical_dir=Path(tmp) / "canonical_v1",
                )

    def test_weight_optional_validator_reuses_full_semantics_for_exact_shared_checkpoint_omission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "optional")
            checkpoint = root / "seed_42" / "shared_phase2" / "checkpoints" / "shared.pt"
            omitted = {"seed_42/shared_phase2/checkpoints/shared.pt": sha256_file(checkpoint)}
            checkpoint.unlink()

            report = validate_smoke_root(
                root, expected_conditions=PRIMARY_CONDITIONS,
                canonical_dir=Path(tmp) / "canonical_v1", omitted_weights=omitted,
            )

            self.assertEqual("pass", report["state"])

    def test_weight_optional_validator_rejects_branch_checkpoint_hash_spoof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "optional-spoof")
            checkpoint = root / "seed_42" / "shared_phase2" / "checkpoints" / "shared.pt"
            omitted = {"seed_42/shared_phase2/checkpoints/shared.pt": sha256_file(checkpoint)}
            checkpoint.unlink()
            manifest_path = root / "seed_42" / "full_sr" / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["shared_phase2_checkpoint"]["sha256"] = "0" * 64
            write_json(manifest_path, manifest)
            _rehash_status(manifest_path.parent)

            with self.assertRaisesRegex(ValueError, "shared checkpoint"):
                validate_smoke_root(
                    root, expected_conditions=PRIMARY_CONDITIONS,
                    canonical_dir=Path(tmp) / "canonical_v1", omitted_weights=omitted,
                )

    def test_validator_recomputes_hans_and_matches_metrics_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "nan")
            canonical = Path(tmp) / "canonical_v1"

            report = validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=canonical)

            self.assertEqual("pass", report["checks"]["hans_recomputation"]["state"])
            self.assertEqual("pass", report["checks"]["audit_order"]["state"])

    def test_validator_requires_ordered_prediction_ids_to_equal_manifest_membership(self):
        def missing(rows):
            return rows[:1]

        def extra(rows):
            return [*rows, dict(rows[0], pair_id="hans_evaluation::ex2")]

        def reordered(rows):
            return list(reversed(rows))

        def duplicate(rows):
            return [rows[0], dict(rows[1], pair_id=rows[0]["pair_id"])]

        def raw(rows):
            return [dict(rows[0], pair_id="ex0"), rows[1]]

        def wrong_namespace(rows):
            return [dict(rows[0], pair_id="hans_train::ex0"), rows[1]]

        transforms = (missing, extra, reordered, duplicate, raw, wrong_namespace)
        with tempfile.TemporaryDirectory() as tmp:
            for index, transform in enumerate(transforms):
                with self.subTest(transform=transform.__name__):
                    root = _create_smoke_root(Path(tmp) / f"ids-{index}")
                    run = root / "seed_42" / "full_sr"
                    prediction_path = run / "hans_predictions.jsonl"
                    rows = [
                        json.loads(line)
                        for line in prediction_path.read_text(encoding="utf-8").splitlines()
                    ]
                    write_jsonl(prediction_path, transform(rows))
                    _rehash_status(run)

                    with self.assertRaisesRegex(
                        ValueError, "ordered HANS prediction IDs.*selected_ids"
                    ):
                        validate_smoke_root(
                            root,
                            expected_conditions=PRIMARY_CONDITIONS,
                            canonical_dir=Path(tmp) / "canonical_v1",
                        )

    def test_validator_rejects_invalid_hans_manifest_identities_and_checksums(self):
        def raw(entry):
            entry["full_ids"][0] = entry["selected_ids"][0] = "ex0"

        def wrong_namespace(entry):
            entry["full_ids"][0] = entry["selected_ids"][0] = "hans_train::ex0"

        def double_qualified(entry):
            entry["full_ids"][0] = entry["selected_ids"][0] = (
                "hans_evaluation::hans_evaluation::ex0"
            )

        def padded_suffix(entry):
            entry["full_ids"][0] = entry["selected_ids"][0] = "hans_evaluation::ex00"

        def signed_suffix(entry):
            entry["full_ids"][0] = entry["selected_ids"][0] = "hans_evaluation::ex-1"

        def delimiter_suffix(entry):
            entry["full_ids"][0] = entry["selected_ids"][0] = "hans_evaluation::ex1::extra"

        def duplicate(entry):
            entry["full_ids"][1] = entry["selected_ids"][1] = entry["full_ids"][0]

        def invalid_checksum(entry):
            entry["selected_ids_sha256"] = "0" * 64

        with tempfile.TemporaryDirectory() as tmp:
            for index, transform in enumerate(
                (
                    raw,
                    wrong_namespace,
                    double_qualified,
                    padded_suffix,
                    signed_suffix,
                    delimiter_suffix,
                    duplicate,
                    invalid_checksum,
                )
            ):
                with self.subTest(transform=transform.__name__):
                    root = _create_smoke_root(Path(tmp) / f"manifest-{index}")
                    path = root / "manifests" / "data_manifest.json"
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                    entry = manifest["hans"]["evaluation"]
                    transform(entry)
                    if transform is not invalid_checksum:
                        digest = sha256(
                            json.dumps(
                                entry["full_ids"],
                                ensure_ascii=False,
                                allow_nan=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()
                        entry["full_ids_sha256"] = entry["selected_ids_sha256"] = digest
                    write_json(path, manifest)

                    with self.assertRaisesRegex(ValueError, "HANS manifest|Stage 2 data profile"):
                        validate_smoke_root(
                            root,
                            expected_conditions=PRIMARY_CONDITIONS,
                            canonical_dir=Path(tmp) / "canonical_v1",
                        )

    def test_validator_requires_v4_content_integrity_and_rejects_bound_tamper(self):
        def missing(manifest):
            del manifest["hans"]["content_integrity"]

        def old_schema(manifest):
            manifest["schema_version"] = "canonical_data_manifest_v2"

        def wrong_fields(manifest):
            manifest["hans"]["content_integrity"]["fields"][-1] = "pairID"

        def bad_ordered_checksum(manifest):
            manifest["hans"]["content_integrity"]["partitions"]["evaluation"][
                "content_sha256_ordered_checksum"
            ] = "0" * 64

        def bad_joint_checksum(manifest):
            manifest["hans"]["content_integrity"]["partitions"]["evaluation"][
                "source_id_content_joint_checksum"
            ] = "0" * 64

        def duplicate_content_with_rebuilt_checksums(manifest):
            integrity = manifest["hans"]["content_integrity"]
            entry = integrity["partitions"]["evaluation"]
            entry["content_sha256"][1] = entry["content_sha256"][0]
            entry["content_sha256_ordered_checksum"] = sha256(
                json.dumps(
                    entry["content_sha256"],
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            pairs = list(
                zip(
                    manifest["hans"]["evaluation"]["full_ids"],
                    entry["content_sha256"],
                )
            )
            entry["source_id_content_joint_checksum"] = sha256(
                json.dumps(
                    pairs,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()

        def overlap_content_with_rebuilt_checksums(manifest):
            integrity = manifest["hans"]["content_integrity"]
            entry = integrity["partitions"]["evaluation"]
            entry["content_sha256"][0] = integrity["partitions"]["build"][
                "content_sha256"
            ][0]
            entry["content_sha256_ordered_checksum"] = sha256(
                json.dumps(
                    entry["content_sha256"],
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            pairs = list(
                zip(
                    manifest["hans"]["evaluation"]["full_ids"],
                    entry["content_sha256"],
                )
            )
            entry["source_id_content_joint_checksum"] = sha256(
                json.dumps(
                    pairs,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()

        transforms = (
            missing,
            old_schema,
            wrong_fields,
            bad_ordered_checksum,
            bad_joint_checksum,
            duplicate_content_with_rebuilt_checksums,
            overlap_content_with_rebuilt_checksums,
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, transform in enumerate(transforms):
                with self.subTest(transform=transform.__name__):
                    root = _create_smoke_root(Path(tmp) / f"content-{index}")
                    path = root / "manifests" / "data_manifest.json"
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                    transform(manifest)
                    write_json(path, manifest)

                    with self.assertRaisesRegex(
                        ValueError,
                        "data manifest|HANS.*content|data profile|semantic anchor",
                    ):
                        validate_smoke_root(
                            root,
                            expected_conditions=PRIMARY_CONDITIONS,
                            canonical_dir=Path(tmp) / "canonical_v1",
                        )

    def test_validator_rejects_evaluation_access_before_final_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "tampered")
            write_jsonl(
                root / "seed_42" / "standard_lora" / "data_access.jsonl",
                [
                    {"sequence": 0, "timestamp": "2026-08-08T00:00:00+00:00", "event": "dataset_access", "dataset": "hans", "split": "evaluation", "purpose": "final"},
                    {"sequence": 1, "timestamp": "2026-08-08T00:00:01+00:00", "event": "final_evaluation_start", "dataset": "hans", "split": None, "purpose": "boundary"},
                ],
            )
            _rehash_status(root / "seed_42" / "standard_lora")
            with self.assertRaisesRegex(ValueError, "before final_evaluation_start"):
                validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

    def test_validator_rejects_nan_json_and_tampered_success_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "binding")
            metrics = root / "seed_42" / "standard_lora" / "metrics.json"
            metrics.write_text('{"final": NaN}\n', encoding="utf-8")
            _rehash_status(root / "seed_42" / "standard_lora")
            with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

            root = _create_smoke_root(Path(tmp) / "log")
            (root / "seed_42" / "full_sr" / "stdout.log").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

    def test_validator_rejects_missing_and_extra_status_hash_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = (("config.json", "missing required output hashes"), ("metrics.json", "missing required output hashes"))
            for index, (missing, message) in enumerate(cases):
                root = _create_smoke_root(Path(tmp) / f"missing-{index}")
                run = root / "seed_42" / "full_sr"
                _set_status_hashes(run, [name for name in _METHOD_OUTPUTS if name != missing])
                with self.assertRaisesRegex(ValueError, message):
                    validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

            root = _create_smoke_root(Path(tmp) / "extra")
            run = root / "seed_42" / "full_sr"
            (run / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
            _set_status_hashes(run, [*_METHOD_OUTPUTS, "unexpected.txt"])
            with self.assertRaisesRegex(ValueError, "unexpected output hashes"):
                validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

    def test_validator_rejects_shared_checkpoint_aliasing_a_required_metadata_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "shared-alias")
            shared = root / "seed_42" / "shared_phase2"
            config_hash = sha256_file(shared / "config.json")
            write_json(shared / "shared_checkpoint.json", {"path_relative": "config.json", "sha256": config_hash})
            metadata = json.loads((shared / "shared_checkpoint_metadata.json").read_text(encoding="utf-8"))
            metadata["checkpoint_path"] = str(shared / "config.json")
            metadata["checkpoint_sha256"] = config_hash
            write_json(shared / "shared_checkpoint_metadata.json", metadata)
            _set_status_hashes(shared, _SHARED_OUTPUTS)
            shared_ref = {"path": str(shared / "config.json"), "sha256": config_hash}
            for method in ("full_sr", "class_prior_reweight"):
                run = root / "seed_42" / method
                manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
                manifest["shared_phase2_checkpoint"] = shared_ref
                write_json(run / "run_manifest.json", manifest)
                _rehash_status(run)

            with self.assertRaisesRegex(ValueError, "checkpoint path"):
                validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

    def test_validator_rejects_exponent_overflow_in_nested_json_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "json")
            config = root / "seed_42" / "standard_lora" / "config.json"
            config.write_text('{"nested":[1e9999]}\n', encoding="utf-8")
            _rehash_status(root / "seed_42" / "standard_lora")
            with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
                validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

            root = _create_smoke_root(Path(tmp) / "jsonl")
            audit = root / "seed_42" / "standard_lora" / "data_access.jsonl"
            audit.write_text('{"sequence":0,"timestamp":"2026-08-08T00:00:00+00:00","event":"final_evaluation_start","dataset":"hans","split":null,"purpose":"boundary"}\n{"sequence":1,"timestamp":"2026-08-08T00:00:01+00:00","event":"dataset_access","dataset":"hans","split":"evaluation","purpose":"final","nested":{"bad":-1e9999}}\n', encoding="utf-8")
            _rehash_status(root / "seed_42" / "standard_lora")
            with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
                validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

    def test_validator_rejects_prediction_manifest_binding_and_bad_class_prior_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "binding")
            prediction_path = root / "seed_42" / "full_sr" / "hans_predictions.jsonl"
            rows = json.loads("[" + ",".join(prediction_path.read_text(encoding="utf-8").splitlines()) + "]")
            rows[0]["method_tag"] = "standard_lora"
            write_jsonl(prediction_path, rows)
            _rehash_status(root / "seed_42" / "full_sr")
            with self.assertRaisesRegex(ValueError, "method_tag"):
                validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

            root = _create_smoke_root(Path(tmp) / "log")
            (root / "seed_42" / "class_prior_reweight" / "stdout.log").write_text("class-prior reweighting ON\n", encoding="utf-8")
            _rehash_status(root / "seed_42" / "class_prior_reweight")
            with self.assertRaisesRegex(ValueError, "class-prior"):
                validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

    def test_validator_rejects_manifest_identity_only_spoofed_in_method_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "spoof")
            audit = root / "seed_42" / "standard_lora" / "data_access.jsonl"
            write_jsonl(
                audit,
                [
                    {"sequence": 0, "timestamp": "2026-08-08T00:00:00+00:00", "event": "dataset_access", "dataset": "hans", "split": "evaluation", "purpose": "manifest_identity_only"},
                    {"sequence": 1, "timestamp": "2026-08-08T00:00:01+00:00", "event": "final_evaluation_start", "dataset": "hans", "split": None, "purpose": "boundary"},
                    {"sequence": 2, "timestamp": "2026-08-08T00:00:02+00:00", "event": "dataset_access", "dataset": "hans", "split": "evaluation", "purpose": "final"},
                ],
            )
            _rehash_status(root / "seed_42" / "standard_lora")
            with self.assertRaisesRegex(ValueError, "manifest_identity_only"):
                validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

    def test_validator_rejects_malformed_method_audit_marker_evaluation_sequence_and_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = [
                ("marker-event", 0, "event", "dataset_access", "final_evaluation_start"),
                ("evaluation-purpose", 1, "purpose", "boundary", "official HANS evaluation"),
                ("marker-schema", 0, "extra", "spoof", "marker schema"),
                ("sequence", 1, "sequence", 0, "sequence"),
                ("timestamp", 1, "timestamp", "2026-08-07T00:00:00+00:00", "timestamp"),
            ]
            for name, index, field, value, message in cases:
                root = _create_smoke_root(Path(tmp) / name)
                run = root / "seed_42" / "standard_lora"
                rows = [json.loads(line) for line in (run / "data_access.jsonl").read_text(encoding="utf-8").splitlines()]
                rows[index][field] = value
                write_jsonl(run / "data_access.jsonl", rows)
                _rehash_status(run)
                with self.assertRaisesRegex(ValueError, message):
                    validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

    def test_validator_rejects_invalid_root_manifest_identity_audit_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(Path(tmp) / "root-audit")
            write_jsonl(
                root / "manifests" / "data_access.jsonl",
                [{"event": "dataset_access", "dataset": "hans", "split": "evaluation", "purpose": "final"}],
            )
            with self.assertRaisesRegex(ValueError, "manifest identity audit"):
                validate_smoke_root(root, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=Path(tmp) / "canonical_v1")

    def test_repeat_tolerance_is_inclusive_at_half_percentage_point(self):
        report = compare_metric_values(0.400, 0.405, tolerance=0.005)
        self.assertTrue(report["within_tolerance"])
        self.assertAlmostEqual(0.005, report["absolute_difference"])

    def test_repeat_comparison_uses_hans_non_entailment_and_requires_frozen_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary = _create_smoke_root(Path(tmp) / "primary", mnli_accuracy=0.80)
            repeat = _repeat_root(Path(tmp) / "repeat", mnli_accuracy=0.70)
            report = compare_a100_repeat(primary, repeat)
            self.assertTrue(report["primary_metric"]["within_tolerance"])
            self.assertEqual(0.80, report["mnli_diagnostic"]["primary"])
            changed = json.loads((repeat / "commands.json").read_text(encoding="utf-8"))
            changed["gpu_name"] = "NVIDIA T4"
            write_json(repeat / "commands.json", changed)
            with self.assertRaisesRegex(ValueError, "gpu"):
                compare_a100_repeat(primary, repeat)

    def test_repeat_comparison_rejects_a_manifest_not_bound_to_its_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary = _create_smoke_root(Path(tmp) / "primary")
            repeat = _repeat_root(Path(tmp) / "repeat")
            manifest_path = repeat / "seed_42" / "full_sr" / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["protocol_sha256"] = HASH_A
            write_json(manifest_path, manifest)
            _rehash_status(repeat / "seed_42" / "full_sr")

            with self.assertRaisesRegex(ValueError, "does not bind"):
                compare_a100_repeat(primary, repeat)

    def test_repeat_comparison_requires_a100_root_identities_and_full_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary = _create_smoke_root(Path(tmp) / "primary")
            wrong_mode = _repeat_root(Path(tmp) / "mode", mode="primary")
            with self.assertRaisesRegex(ValueError, "mode"):
                compare_a100_repeat(primary, wrong_mode)

            wrong_matrix = _repeat_root(Path(tmp) / "matrix", conditions=PRIMARY_CONDITIONS)
            with self.assertRaisesRegex(ValueError, "condition"):
                compare_a100_repeat(primary, wrong_matrix)

            wrong_environment = _repeat_root(Path(tmp) / "environment", smoke_environment="local_rtx5080")
            with self.assertRaisesRegex(ValueError, "environment"):
                compare_a100_repeat(primary, wrong_environment)

            wrong_gpu = _repeat_root(Path(tmp) / "gpu", environment="NVIDIA T4")
            with self.assertRaisesRegex(ValueError, "A100"):
                compare_a100_repeat(primary, wrong_gpu)

            tampered = _repeat_root(Path(tmp) / "tampered")
            predictions = tampered / "seed_42" / "full_sr" / "hans_predictions.jsonl"
            rows = json.loads("[" + ",".join(predictions.read_text(encoding="utf-8").splitlines()) + "]")
            rows[0]["predicted_label"] = "non-entailment"
            write_jsonl(predictions, rows)
            _rehash_status(tampered / "seed_42" / "full_sr")
            with self.assertRaisesRegex(ValueError, "HANS metrics"):
                compare_a100_repeat(primary, tampered)

    def test_repeat_comparison_rejects_missing_commit_and_seed_on_either_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary = _create_smoke_root(Path(tmp) / "primary")
            repeat = _repeat_root(Path(tmp) / "repeat")
            primary_manifest = primary / "seed_42" / "standard_lora" / "run_manifest.json"
            document = json.loads(primary_manifest.read_text(encoding="utf-8"))
            document["git"].pop("commit")
            write_json(primary_manifest, document)
            _rehash_status(primary / "seed_42" / "standard_lora")
            with self.assertRaisesRegex(ValueError, "git.commit"):
                compare_a100_repeat(primary, repeat)

            primary = _create_smoke_root(Path(tmp) / "primary-seed")
            repeat = _repeat_root(Path(tmp) / "repeat-seed")
            repeat_manifest = repeat / "seed_42" / "full_sr" / "run_manifest.json"
            document = json.loads(repeat_manifest.read_text(encoding="utf-8"))
            document["training_seed"] = None
            write_json(repeat_manifest, document)
            _rehash_status(repeat / "seed_42" / "full_sr")
            with self.assertRaisesRegex(ValueError, "training_seed"):
                compare_a100_repeat(primary, repeat)

    def test_cli_writes_reports_without_importing_ml_dependencies_for_help(self):
        help_result = subprocess.run(
            [sys.executable, "validate_stage2_smoke.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, help_result.returncode, help_result.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            root = _create_smoke_root(tmp)
            import validate_stage2_smoke

            profile, anchors = _controlled_contract()
            with profile, anchors:
                result = validate_stage2_smoke.main(
                    ["--root", str(root), "--conditions", *PRIMARY_CONDITIONS]
                )
            self.assertEqual(0, result)
            self.assertTrue((root / "stage2_validation.json").is_file())
            self.assertTrue((root / "stage2_validation.md").is_file())

    def test_cli_includes_repeat_comparison_in_written_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary = _create_smoke_root(Path(tmp) / "primary")
            repeat = _repeat_root(Path(tmp) / "repeat")
            import validate_stage2_smoke

            profile, anchors = _controlled_contract()
            with profile, anchors:
                result = validate_stage2_smoke.main(
                    [
                        "--root", str(primary), "--conditions", *PRIMARY_CONDITIONS,
                        "--compare-repeat", str(repeat),
                    ]
                )
            self.assertEqual(0, result)
            report = json.loads((primary / "stage2_validation.json").read_text(encoding="utf-8"))
            self.assertEqual("pass", report["repeat_comparison"]["state"])
            self.assertIn("A100 Repeat", (primary / "stage2_validation.md").read_text(encoding="utf-8"))

    def test_cli_writes_failed_repeat_report_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary = _create_smoke_root(Path(tmp) / "primary")
            repeat = _repeat_root(Path(tmp) / "repeat")
            run = repeat / "seed_42" / "full_sr"
            predictions = run / "hans_predictions.jsonl"
            rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
            rows[1]["predicted_label"] = "entailment"
            write_jsonl(predictions, rows)
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            metrics["final"]["hans"] = aggregate_hans_predictions(rows)
            write_json(run / "metrics.json", metrics)
            _rehash_status(run)
            import validate_stage2_smoke

            profile, anchors = _controlled_contract()
            with profile, anchors:
                result = validate_stage2_smoke.main(
                    [
                        "--root", str(primary), "--conditions", *PRIMARY_CONDITIONS,
                        "--compare-repeat", str(repeat),
                    ]
                )
            self.assertEqual(1, result)
            report = json.loads((primary / "stage2_validation.json").read_text(encoding="utf-8"))
            self.assertEqual("fail", report["state"])
            self.assertEqual("fail", report["repeat_comparison"]["state"])
            self.assertTrue((primary / "stage2_validation.md").is_file())


if __name__ == "__main__":
    unittest.main()
