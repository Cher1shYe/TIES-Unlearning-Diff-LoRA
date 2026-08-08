"""Pure data-selection and HANS partition contracts for canonical v1."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any, Callable, Iterable, Mapping, Sequence


HansRecord = dict[str, Any]
RngFactory = Callable[[int], Any]
_HANS_SOURCE_PREFIXES = {
    "train": "hans_train::",
    "evaluation": "hans_evaluation::",
}
_HANS_LOCAL_ID_RE = re.compile(r"^ex(?:0|[1-9][0-9]*)$")
_HANS_CONTENT_FIELDS = ("gold_label", "premise", "hypothesis", "heuristic", "subcase")
_HANS_CONTENT_ALGORITHM = "sha256_canonical_json_utf8_v1"
_HANS_SPLIT_ALGORITHM = "source_local_id_sort_numpy_default_rng_per_stratum_v1"
_HANS_SELECTION_ALGORITHM = "sha256_seed_nul_source_local_id_stratified_round_robin_v1"
_HANS_ARTIFACT_TRANSFORM = "hans_evaluation::<source_local_pair_id>"

_HANS_SOURCE_FIELDS = (
    "gold_label",
    "sentence1_binary_parse",
    "sentence2_binary_parse",
    "sentence1_parse",
    "sentence2_parse",
    "sentence1",
    "sentence2",
    "pairID",
    "heuristic",
    "subcase",
    "template",
)
_HANS_SOURCE_ORDERING = "numeric_pair_id_then_raw_pair_id_v1"

HANS_OFFICIAL_ANCHORS_V2 = {
    "schema_version": "hans_official_semantic_anchors_v2",
    "derivation_algorithm": "official_tsv_exact_11_fields_numeric_pair_id_canonical_json_utf8_sha256_v1",
    "informational_raw_file_sha256": {
        "train": "49245bd5fdb0b185dcbfbf48f0f16513c62ad5bc9fad0b8800dc48d6818ee5cf",
        "evaluation": "c55b62feef9913070e88f38938dc2492018c945ac81f70139346472494124e79",
    },
    "source_integrity": {
        "train": {
            "count": 30000,
            "records_sha256": "841ffee28e0310f1f95d692a534f362a8a171a69d7f659ec3ed07a4205840cf5",
        },
        "evaluation": {
            "count": 30000,
            "records_sha256": "5d170c471cde96e61c24d640cb50652bf7c594c4800e40d7ebf8133ec7d5df6b",
        },
    },
    "split_checksum": "f2d240a1709481a8c37c0721104697469383e9ad49ed22496f9265633c9f129a",
    "partitions": {
        "build": {
            "count": 24000,
            "source_pair_ids_sha256": "cf37089c0550410096e718e8c5a8f996650afe0afcef91c2660f27ce43560eab",
            "qualified_ids_sha256": "cd8e7f745cc93703a71bd9c62b36647a8c3fe04596528f1b5a4be002ebe74bcc",
            "content_sha256_ordered_checksum": "3eea3fea671f926bcc3975dd01595d70842e37ae3ede45b8324d37b2a6dd6de1",
            "source_id_content_joint_checksum": "c74c88f8edcfd138b99b21571e45fc0520c460eba194edc75dfd7da20f5bde5c",
        },
        "dev": {
            "count": 6000,
            "source_pair_ids_sha256": "53f63723dfe459bfbd1b1ffe045af5d61beb3181ba37a379dfa96f92e08c1ba8",
            "qualified_ids_sha256": "f2d2fd8a0c43d8d1c449ab4ed990eedf7d3600afbdd38c0ac8ec0ccde07887ce",
            "content_sha256_ordered_checksum": "d949b61e6d75889de00d1266ee73633bde9181e23be110005cb106c4328aa8d7",
            "source_id_content_joint_checksum": "48f62af7a35125b87195fa0c6590918bf472de9b5b1315e303d9e10fd2ac214b",
        },
        "evaluation": {
            "count": 30000,
            "source_pair_ids_sha256": "495a55ae9bad6e464684b3b205ae6b591f5abb424dd5c0fdb98f2ad3db63be70",
            "qualified_ids_sha256": "0a6d3beb1d2f182f2c7decd199bd7ca854baaf5fb1acf322297257d20bcf75a0",
            "content_sha256_ordered_checksum": "2b9b28d55b07245e3040aa0bcbcd14cd4a9598e4b55202b759d7f515aeb1cbfa",
            "source_id_content_joint_checksum": "24eea2e3cb75c2910de142154803e8bdd98fcba6f12c61293de997faccff43ef",
        },
    },
    "selection_384": {
        "count": 384,
        "selected_source_pair_ids_sha256": "afa0aea6a159eb3b4f68077da8a665e1c277d47815d01398633af5cfe8e53b51",
        "selected_artifact_ids_sha256": "2dad8b0ee67b7c3cbc8a621826c64cd7cb87bf78965a93e58c4519f092bd07c0",
        "source_to_artifact_mapping_sha256": "d755522b3f3e492d3543400f5fe07fd2ba354f62e89525f7170e3432cc178b96",
    },
    "selection_full": {
        "count": 30000,
        "selected_source_pair_ids_sha256": "495a55ae9bad6e464684b3b205ae6b591f5abb424dd5c0fdb98f2ad3db63be70",
        "selected_artifact_ids_sha256": "0a6d3beb1d2f182f2c7decd199bd7ca854baaf5fb1acf322297257d20bcf75a0",
        "source_to_artifact_mapping_sha256": "fe500cf664524e54dfe55702418b03d717e5a25468f673b36fad5cdf5f82f2d9",
    },
}


def sample_dataset(dataset: Any, count: int, seed: int) -> Any:
    """Deterministically sample a duck-typed Hugging Face dataset."""
    if count and count < len(dataset):
        return dataset.shuffle(seed=seed).select(range(count))
    return dataset


def dataset_row_ids(dataset: Iterable[Mapping[str, Any]]) -> list[Any]:
    """Return stable source IDs without deriving them from sampled row order."""
    ids = []
    for position, record in enumerate(dataset):
        if "idx" in record:
            row_id = record["idx"]
        elif "row_id" in record:
            row_id = record["row_id"]
        else:
            raise ValueError(
                f"dataset row {position} has no stable 'idx' or 'row_id' field"
            )
        ids.append(row_id)
    if len(set(ids)) != len(ids):
        raise ValueError("dataset contains duplicate stable row IDs")
    return ids


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_record_id(
    record: Mapping[str, Any], preferred_fields: Sequence[str] = ()
) -> str:
    """Return a source identity when available, otherwise a canonical record hash."""
    for field in preferred_fields:
        value = record.get(field)
        if value is not None:
            stable_id = str(value)
            if stable_id:
                return stable_id
    return sha256(_canonical_bytes(dict(record))).hexdigest()


def qualify_hans_pair_id(pair_id: Any, physical_source_partition: str) -> str:
    """Return the source-file-qualified identity for one official HANS row."""
    if physical_source_partition not in _HANS_SOURCE_PREFIXES:
        raise ValueError("HANS physical source partition must be 'train' or 'evaluation'")
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("HANS pairID must be a canonical source-local exN identity")
    expected_prefix = _HANS_SOURCE_PREFIXES[physical_source_partition]
    matching_prefix = next(
        (prefix for prefix in _HANS_SOURCE_PREFIXES.values() if pair_id.startswith(prefix)),
        None,
    )
    if matching_prefix is not None:
        suffix = pair_id[len(matching_prefix):]
        if matching_prefix != expected_prefix:
            raise ValueError("HANS pairID does not match its physical source partition")
        if _HANS_LOCAL_ID_RE.fullmatch(suffix) is None:
            raise ValueError("HANS pairID must contain exactly one prefix and a canonical source-local exN identity")
        return pair_id
    if _HANS_LOCAL_ID_RE.fullmatch(pair_id) is None:
        raise ValueError("HANS pairID must be a canonical source-local exN identity")
    return f"{expected_prefix}{pair_id}"


def deterministic_cap_records(
    records: Sequence[Mapping[str, Any]],
    limit: int,
    seed: int,
    strata_fields: Sequence[str] = (),
    preferred_fields: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select a seeded, order-independent cap with optional round-robin strata."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    ranked_strata: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    seen_ids: set[str] = set()
    for source_record in records:
        record = dict(source_record)
        stable_id = stable_record_id(record, preferred_fields)
        if stable_id in seen_ids:
            raise ValueError(f"duplicate stable record ID: {stable_id}")
        seen_ids.add(stable_id)
        stratum = _canonical_bytes([record.get(field) for field in strata_fields]).decode("utf-8")
        ranked_strata.setdefault(stratum, []).append((stable_id, record))

    for candidates in ranked_strata.values():
        candidates.sort(
            key=lambda item: (
                sha256(f"{seed}\x00{item[0]}".encode("utf-8")).hexdigest(),
                item[0],
            )
        )

    selected: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    offsets = {stratum: 0 for stratum in ranked_strata}
    while len(selected) < limit:
        selected_this_round = False
        for stratum in sorted(ranked_strata):
            offset = offsets[stratum]
            candidates = ranked_strata[stratum]
            if offset >= len(candidates):
                continue
            stable_id, record = candidates[offset]
            offsets[stratum] = offset + 1
            selected.append(record)
            selected_ids.append(stable_id)
            selected_this_round = True
            if len(selected) == limit:
                break
        if not selected_this_round:
            break
    return selected, selected_ids


def select_hans_evaluation_records(
    records: Sequence[Mapping[str, Any]],
    limit: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select HANS evaluation rows by raw exN while returning artifact identities."""
    rows = [dict(record) for record in records]
    for row in rows:
        local_id = row.get("pairID")
        expected = qualify_hans_pair_id(local_id, "evaluation")
        if row.get("canonical_pair_id") != expected:
            raise ValueError("HANS evaluation row lacks its qualified artifact identity")
    if limit is None:
        selected = rows
    else:
        selected, _ = deterministic_cap_records(
            rows,
            limit,
            seed,
            ("gold_label", "heuristic", "subcase"),
            ("pairID",),
        )
    artifact_ids = [str(row["canonical_pair_id"]) for row in selected]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("HANS evaluation selection contains duplicate artifact identities")
    return selected, artifact_ids


def _record_value(record: Mapping[str, Any], canonical: str, source: str) -> Any:
    if canonical in record:
        return record[canonical]
    if source in record:
        return record[source]
    raise ValueError(f"HANS record is missing required field {source!r}")


def _hans_content_identity(record: Mapping[str, Any]) -> str:
    """Hash exact semantic HANS content without including its source-local pairID."""
    content = {
        "gold_label": str(_record_value(record, "gold_label", "gold_label")),
        "premise": str(_record_value(record, "premise", "sentence1")),
        "hypothesis": str(_record_value(record, "hypothesis", "sentence2")),
        "heuristic": str(_record_value(record, "heuristic", "heuristic")),
        "subcase": str(_record_value(record, "subcase", "subcase")),
    }
    return sha256(_canonical_bytes(content)).hexdigest()


def _ordered_checksum(values: Sequence[Any]) -> str:
    return sha256(_canonical_bytes(list(values))).hexdigest()


def build_hans_source_integrity_manifest(
    train_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Hash every official parsed HANS field in canonical raw-ID order."""
    sources: dict[str, dict[str, Any]] = {}
    expected_fields = set(_HANS_SOURCE_FIELDS)
    for physical_source, source_records in (
        ("train", train_records),
        ("evaluation", evaluation_records),
    ):
        canonical_records = []
        seen_pair_ids: set[str] = set()
        for position, source_record in enumerate(source_records):
            record = dict(source_record)
            record_keys = set(record)
            if record_keys not in (
                expected_fields,
                {*expected_fields, "canonical_pair_id"},
            ):
                raise ValueError(
                    "HANS source integrity records must contain exactly the 11 official fields plus only the derived canonical_pair_id"
                )
            if not all(isinstance(record[field], str) for field in _HANS_SOURCE_FIELDS):
                raise ValueError("HANS source integrity official fields must be parsed strings")
            pair_id = record["pairID"]
            qualified_id = qualify_hans_pair_id(pair_id, physical_source)
            if "canonical_pair_id" in record and record["canonical_pair_id"] != qualified_id:
                raise ValueError(
                    f"HANS source integrity {physical_source} record {position} has an invalid derived canonical_pair_id"
                )
            if pair_id in seen_pair_ids:
                raise ValueError(
                    f"HANS source integrity {physical_source} has duplicate source-local pairID: {pair_id}"
                )
            seen_pair_ids.add(pair_id)
            canonical_records.append(
                {field: record[field] for field in _HANS_SOURCE_FIELDS}
            )
        if not canonical_records:
            raise ValueError(f"HANS source integrity {physical_source} records are empty")
        canonical_records.sort(
            key=lambda record: (int(record["pairID"][2:]), record["pairID"])
        )
        sources[physical_source] = {
            "count": len(canonical_records),
            "records_sha256": _ordered_checksum(canonical_records),
        }
    return {
        "schema_version": "hans_source_integrity_v1",
        "algorithm": _HANS_CONTENT_ALGORITHM,
        "fields": list(_HANS_SOURCE_FIELDS),
        "ordering": _HANS_SOURCE_ORDERING,
        "sources": sources,
    }


def build_hans_content_integrity_manifest(
    records_by_partition: Mapping[str, Sequence[Mapping[str, Any]]],
    ids_by_partition: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Build fail-closed ordered HANS content evidence bound to qualified IDs."""
    names = ("build", "dev", "evaluation")
    if set(records_by_partition) != set(names) or set(ids_by_partition) != set(names):
        raise ValueError("HANS content integrity requires build, dev, and evaluation")
    partitions: dict[str, dict[str, Any]] = {}
    content_sets: dict[str, set[str]] = {}
    for name in names:
        records = list(records_by_partition[name])
        identities = [str(value) for value in ids_by_partition[name]]
        if len(records) != len(identities) or not records:
            raise ValueError(f"HANS content {name} records do not align with identities")
        physical_source = "evaluation" if name == "evaluation" else "train"
        for record, identity in zip(records, identities):
            canonical_pair_id = record.get("canonical_pair_id")
            if (
                not isinstance(canonical_pair_id, str)
                or canonical_pair_id != identity
                or qualify_hans_pair_id(identity, physical_source) != identity
            ):
                raise ValueError(
                    f"HANS content {name} canonical_pair_id does not equal its parallel qualified identity"
                )
        hashes = [_hans_content_identity(record) for record in records]
        duplicate_count = len(hashes) - len(set(hashes))
        partitions[name] = {
            "count": len(hashes),
            "content_sha256": hashes,
            "content_sha256_ordered_checksum": _ordered_checksum(hashes),
            "source_id_content_joint_checksum": _ordered_checksum(
                [[identity, content_hash] for identity, content_hash in zip(identities, hashes)]
            ),
            "duplicate_content_count": duplicate_count,
        }
        content_sets[name] = set(hashes)
    overlap_counts = {
        "build_dev": len(content_sets["build"] & content_sets["dev"]),
        "build_evaluation": len(content_sets["build"] & content_sets["evaluation"]),
        "dev_evaluation": len(content_sets["dev"] & content_sets["evaluation"]),
    }
    if any(entry["duplicate_content_count"] for entry in partitions.values()) or any(overlap_counts.values()):
        raise ValueError("HANS content integrity detected duplicate or overlapping content")
    return {
        "schema_version": "hans_content_integrity_v1",
        "algorithm": _HANS_CONTENT_ALGORITHM,
        "fields": list(_HANS_CONTENT_FIELDS),
        "excludes_pair_id": True,
        "partitions": partitions,
        "overlap_counts": overlap_counts,
    }


def _content_identities(
    records: Sequence[Mapping[str, Any]], name: str
) -> set[str]:
    identities = [_hans_content_identity(record) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate {name} content")
    return set(identities)


def validate_hans_content_integrity(
    build_records: Sequence[Mapping[str, Any]],
    dev_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
) -> None:
    """Reject exact-content duplicates and leakage independently of HANS pair IDs."""
    partitions = (
        ("HANS build", _content_identities(build_records, "HANS build")),
        ("HANS dev", _content_identities(dev_records, "HANS dev")),
        ("HANS evaluation", _content_identities(evaluation_records, "HANS evaluation")),
    )
    for left in range(len(partitions)):
        for right in range(left + 1, len(partitions)):
            overlap = partitions[left][1] & partitions[right][1]
            if overlap:
                label = f"{partitions[left][0].removeprefix('HANS ').lower()}/{partitions[right][0].removeprefix('HANS ').lower()}"
                preview = ", ".join(sorted(overlap)[:5])
                raise ValueError(f"HANS content {label} overlap detected: {preview}")


def validate_hans_source_integrity_manifest(
    integrity: Any,
    official_anchors: Mapping[str, Any] | None = None,
) -> None:
    """Validate parsed-record evidence and optionally pin official source anchors."""
    if (
        not isinstance(integrity, Mapping)
        or set(integrity) != {
            "schema_version", "algorithm", "fields", "ordering", "sources"
        }
        or integrity.get("schema_version") != "hans_source_integrity_v1"
        or integrity.get("algorithm") != _HANS_CONTENT_ALGORITHM
        or integrity.get("fields") != list(_HANS_SOURCE_FIELDS)
        or integrity.get("ordering") != _HANS_SOURCE_ORDERING
    ):
        raise ValueError("HANS source integrity declaration is invalid")
    sources = integrity.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {"train", "evaluation"}:
        raise ValueError("HANS source integrity physical sources are invalid")
    for name in ("train", "evaluation"):
        source = sources[name]
        if (
            not isinstance(source, Mapping)
            or set(source) != {"count", "records_sha256"}
            or not isinstance(source.get("count"), int)
            or isinstance(source.get("count"), bool)
            or source.get("count", 0) <= 0
            or not isinstance(source.get("records_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", source["records_sha256"]) is None
        ):
            raise ValueError(f"HANS source integrity {name} evidence is invalid")
    if official_anchors is not None:
        if (
            official_anchors.get("schema_version")
            != "hans_official_semantic_anchors_v2"
            or official_anchors.get("source_integrity") != sources
        ):
            raise ValueError("official HANS source integrity anchor mismatch")


def validate_hans_manifest_identities(
    hans: Mapping[str, Any],
    *,
    expected_seed: int | None = None,
    expected_selection_cap: int | None | object = ...,
    official_anchors: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate HANS manifest arrays/checksums and return ordered evaluation IDs."""
    expected_sources = {
        "build": "train",
        "dev": "train",
        "evaluation": "evaluation",
    }
    if not isinstance(hans, Mapping) or not set(expected_sources).issubset(hans):
        raise ValueError("HANS manifest must contain build, dev, and evaluation identities")
    full_partitions: dict[str, list[str]] = {}
    for name, physical_source in expected_sources.items():
        entry = hans[name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"HANS manifest {name} identity entry is invalid")
        arrays: dict[str, list[str]] = {}
        for key in ("full_ids", "selected_ids"):
            values = entry.get(key)
            identities_are_canonical = False
            if isinstance(values, list) and all(isinstance(value, str) for value in values):
                try:
                    identities_are_canonical = all(
                        qualify_hans_pair_id(value, physical_source) == value
                        for value in values
                    )
                except ValueError:
                    identities_are_canonical = False
            if (
                not isinstance(values, list)
                or not values
                or not identities_are_canonical
                or len(values) != len(set(values))
            ):
                raise ValueError(f"HANS manifest {name} {key} identities are malformed, duplicated, or in the wrong namespace")
            arrays[key] = values
            count_key = "full_count" if key == "full_ids" else "selected_count"
            if entry.get(count_key) != len(values):
                raise ValueError(f"HANS manifest {name} {key} count is invalid")
            checksum_key = f"{key}_sha256"
            expected_checksum = sha256(_canonical_bytes(values)).hexdigest()
            if entry.get(checksum_key) != expected_checksum:
                raise ValueError(f"HANS manifest {name} {key} checksum is invalid")
        if not set(arrays["selected_ids"]).issubset(arrays["full_ids"]):
            raise ValueError(f"HANS manifest {name} selected_ids are not full members")
        full_partitions[name] = arrays["full_ids"]
    validate_hans_disjointness(
        full_partitions["build"],
        full_partitions["dev"],
        full_partitions["evaluation"],
    )
    if set(hans) != {
        *expected_sources,
        "split_integrity",
        "content_integrity",
        "selection_integrity",
    }:
        raise ValueError("HANS split/content/selection integrity objects are missing or unexpected")
    _validate_hans_split_integrity_manifest(
        hans["split_integrity"],
        {name: full_partitions[name] for name in ("build", "dev")},
        expected_seed=expected_seed,
    )
    _validate_hans_content_integrity_manifest(
        hans["content_integrity"],
        full_partitions,
    )
    _validate_hans_selection_integrity_manifest(
        hans["selection_integrity"],
        list(hans["evaluation"]["selected_ids"]),
        expected_seed=expected_seed,
        expected_cap=expected_selection_cap,
    )
    if official_anchors is not None:
        _validate_official_hans_anchors(
            hans,
            official_anchors,
            expected_selection_cap=expected_selection_cap,
        )
    return list(hans["evaluation"]["selected_ids"])


def _validate_hans_split_integrity_manifest(
    integrity: Any,
    qualified_ids: Mapping[str, Sequence[str]],
    *,
    expected_seed: int | None,
) -> None:
    keys = {
        "schema_version", "seed", "split_algorithm", "checksum_algorithm",
        "build_count", "dev_count", "build_source_pair_ids",
        "dev_source_pair_ids", "small_strata", "split_checksum",
    }
    if (
        not isinstance(integrity, Mapping)
        or set(integrity) != keys
        or integrity.get("schema_version") != "hans_split_integrity_v1"
        or integrity.get("split_algorithm") != _HANS_SPLIT_ALGORITHM
        or integrity.get("checksum_algorithm") != _HANS_CONTENT_ALGORITHM
        or not isinstance(integrity.get("seed"), int)
        or isinstance(integrity.get("seed"), bool)
        or integrity.get("seed", -1) < 0
        or (expected_seed is not None and integrity.get("seed") != expected_seed)
    ):
        raise ValueError("HANS split integrity declaration is invalid")
    raw: dict[str, list[str]] = {}
    for name in ("build", "dev"):
        values = integrity.get(f"{name}_source_pair_ids")
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and _HANS_LOCAL_ID_RE.fullmatch(value) for value in values)
            or len(values) != len(set(values))
            or integrity.get(f"{name}_count") != len(values)
            or [qualify_hans_pair_id(value, "train") for value in values]
            != list(qualified_ids[name])
        ):
            raise ValueError(f"HANS split integrity {name} raw/artifact binding is invalid")
        raw[name] = values
    if set(raw["build"]) & set(raw["dev"]):
        raise ValueError("HANS split integrity build/dev raw identities are not disjoint")
    small = integrity.get("small_strata")
    small_keys = {
        "gold_label", "heuristic", "subcase", "count",
        "build_source_pair_ids", "dev_source_pair_ids",
    }
    if not isinstance(small, list):
        raise ValueError("HANS split integrity small strata are invalid")
    for item in small:
        if not isinstance(item, Mapping) or set(item) != small_keys:
            raise ValueError("HANS split integrity small stratum schema is invalid")
        build_ids = item["build_source_pair_ids"]
        dev_ids = item["dev_source_pair_ids"]
        if (
            not isinstance(build_ids, list)
            or not isinstance(dev_ids, list)
            or not all(isinstance(value, str) and _HANS_LOCAL_ID_RE.fullmatch(value) for value in [*build_ids, *dev_ids])
            or item.get("count") != len(build_ids) + len(dev_ids)
            or item.get("count", 5) >= 5
            or not set(build_ids).issubset(raw["build"])
            or not set(dev_ids).issubset(raw["dev"])
        ):
            raise ValueError("HANS split integrity small stratum membership is invalid")
    checksum_payload = {
        "schema_version": "hans_split_v1",
        "hans_split_seed": integrity["seed"],
        "build_pair_ids": raw["build"],
        "dev_pair_ids": raw["dev"],
        "small_strata": [
            {
                "gold_label": item["gold_label"],
                "heuristic": item["heuristic"],
                "subcase": item["subcase"],
                "count": item["count"],
                "build_pair_ids": item["build_source_pair_ids"],
                "dev_pair_ids": item["dev_source_pair_ids"],
            }
            for item in small
        ],
    }
    if integrity.get("split_checksum") != sha256(_canonical_bytes(checksum_payload)).hexdigest():
        raise ValueError("HANS split integrity checksum is invalid")


def _validate_hans_selection_integrity_manifest(
    integrity: Any,
    selected_artifact_ids: Sequence[str],
    *,
    expected_seed: int | None,
    expected_cap: int | None | object,
) -> None:
    payload_keys = {
        "schema_version", "ranking_key", "ranking_algorithm", "artifact_transform",
        "selected_order", "seed", "strata_fields", "cap", "selected_count",
        "selected_source_pair_ids", "selected_source_pair_ids_sha256",
        "selected_artifact_ids_sha256", "source_to_artifact_mapping_sha256",
    }
    if not isinstance(integrity, Mapping) or set(integrity) != {*payload_keys, "integrity_checksum"}:
        raise ValueError("HANS selection integrity schema is invalid")
    cap = integrity.get("cap")
    if (
        integrity.get("schema_version") != "hans_selection_integrity_v1"
        or integrity.get("ranking_key") != "source_local_pair_id"
        or integrity.get("ranking_algorithm") != _HANS_SELECTION_ALGORITHM
        or integrity.get("artifact_transform") != _HANS_ARTIFACT_TRANSFORM
        or integrity.get("selected_order") != ("source_order" if cap is None else "ranked_cap_order")
        or integrity.get("strata_fields") != ["gold_label", "heuristic", "subcase"]
        or not isinstance(integrity.get("seed"), int)
        or isinstance(integrity.get("seed"), bool)
        or integrity.get("seed", -1) < 0
        or (expected_seed is not None and integrity.get("seed") != expected_seed)
        or (cap is not None and (not isinstance(cap, int) or isinstance(cap, bool) or cap < 0))
        or (expected_cap is not ... and cap != expected_cap)
    ):
        raise ValueError("HANS selection integrity declaration is invalid")
    source_ids = integrity.get("selected_source_pair_ids")
    artifact_ids = list(selected_artifact_ids)
    if (
        not isinstance(source_ids, list)
        or not source_ids
        or not all(isinstance(value, str) and _HANS_LOCAL_ID_RE.fullmatch(value) for value in source_ids)
        or len(source_ids) != len(set(source_ids))
        or integrity.get("selected_count") != len(source_ids)
        or len(source_ids) != len(artifact_ids)
        or (cap is not None and len(source_ids) != cap)
        or [qualify_hans_pair_id(value, "evaluation") for value in source_ids] != artifact_ids
        or integrity.get("selected_source_pair_ids_sha256") != _ordered_checksum(source_ids)
        or integrity.get("selected_artifact_ids_sha256") != _ordered_checksum(artifact_ids)
        or integrity.get("source_to_artifact_mapping_sha256")
        != _ordered_checksum([[raw, artifact] for raw, artifact in zip(source_ids, artifact_ids)])
    ):
        raise ValueError("HANS selection integrity raw/artifact mapping is invalid")
    payload = {key: integrity[key] for key in payload_keys}
    if integrity.get("integrity_checksum") != _ordered_checksum(
        [[key, payload[key]] for key in sorted(payload)]
    ):
        raise ValueError("HANS selection integrity checksum is invalid")


def _validate_official_hans_anchors(
    hans: Mapping[str, Any],
    anchors: Mapping[str, Any],
    *,
    expected_selection_cap: int | None | object,
) -> None:
    if anchors.get("schema_version") != "hans_official_semantic_anchors_v1":
        raise ValueError("official HANS semantic anchors schema is invalid")
    split = hans["split_integrity"]
    if split.get("split_checksum") != anchors.get("split_checksum"):
        raise ValueError("official HANS split semantic anchor mismatch")
    for name in ("build", "dev", "evaluation"):
        expected = anchors["partitions"][name]
        identity = hans[name]
        content = hans["content_integrity"]["partitions"][name]
        if (
            identity.get("full_count") != expected["count"]
            or identity.get("full_ids_sha256") != expected["qualified_ids_sha256"]
            or content.get("count") != expected["count"]
            or content.get("content_sha256_ordered_checksum")
            != expected["content_sha256_ordered_checksum"]
            or content.get("source_id_content_joint_checksum")
            != expected["source_id_content_joint_checksum"]
        ):
            raise ValueError(f"official HANS {name} content/identity semantic anchor mismatch")
    for name in ("build", "dev"):
        if _ordered_checksum(split[f"{name}_source_pair_ids"]) != anchors["partitions"][name]["source_pair_ids_sha256"]:
            raise ValueError(f"official HANS {name} source membership anchor mismatch")
    anchor_name = (
        "selection_full"
        if expected_selection_cap is None
        else f"selection_{expected_selection_cap}"
    )
    if anchor_name not in anchors:
        raise ValueError("official HANS selection cap is not frozen")
    expected_selection = anchors[anchor_name]
    selection = hans["selection_integrity"]
    actual_selection = {
        "count": selection.get("selected_count"),
        "selected_source_pair_ids_sha256": selection.get("selected_source_pair_ids_sha256"),
        "selected_artifact_ids_sha256": selection.get("selected_artifact_ids_sha256"),
        "source_to_artifact_mapping_sha256": selection.get("source_to_artifact_mapping_sha256"),
    }
    if actual_selection != expected_selection:
        raise ValueError("official HANS selection semantic anchor mismatch")


def _validate_hans_content_integrity_manifest(
    integrity: Any,
    ids_by_partition: Mapping[str, Sequence[str]],
) -> None:
    expected_top = {
        "schema_version", "algorithm", "fields", "excludes_pair_id",
        "partitions", "overlap_counts",
    }
    if (
        not isinstance(integrity, Mapping)
        or set(integrity) != expected_top
        or integrity.get("schema_version") != "hans_content_integrity_v1"
        or integrity.get("algorithm") != _HANS_CONTENT_ALGORITHM
        or integrity.get("fields") != list(_HANS_CONTENT_FIELDS)
        or integrity.get("excludes_pair_id") is not True
    ):
        raise ValueError("HANS content integrity declaration is invalid")
    partitions = integrity.get("partitions")
    if not isinstance(partitions, Mapping) or set(partitions) != set(ids_by_partition):
        raise ValueError("HANS content integrity partitions are invalid")
    content_sets: dict[str, set[str]] = {}
    entry_keys = {
        "count", "content_sha256", "content_sha256_ordered_checksum",
        "source_id_content_joint_checksum", "duplicate_content_count",
    }
    for name in ("build", "dev", "evaluation"):
        entry = partitions[name]
        ids = list(ids_by_partition[name])
        if not isinstance(entry, Mapping) or set(entry) != entry_keys:
            raise ValueError(f"HANS content {name} entry is invalid")
        hashes = entry.get("content_sha256")
        if (
            not isinstance(hashes, list)
            or entry.get("count") != len(ids)
            or len(hashes) != len(ids)
            or not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes)
        ):
            raise ValueError(f"HANS content {name} hashes/count are invalid")
        if entry.get("content_sha256_ordered_checksum") != _ordered_checksum(hashes):
            raise ValueError(f"HANS content {name} ordered checksum is invalid")
        joint = [[identity, content_hash] for identity, content_hash in zip(ids, hashes)]
        if entry.get("source_id_content_joint_checksum") != _ordered_checksum(joint):
            raise ValueError(f"HANS content {name} source-ID/content binding is invalid")
        duplicate_count = len(hashes) - len(set(hashes))
        if entry.get("duplicate_content_count") != duplicate_count or duplicate_count != 0:
            raise ValueError(f"HANS content {name} duplicate count is not zero")
        content_sets[name] = set(hashes)
    actual_overlap = {
        "build_dev": len(content_sets["build"] & content_sets["dev"]),
        "build_evaluation": len(content_sets["build"] & content_sets["evaluation"]),
        "dev_evaluation": len(content_sets["dev"] & content_sets["evaluation"]),
    }
    if integrity.get("overlap_counts") != actual_overlap or any(actual_overlap.values()):
        raise ValueError("HANS content overlap counts are invalid or nonzero")


def _hans_artifact_id(record: Mapping[str, Any]) -> str:
    value = record.get("canonical_pair_id")
    if value is None:
        value = _record_value(record, "pair_id", "pairID")
    artifact_id = str(value)
    if not artifact_id:
        raise ValueError("HANS artifact pair ID must not be empty")
    return artifact_id


def _normalized_hans_record(
    record: Mapping[str, Any],
) -> tuple[str, str, tuple[str, str, str]]:
    pair_id = str(_record_value(record, "source_pair_id", "pairID"))
    artifact_id = _hans_artifact_id(record)
    gold_label = str(_record_value(record, "gold_label", "gold_label"))
    heuristic = str(_record_value(record, "heuristic", "heuristic"))
    subcase = str(_record_value(record, "subcase", "subcase"))
    if not pair_id:
        raise ValueError("HANS pairID must not be empty")
    return pair_id, artifact_id, (gold_label, heuristic, subcase)


@dataclass(frozen=True)
class HansSplit:
    seed: int
    build_records: tuple[HansRecord, ...]
    dev_records: tuple[HansRecord, ...]
    small_strata: tuple[dict[str, Any], ...]
    checksum: str

    @property
    def build_pair_ids(self) -> list[str]:
        return [_hans_artifact_id(row) for row in self.build_records]

    @property
    def dev_pair_ids(self) -> list[str]:
        return [_hans_artifact_id(row) for row in self.dev_records]

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "hans_split_v1",
            "hans_split_seed": self.seed,
            "build_count": len(self.build_records),
            "dev_count": len(self.dev_records),
            "build_pair_ids": self.build_pair_ids,
            "dev_pair_ids": self.dev_pair_ids,
            "small_strata": [dict(item) for item in self.small_strata],
            "split_checksum": self.checksum,
        }


def build_hans_split_integrity(split: HansSplit) -> dict[str, Any]:
    """Persist raw-only build/dev split membership with the frozen checksum."""
    build_source_ids = [
        str(_record_value(row, "source_pair_id", "pairID"))
        for row in split.build_records
    ]
    dev_source_ids = [
        str(_record_value(row, "source_pair_id", "pairID"))
        for row in split.dev_records
    ]
    for name, records, source_ids in (
        ("build", split.build_records, build_source_ids),
        ("dev", split.dev_records, dev_source_ids),
    ):
        for record, source_id in zip(records, source_ids):
            if _HANS_LOCAL_ID_RE.fullmatch(source_id) is None:
                raise ValueError(f"HANS split {name} source-local identity is invalid")
            if record.get("canonical_pair_id") != qualify_hans_pair_id(source_id, "train"):
                raise ValueError(f"HANS split {name} source/artifact identity binding is invalid")
    small_strata = []
    for item in split.small_strata:
        build_ids = list(item.get("build_pair_ids", []))
        dev_ids = list(item.get("dev_pair_ids", []))
        if not all(
            isinstance(value, str) and _HANS_LOCAL_ID_RE.fullmatch(value)
            for value in [*build_ids, *dev_ids]
        ):
            raise ValueError("HANS split small-strata membership must use raw source-local IDs")
        small_strata.append(
            {
                "gold_label": item.get("gold_label"),
                "heuristic": item.get("heuristic"),
                "subcase": item.get("subcase"),
                "count": item.get("count"),
                "build_source_pair_ids": build_ids,
                "dev_source_pair_ids": dev_ids,
            }
        )
    checksum_payload = {
        "schema_version": "hans_split_v1",
        "hans_split_seed": split.seed,
        "build_pair_ids": build_source_ids,
        "dev_pair_ids": dev_source_ids,
        "small_strata": [
            {
                "gold_label": item["gold_label"],
                "heuristic": item["heuristic"],
                "subcase": item["subcase"],
                "count": item["count"],
                "build_pair_ids": item["build_source_pair_ids"],
                "dev_pair_ids": item["dev_source_pair_ids"],
            }
            for item in small_strata
        ],
    }
    checksum = sha256(_canonical_bytes(checksum_payload)).hexdigest()
    if checksum != split.checksum:
        raise ValueError("HANS split checksum does not match raw source-local membership")
    return {
        "schema_version": "hans_split_integrity_v1",
        "seed": split.seed,
        "split_algorithm": _HANS_SPLIT_ALGORITHM,
        "checksum_algorithm": _HANS_CONTENT_ALGORITHM,
        "build_count": len(build_source_ids),
        "dev_count": len(dev_source_ids),
        "build_source_pair_ids": build_source_ids,
        "dev_source_pair_ids": dev_source_ids,
        "small_strata": small_strata,
        "split_checksum": checksum,
    }


def build_hans_selection_integrity(
    selected_records: Sequence[Mapping[str, Any]],
    selected_artifact_ids: Sequence[str],
    *,
    limit: int | None,
    seed: int,
) -> dict[str, Any]:
    """Bind raw evaluation ranking IDs to ordered qualified artifact IDs."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("HANS selection seed must be a non-negative integer")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
    ):
        raise ValueError("HANS selection cap must be a non-negative integer or null")
    rows = [dict(record) for record in selected_records]
    artifact_ids = list(selected_artifact_ids)
    if not rows or len(rows) != len(artifact_ids):
        raise ValueError("HANS selection source rows do not align with artifact identities")
    if limit is not None and len(rows) != limit:
        raise ValueError("HANS selection membership does not equal its cap")
    source_ids = []
    for record, artifact_id in zip(rows, artifact_ids):
        source_id = record.get("pairID")
        expected = qualify_hans_pair_id(source_id, "evaluation")
        if record.get("canonical_pair_id") != expected or artifact_id != expected:
            raise ValueError("HANS selection raw-to-qualified identity mapping is invalid")
        source_ids.append(str(source_id))
    if len(source_ids) != len(set(source_ids)) or len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("HANS selection contains duplicate identities")
    payload = {
        "schema_version": "hans_selection_integrity_v1",
        "ranking_key": "source_local_pair_id",
        "ranking_algorithm": _HANS_SELECTION_ALGORITHM,
        "artifact_transform": _HANS_ARTIFACT_TRANSFORM,
        "selected_order": "source_order" if limit is None else "ranked_cap_order",
        "seed": seed,
        "strata_fields": ["gold_label", "heuristic", "subcase"],
        "cap": limit,
        "selected_count": len(source_ids),
        "selected_source_pair_ids": source_ids,
        "selected_source_pair_ids_sha256": _ordered_checksum(source_ids),
        "selected_artifact_ids_sha256": _ordered_checksum(artifact_ids),
        "source_to_artifact_mapping_sha256": _ordered_checksum(
            [[source_id, artifact_id] for source_id, artifact_id in zip(source_ids, artifact_ids)]
        ),
    }
    payload["integrity_checksum"] = _ordered_checksum(
        [[key, payload[key]] for key in sorted(payload)]
    )
    return payload


def hans_manifest_identity_summary(hans: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact compact HANS identity/integrity audit binding."""
    names = ("build", "dev", "evaluation")
    split = hans["split_integrity"]
    content = hans["content_integrity"]
    selection = hans["selection_integrity"]
    return {
        "identity_counts": {name: hans[name]["full_count"] for name in names},
        "identity_checksums": {name: hans[name]["full_ids_sha256"] for name in names},
        "split_integrity_summary": {
            "schema_version": split["schema_version"],
            "build_count": split["build_count"],
            "dev_count": split["dev_count"],
            "build_source_pair_ids_sha256": _ordered_checksum(
                split["build_source_pair_ids"]
            ),
            "dev_source_pair_ids_sha256": _ordered_checksum(
                split["dev_source_pair_ids"]
            ),
            "split_checksum": split["split_checksum"],
        },
        "content_integrity_summary": {
            "schema_version": content["schema_version"],
            "partitions": {
                name: {
                    "count": content["partitions"][name]["count"],
                    "content_sha256_ordered_checksum": content["partitions"][name][
                        "content_sha256_ordered_checksum"
                    ],
                    "source_id_content_joint_checksum": content["partitions"][name][
                        "source_id_content_joint_checksum"
                    ],
                }
                for name in names
            },
            "overlap_counts": dict(content["overlap_counts"]),
        },
        "selection_integrity_summary": {
            key: selection[key]
            for key in (
                "schema_version", "ranking_key", "ranking_algorithm",
                "artifact_transform", "selected_order", "seed", "strata_fields",
                "cap", "selected_count", "selected_source_pair_ids_sha256",
                "selected_artifact_ids_sha256", "source_to_artifact_mapping_sha256",
                "integrity_checksum",
            )
        },
    }


def _default_rng_factory(seed: int) -> Any:
    import numpy as np

    return np.random.default_rng(seed)


def split_hans_records(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    rng_factory: RngFactory | None = None,
) -> HansSplit:
    """Create the protocol-frozen, per-stratum HANS build/dev split."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("HANS split seed must be a non-negative integer")
    factory = rng_factory or _default_rng_factory
    strata: dict[tuple[str, str, str], list[tuple[str, str, HansRecord]]] = {}
    seen_ids: set[str] = set()
    seen_artifact_ids: set[str] = set()

    for record in records:
        pair_id, artifact_id, stratum = _normalized_hans_record(record)
        if pair_id in seen_ids:
            raise ValueError(f"duplicate HANS pair ID: {pair_id}")
        if artifact_id in seen_artifact_ids:
            raise ValueError(f"duplicate HANS artifact pair ID: {artifact_id}")
        seen_ids.add(pair_id)
        seen_artifact_ids.add(artifact_id)
        strata.setdefault(stratum, []).append((pair_id, artifact_id, dict(record)))

    build: list[HansRecord] = []
    dev: list[HansRecord] = []
    small: list[dict[str, Any]] = []

    for stratum in sorted(strata):
        ordered = sorted(strata[stratum], key=lambda item: item[0])
        count = len(ordered)
        if count < 5:
            build.extend(record for _, _, record in ordered)
            small.append(
                {
                    "gold_label": stratum[0],
                    "heuristic": stratum[1],
                    "subcase": stratum[2],
                    "count": count,
                    "build_pair_ids": [pair_id for pair_id, _, _ in ordered],
                    "dev_pair_ids": [],
                }
            )
            continue

        permutation = [int(index) for index in factory(seed).permutation(count)]
        if sorted(permutation) != list(range(count)):
            raise ValueError("RNG permutation is not a complete index permutation")
        dev_count = math.floor(0.20 * count)
        dev.extend(ordered[index][2] for index in permutation[:dev_count])
        build.extend(ordered[index][2] for index in permutation[dev_count:])

    build_ids = [_hans_artifact_id(row) for row in build]
    dev_ids = [_hans_artifact_id(row) for row in dev]
    validate_hans_disjointness(build_ids, dev_ids, [])
    build_source_ids = [str(_record_value(row, "source_pair_id", "pairID")) for row in build]
    dev_source_ids = [str(_record_value(row, "source_pair_id", "pairID")) for row in dev]
    membership = {
        "schema_version": "hans_split_v1",
        "hans_split_seed": seed,
        "build_pair_ids": build_source_ids,
        "dev_pair_ids": dev_source_ids,
        "small_strata": small,
    }
    checksum = sha256(_canonical_bytes(membership)).hexdigest()
    return HansSplit(seed, tuple(build), tuple(dev), tuple(small), checksum)


def _duplicate_ids(ids: Sequence[str], name: str) -> None:
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate pair IDs")


def validate_hans_disjointness(
    build_ids: Sequence[str],
    dev_ids: Sequence[str],
    evaluation_ids: Sequence[str],
) -> None:
    """Reject duplicates or overlap across the three HANS partitions."""
    build = [str(value) for value in build_ids]
    dev = [str(value) for value in dev_ids]
    evaluation = [str(value) for value in evaluation_ids]
    _duplicate_ids(build, "HANS build")
    _duplicate_ids(dev, "HANS dev")
    _duplicate_ids(evaluation, "HANS evaluation")
    overlaps = (
        ("build/dev", set(build) & set(dev)),
        ("build/evaluation", set(build) & set(evaluation)),
        ("dev/evaluation", set(dev) & set(evaluation)),
    )
    for label, values in overlaps:
        if values:
            preview = ", ".join(sorted(values)[:5])
            raise ValueError(f"HANS {label} overlap detected: {preview}")
