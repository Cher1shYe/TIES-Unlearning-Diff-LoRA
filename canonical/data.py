"""Pure data-selection and HANS partition contracts for canonical v1."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Callable, Iterable, Mapping, Sequence


HansRecord = dict[str, Any]
RngFactory = Callable[[int], Any]


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


def _record_value(record: Mapping[str, Any], canonical: str, source: str) -> Any:
    if canonical in record:
        return record[canonical]
    if source in record:
        return record[source]
    raise ValueError(f"HANS record is missing required field {source!r}")


def _normalized_hans_record(record: Mapping[str, Any]) -> tuple[str, tuple[str, str, str]]:
    pair_id = str(_record_value(record, "pair_id", "pairID"))
    gold_label = str(_record_value(record, "gold_label", "gold_label"))
    heuristic = str(_record_value(record, "heuristic", "heuristic"))
    subcase = str(_record_value(record, "subcase", "subcase"))
    if not pair_id:
        raise ValueError("HANS pairID must not be empty")
    return pair_id, (gold_label, heuristic, subcase)


@dataclass(frozen=True)
class HansSplit:
    seed: int
    build_records: tuple[HansRecord, ...]
    dev_records: tuple[HansRecord, ...]
    small_strata: tuple[dict[str, Any], ...]
    checksum: str

    @property
    def build_pair_ids(self) -> list[str]:
        return [str(_record_value(row, "pair_id", "pairID")) for row in self.build_records]

    @property
    def dev_pair_ids(self) -> list[str]:
        return [str(_record_value(row, "pair_id", "pairID")) for row in self.dev_records]

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
    strata: dict[tuple[str, str, str], list[tuple[str, HansRecord]]] = {}
    seen_ids: set[str] = set()

    for record in records:
        pair_id, stratum = _normalized_hans_record(record)
        if pair_id in seen_ids:
            raise ValueError(f"duplicate HANS pair ID: {pair_id}")
        seen_ids.add(pair_id)
        strata.setdefault(stratum, []).append((pair_id, dict(record)))

    build: list[HansRecord] = []
    dev: list[HansRecord] = []
    small: list[dict[str, Any]] = []

    for stratum in sorted(strata):
        ordered = sorted(strata[stratum], key=lambda item: item[0])
        count = len(ordered)
        if count < 5:
            build.extend(record for _, record in ordered)
            small.append(
                {
                    "gold_label": stratum[0],
                    "heuristic": stratum[1],
                    "subcase": stratum[2],
                    "count": count,
                    "build_pair_ids": [pair_id for pair_id, _ in ordered],
                    "dev_pair_ids": [],
                }
            )
            continue

        permutation = [int(index) for index in factory(seed).permutation(count)]
        if sorted(permutation) != list(range(count)):
            raise ValueError("RNG permutation is not a complete index permutation")
        dev_count = math.floor(0.20 * count)
        dev.extend(ordered[index][1] for index in permutation[:dev_count])
        build.extend(ordered[index][1] for index in permutation[dev_count:])

    build_ids = [str(_record_value(row, "pair_id", "pairID")) for row in build]
    dev_ids = [str(_record_value(row, "pair_id", "pairID")) for row in dev]
    validate_hans_disjointness(build_ids, dev_ids, [])
    membership = {
        "schema_version": "hans_split_v1",
        "hans_split_seed": seed,
        "build_pair_ids": build_ids,
        "dev_pair_ids": dev_ids,
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
