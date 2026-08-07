"""Stable data-identity entries for canonical and Stage 2 smoke manifests."""

from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from canonical.data import deterministic_cap_records, stable_record_id


def _ids_checksum(ids: Sequence[str]) -> str:
    """Hash an ordered list of stable IDs with canonical JSON encoding."""
    payload = json.dumps(
        list(ids),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def dataset_identity_entry(
    records: Sequence[Mapping[str, Any]],
    *,
    source: str,
    split: str,
    preferred_id_fields: Sequence[str],
    selected_limit: int | None,
    seed: int,
    strata_fields: Sequence[str] = (),
    selected_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Bind a full record collection and a deterministic selected subset to IDs."""
    full_rows = [dict(record) for record in records]
    if not full_rows:
        raise ValueError("dataset identity entry rejects an empty dataset")

    full_ids = [stable_record_id(row, preferred_id_fields) for row in full_rows]
    if len(full_ids) != len(set(full_ids)):
        raise ValueError("dataset identity entry contains duplicate full IDs")

    if selected_records is None:
        if selected_limit is None:
            selected_ids = list(full_ids)
        else:
            if selected_limit > len(full_rows):
                raise ValueError("selected limit exceeds full dataset membership")
            _, selected_ids = deterministic_cap_records(
                full_rows,
                selected_limit,
                seed,
                strata_fields,
                preferred_id_fields,
            )
    else:
        selected_rows = [dict(record) for record in selected_records]
        selected_ids = [stable_record_id(row, preferred_id_fields) for row in selected_rows]
        if selected_limit is not None and len(selected_ids) > selected_limit:
            raise ValueError("selected records exceed selected limit")

    if not selected_ids:
        raise ValueError("dataset identity entry rejects empty selected membership")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("dataset identity entry contains duplicate selected IDs")
    if not set(selected_ids).issubset(full_ids):
        raise ValueError("selected ID is not present in full dataset membership")

    return {
        "source": str(source),
        "split": str(split),
        "id_strategy": "preferred_field_or_content_sha256",
        "preferred_id_fields": [str(field) for field in preferred_id_fields],
        "strata_fields": [str(field) for field in strata_fields],
        "selection_seed": int(seed),
        "selected_limit": selected_limit,
        "full_count": len(full_ids),
        "selected_count": len(selected_ids),
        "full_ids": full_ids,
        "selected_ids": selected_ids,
        "full_ids_sha256": _ids_checksum(full_ids),
        "selected_ids_sha256": _ids_checksum(selected_ids),
    }
