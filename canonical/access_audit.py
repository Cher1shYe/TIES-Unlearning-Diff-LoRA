"""Append-only structured data-access audit events for canonical runs."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from canonical.artifacts import read_jsonl, write_jsonl


def append_access_event(path: Path, **payload: Any) -> dict[str, Any]:
    """Append one sequence-numbered access event and return its persisted payload."""
    path = Path(path)
    existing = read_jsonl(path) if path.is_file() else []
    event = {
        "sequence": len(existing),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    write_jsonl(path, [*existing, event])
    return event


def record_dataset_access(cfg: Any, *, dataset: str, split: str, purpose: str) -> None:
    """Record a dataset loader construction when the run enables access auditing."""
    path = getattr(cfg, "data_access_log", None)
    if path:
        append_access_event(
            Path(path),
            dataset=dataset,
            split=split,
            purpose=purpose,
            event="dataset_access",
        )


def record_final_evaluation_start(cfg: Any) -> None:
    """Mark the boundary immediately before official final evaluation loaders."""
    path = getattr(cfg, "data_access_log", None)
    if path:
        append_access_event(
            Path(path),
            dataset="hans",
            split=None,
            purpose="boundary",
            event="final_evaluation_start",
        )
