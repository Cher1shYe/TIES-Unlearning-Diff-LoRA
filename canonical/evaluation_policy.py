"""Fail-closed access policy for HANS development and final evaluation data."""

from typing import Literal


_EVENT_SPLITS = {
    "phase1_end": "dev",
    "phase2_end": "dev",
    "phase2_5": "dev",
    "phase3_epoch": "dev",
    "final_evaluation": "evaluation",
}


def hans_split_for_event(event: str) -> Literal["dev", "evaluation"]:
    try:
        return _EVENT_SPLITS[event]
    except KeyError as exc:
        raise ValueError(f"unknown HANS evaluation event: {event!r}") from exc
