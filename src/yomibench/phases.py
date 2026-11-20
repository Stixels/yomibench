"""Per-thread running time totals of self-play's phases (value-net scoring, the matrix-game solve, candidate
expansion, engine steps, record writing), for the chunk ledger row's per-phase timings. Timing only: nothing
here changes what is played or recorded. Each self-play worker plays its games on its own thread, so a
thread's totals are its own games'; `selfplay.run_selfplay` adds each game's difference to the run's stats."""

from __future__ import annotations

import threading

PHASES = ("net", "solve", "expand", "step", "record")

_local = threading.local()


def totals() -> dict[str, float]:
    """This thread's running totals (seconds by phase), live: callers copy it to take a difference."""
    t = getattr(_local, "totals", None)
    if t is None:
        t = _local.totals = dict.fromkeys(PHASES, 0.0)
    return t


def add(phase: str, seconds: float) -> None:
    totals()[phase] += seconds
