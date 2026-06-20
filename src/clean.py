"""Phase 2 — raw snapshot -> canonical, leakage-free event-level table.

Planned transformation (one row per alert)::

    region, raion, hromada, level, source,
    started_at_utc, finished_at_utc,
    started_at_kyiv, finished_at_kyiv,        # via config.TZ_LOCAL
    duration_min, is_censored,                # censored = finished_at missing
    date_kyiv, hour_kyiv, dow_kyiv

QA steps: drop non-positive durations, dedupe, resolve overlapping episodes.
No future-derived columns live here — supervised features are built later,
strictly after the chronological split (see :mod:`features`).
"""
from __future__ import annotations

import pandas as pd

from . import config  # noqa: F401  (used once implemented)


def to_event_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert the raw snapshot into the canonical event-level table."""
    raise NotImplementedError("Phase 2: event-level cleaning")
