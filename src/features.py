"""Phase 6 — supervised framing for next-day forecasting (past-only, no leakage).

Target:
    has_alert_next_day = any_alert shifted by -1 day.

Features (computed AFTER the chronological split, strictly from past days):
    lag_1, lag_7 of any_alert; rolling_7 mean; calendar (dow, month, is_weekend).

``finished_at`` / duration are NEVER used as next-day features (target leakage):
at prediction time the current day's alerts may not have ended yet.
"""
from __future__ import annotations

import pandas as pd

from . import config  # noqa: F401  (used once implemented)


def make_supervised(daily: pd.DataFrame, target: str = "has_alert_next_day") -> pd.DataFrame:
    """Build the (features, target) frame for daily next-day forecasting."""
    raise NotImplementedError("Phase 6: feature engineering")
