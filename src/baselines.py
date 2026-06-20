"""Phase 5 — naive reference forecasts for ``has_alert_next_day``.

    persistence     : yhat(t+1) = any_alert(t)                    ("yesterday")
    seasonal_naive  : yhat(t+1) = any_alert(t-6)                  ("same weekday last week")
    rolling_mean_7  : P(t+1)    = mean(any_alert) over last 7 days (probabilistic)

Every trained model is reported against these baselines — the headline result is
*improvement over the naive baseline*, not an absolute score.
"""
from __future__ import annotations

import pandas as pd


def persistence(any_alert: pd.Series) -> pd.Series:
    """Predict next day == today (lag-1)."""
    raise NotImplementedError("Phase 5: persistence baseline")


def seasonal_naive(any_alert: pd.Series) -> pd.Series:
    """Predict next day == same weekday last week (lag-7)."""
    raise NotImplementedError("Phase 5: seasonal-naive baseline")


def rolling_mean(any_alert: pd.Series, window: int = 7) -> pd.Series:
    """Predict next-day probability == mean over the trailing window."""
    raise NotImplementedError("Phase 5: rolling-mean baseline")
