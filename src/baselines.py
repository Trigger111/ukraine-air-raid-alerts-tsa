"""Phase 5 - naive reference forecasts for ``has_alert_next_day``.

Each baseline returns P(has_alert_next_day) for the given rows. Baselines that
need fitting (base-rate constant, day-of-week climatology) are fit on TRAIN only.
Rolling / persistence baselines are inherently past-only (the window ends at day
t, the target is t+1), so they need no fitting and are valid on the test set.

    majority_always_alert : P = 1.0 every day
    base_rate_constant    : P = train mean of has_alert_next_day
    persistence           : P = has_alert today (lag-0)
    rolling_7d            : P = mean has_alert over [t-6 .. t]
    rolling_30d           : P = mean has_alert over [t-29 .. t]
    dow_climatology       : P = train P(alert | day-of-week of t+1)
"""
from __future__ import annotations

import pandas as pd


def majority_always_alert(index: pd.Index) -> pd.Series:
    return pd.Series(1.0, index=index)


def base_rate_constant(train_y: pd.Series, index: pd.Index) -> pd.Series:
    return pd.Series(float(train_y.mean()), index=index)


def persistence(test: pd.DataFrame) -> pd.Series:
    return test["has_alert_today"].astype(float)


def rolling_7d(test: pd.DataFrame) -> pd.Series:
    return test["has_alert_roll7"].astype(float)


def rolling_30d(daily_has: pd.Series, index: pd.Index) -> pd.Series:
    roll = daily_has.rolling(30, min_periods=1).mean()
    return roll.reindex(index).astype(float)


def dow_climatology(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    rate = train.groupby("next_dow")["has_alert_next_day"].mean()
    fallback = float(train["has_alert_next_day"].mean())
    return test["next_dow"].map(rate).fillna(fallback).astype(float)


def predict_all(train: pd.DataFrame, test: pd.DataFrame, daily_has: pd.Series) -> dict[str, pd.Series]:
    """All six baseline probability series for the test rows."""
    return {
        "majority_always_alert": majority_always_alert(test.index),
        "base_rate_constant": base_rate_constant(train["has_alert_next_day"], test.index),
        "persistence": persistence(test),
        "rolling_7d": rolling_7d(test),
        "rolling_30d": rolling_30d(daily_has, test.index),
        "dow_climatology": dow_climatology(train, test),
    }
