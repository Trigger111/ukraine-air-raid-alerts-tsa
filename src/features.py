"""Phase 4 - supervised framing for next-day forecasting (past-only, no leakage).

Row = origin day ``t``; we predict day ``t+1``::

    has_alert_next_day     = has_alert.shift(-1)       (primary, binary)
    alert_minutes_next_day = alert_minutes.shift(-1)   (secondary, optional)

Features are known at the end of day ``t`` (or deterministic for ``t+1``)::

    has_alert_today, has_alert_lag1, has_alert_lag6, has_alert_lag7
    has_alert_roll7         = mean has_alert over [t-6 .. t]
    alert_minutes_today, alert_minutes_roll7
    next_dow, next_month, next_is_weekend     (calendar of t+1, known in advance)

Leakage controls:
  * no feature uses day t+1 alert info; rolling windows end at t (exclude t+1);
  * finished_at / duration are never features;
  * the daily index must be gap-free so row-lags equal calendar-day lags;
  * NO scaling/normalisation here (fit on train only, in Phase 5);
  * rows stay in chronological order (the split happens later).

Run::

    python -m src.features
"""
from __future__ import annotations

import pandas as pd

from . import config

TARGETS = ["has_alert_next_day", "alert_minutes_next_day"]
FEATURES = [
    "has_alert_today", "has_alert_lag1", "has_alert_lag6", "has_alert_lag7",
    "has_alert_roll7", "alert_minutes_today", "alert_minutes_roll7",
    "next_dow", "next_month", "next_is_weekend",
]


def _as_int_bool(s: pd.Series) -> pd.Series:
    if s.dtype != bool:
        s = s.astype(str).str.lower().map({"true": True, "false": False})
    return s.fillna(False).astype(int)


def make_supervised(daily: pd.DataFrame) -> pd.DataFrame:
    """Build the (features, targets) table for next-day forecasting from `daily`."""
    d = daily.sort_index().copy()
    d.index = pd.DatetimeIndex(d.index)

    # Require a gap-free daily index so row-lags == calendar-day lags.
    full = pd.date_range(d.index.min(), d.index.max(), freq="D")
    if not d.index.equals(full):
        d = d.reindex(full)
    d.index.name = "date"

    has = _as_int_bool(d["has_alert"])
    mins = d["alert_minutes"].astype(float).fillna(0.0)

    out = pd.DataFrame(index=d.index)
    # targets (day t+1)
    out["has_alert_next_day"] = has.shift(-1)
    out["alert_minutes_next_day"] = mins.shift(-1)
    # past-only features (<= t)
    out["has_alert_today"] = has
    out["has_alert_lag1"] = has.shift(1)
    out["has_alert_lag6"] = has.shift(6)            # same weekday as t+1
    out["has_alert_lag7"] = has.shift(7)
    out["has_alert_roll7"] = has.rolling(7).mean()  # window [t-6 .. t]
    out["alert_minutes_today"] = mins
    out["alert_minutes_roll7"] = mins.rolling(7).mean()
    # calendar of the target day t+1 (deterministic, known in advance)
    tgt = out.index + pd.Timedelta(days=1)
    out["next_dow"] = tgt.dayofweek
    out["next_month"] = tgt.month
    out["next_is_weekend"] = (tgt.dayofweek >= 5).astype(int)

    out = out.dropna().copy()  # drops warmup (lag/roll NaN) + last row (no t+1)
    out["has_alert_next_day"] = out["has_alert_next_day"].astype(int)
    return out[TARGETS + FEATURES]


def _qa(sup: pd.DataFrame) -> None:
    line = "-" * 60
    print(line)
    print(f"SUPERVISED TABLE - {config.TARGET_REGION}")
    print(line)
    print(f"  rows           : {len(sup)}")
    print(f"  date range     : {sup.index.min().date()} .. {sup.index.max().date()}")
    print(f"  features ({len(FEATURES)})   : {FEATURES}")
    print(f"  targets        : {TARGETS}")
    bal = sup["has_alert_next_day"].mean()
    print(f"  target balance : has_alert_next_day mean = {bal:.3f} "
          f"({int(sup['has_alert_next_day'].sum())}/{len(sup)})  "
          f"-> {'OK (non-degenerate)' if 0.1 < bal < 0.9 else 'CHECK (imbalanced)'}")
    print(f"  missing values : {int(sup.isna().sum().sum())}")
    print(f"  minutes_next_day: min={sup['alert_minutes_next_day'].min():.0f} "
          f"median={sup['alert_minutes_next_day'].median():.0f} "
          f"max={sup['alert_minutes_next_day'].max():.0f}")
    # Leakage sanity: today != next-day perfectly (else a feature == target).
    agree = (sup["has_alert_today"] == sup["has_alert_next_day"]).mean()
    print(f"  persistence agreement (today==next): {agree:.3f}  "
          f"(<1.0 => no trivial target leak; ref for Phase-5 persistence)")


def main() -> None:
    slug = config.TARGET_REGION.lower().replace(" ", "_")
    daily = pd.read_csv(config.DATA_PROCESSED / f"daily_{slug}.csv",
                        parse_dates=["date"], index_col="date")
    sup = make_supervised(daily)
    out_path = config.DATA_PROCESSED / f"supervised_{slug}.csv"
    sup.to_csv(out_path)
    _qa(sup)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
