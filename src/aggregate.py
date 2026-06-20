"""Phase 2/3 — event-level table -> regular time series, parameterised by region.

``daily_region`` builds a gap-free daily index (missing days filled with 0) for a
single region (default ``config.TARGET_REGION``). Minutes are split across local
calendar days so a night-time alert is attributed correctly:

    alert 2024-01-01 23:30 -> 2024-01-02 00:20 (Kyiv) gives
        2024-01-01: +30 min,  2024-01-02: +20 min,  has_alert True on both days.

Columns:
    alert_start_count   alerts that *started* that day (Europe/Kyiv)
    night_start_count   subset starting 00:00-05:59 local
    alert_minutes       active minutes that day (split; censored excluded)
    has_alert           True if any alert was active that day (catches spillover)
    has_censored_event  True if a censored alert started that day

``hourly_region`` (EDA / future work) is not part of the daily MVP.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

import pandas as pd

from . import config


def _split_minutes_by_day(start_local: pd.Timestamp, end_local: pd.Timestamp) -> dict[dt.date, float]:
    """Split [start, end] into {local_date: minutes}, DST-correct."""
    out: dict[dt.date, float] = defaultdict(float)
    cur = start_local
    while cur.date() < end_local.date():
        nxt = pd.Timestamp(cur.date() + dt.timedelta(days=1), tz=config.TZ_LOCAL)
        out[cur.date()] += (nxt - cur).total_seconds() / 60.0
        cur = nxt
    out[cur.date()] += (end_local - cur).total_seconds() / 60.0
    return out


def daily_region(events: pd.DataFrame, region: str = config.TARGET_REGION) -> pd.DataFrame:
    """Daily aggregate series for a single region (default: config.TARGET_REGION)."""
    ev = events[events["region"] == region]
    if ev.empty:
        raise ValueError(f"No events for region {region!r}")

    start_counts: dict[dt.date, int] = defaultdict(int)
    night_counts: dict[dt.date, int] = defaultdict(int)
    minutes: dict[dt.date, float] = defaultdict(float)
    active_days: set[dt.date] = set()
    censored_days: set[dt.date] = set()
    max_end = ev["started_at_kyiv"].max()

    for row in ev.itertuples(index=False):
        sd = row.started_at_kyiv.date()
        start_counts[sd] += 1
        if 0 <= row.started_at_kyiv.hour < 6:
            night_counts[sd] += 1
        active_days.add(sd)  # an alert that started here makes the day active
        if row.is_censored:
            censored_days.add(sd)
            continue  # unknown end -> contribute no minutes
        for d, m in _split_minutes_by_day(row.started_at_kyiv, row.finished_at_kyiv).items():
            minutes[d] += m
            if m > 0:
                active_days.add(d)
        if row.finished_at_kyiv > max_end:
            max_end = row.finished_at_kyiv

    idx = pd.date_range(ev["started_at_kyiv"].min().date(), max_end.date(), freq="D")
    d = idx.date
    daily = pd.DataFrame(
        {
            "alert_start_count": [start_counts.get(x, 0) for x in d],
            "night_start_count": [night_counts.get(x, 0) for x in d],
            "alert_minutes": [round(minutes.get(x, 0.0), 2) for x in d],
            "has_alert": [x in active_days for x in d],
            "has_censored_event": [x in censored_days for x in d],
        },
        index=pd.Index(idx, name="date"),
    )
    return daily


def hourly_region(events: pd.DataFrame, region: str = config.TARGET_REGION) -> pd.DataFrame:
    """Hourly overlap-based table for a single region (EDA; model = future work)."""
    raise NotImplementedError("EDA/future work: hourly aggregation")
