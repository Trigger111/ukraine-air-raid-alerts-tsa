"""Phase 2 — raw snapshot -> canonical, leakage-free event-level table.

Cleaning rules (agreed):
  * Exact unit = ``level + oblast + raion + hromada``. Two alerts cannot be
    active simultaneously *within one exact unit*, so an overlap there is a
    data-quality signal — not a reason to merge across raion/hromada.
  * Drop exact duplicates (same unit + same start + same end). NOTE: the
    upstream snapshot contains ~42% full-row duplicates; deduping is essential.
  * Merge only STRICT overlaps within an exact unit (for duration accounting),
    recording ``source_event_count`` = number of raw rows merged.
  * Censoring: a missing ``finished_at`` is NOT imputed. It becomes
    ``is_censored=True`` with ``duration_min=NaN`` (no fake 30-minute end).
  * An impossible ``finished_at < started_at`` is treated as missing (censored).
  * Implausibly long alerts (> ``config.LONG_ALERT_HOURS``) are flagged via
    ``is_long_alert`` (not dropped) — they cluster on front-line hromadas.

No future-derived columns live here; supervised features are built later,
strictly after the chronological split (see :mod:`features`).
"""
from __future__ import annotations

import pandas as pd

from . import config

STR_COLS = ["oblast", "raion", "hromada", "level", "source"]


def _merge_overlaps(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Merge strict overlaps within each exact unit (non-censored only)."""
    df = df.copy()
    df["source_event_count"] = 1
    keep = STR_COLS + [
        "started_at_utc", "finished_at_utc", "is_censored",
        "source_event_count", "exact_unit",
    ]

    cens = df[df["is_censored"]].copy()
    nonc = df[~df["is_censored"]].copy()

    if len(nonc):
        nonc = nonc.sort_values(["exact_unit", "started_at_utc"]).reset_index(drop=True)
        cummax_end = nonc.groupby("exact_unit")["finished_at_utc"].cummax()
        prev = cummax_end.groupby(nonc["exact_unit"]).shift()
        # New episode when this alert starts at/after everything seen so far ended
        # (strict overlap => start < prev end => same episode => merged).
        new_ep = (nonc["started_at_utc"] >= prev) | prev.isna()
        nonc["ep"] = new_ep.groupby(nonc["exact_unit"]).cumsum()

        sizes = nonc.groupby(["exact_unit", "ep"]).size()
        multi = sizes[sizes > 1]
        report["overlap_groups"] = int(len(multi))
        report["rows_merged_into_overlaps"] = int((multi - 1).sum())

        by_region = (
            multi.reset_index()
            .assign(region=lambda d: d["exact_unit"].str.split("|").str[1])
            .groupby("region").size().sort_values(ascending=False)
        )
        report["overlaps_by_region"] = by_region.to_dict()
        report["kyiv_city_overlap_groups"] = int(by_region.get("Kyiv City", 0))

        examples = []
        for unit, ep in list(multi.index)[:5]:
            members = nonc[(nonc["exact_unit"] == unit) & (nonc["ep"] == ep)]
            examples.append(members[["exact_unit", "started_at_utc", "finished_at_utc"]])
        report["overlap_examples"] = examples

        merged = (
            nonc.groupby(["exact_unit", "ep"], sort=False)
            .agg(
                oblast=("oblast", "first"), raion=("raion", "first"),
                hromada=("hromada", "first"), level=("level", "first"),
                source=("source", "first"),
                started_at_utc=("started_at_utc", "min"),
                finished_at_utc=("finished_at_utc", "max"),
                source_event_count=("source_event_count", "sum"),
            )
            .reset_index()
        )
        merged["is_censored"] = False
        merged = merged[keep]
    else:
        report["overlap_groups"] = 0
        report["rows_merged_into_overlaps"] = 0
        report["overlaps_by_region"] = {}
        report["kyiv_city_overlap_groups"] = 0
        report["overlap_examples"] = []
        merged = nonc[keep] if len(nonc) else nonc

    cens = cens[keep] if len(cens) else cens
    return pd.concat([merged, cens], ignore_index=True)


def to_event_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Convert the raw snapshot into the canonical event table + a QA report."""
    report: dict = {"total_raw": len(raw)}
    df = raw.copy()

    for c in STR_COLS:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["started_at_utc"] = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
    df["finished_at_utc"] = pd.to_datetime(df["finished_at"], utc=True, errors="coerce")

    bad_start = df["started_at_utc"].isna()
    report["dropped_bad_start"] = int(bad_start.sum())
    df = df[~bad_start].copy()

    neg = df["finished_at_utc"].notna() & (df["finished_at_utc"] < df["started_at_utc"])
    report["negative_duration_rows"] = int(neg.sum())
    df.loc[neg, "finished_at_utc"] = pd.NaT

    df["is_censored"] = df["finished_at_utc"].isna()
    report["censored_count"] = int(df["is_censored"].sum())

    df["exact_unit"] = (
        df["level"] + "|" + df["oblast"] + "|" + df["raion"] + "|" + df["hromada"]
    )

    dup = df.duplicated(["exact_unit", "started_at_utc", "finished_at_utc"], keep="first")
    report["exact_duplicates_removed"] = int(dup.sum())
    df = df[~dup].copy()

    events = _merge_overlaps(df, report)

    events["started_at_kyiv"] = events["started_at_utc"].dt.tz_convert(config.TZ_LOCAL)
    events["finished_at_kyiv"] = events["finished_at_utc"].dt.tz_convert(config.TZ_LOCAL)
    events["duration_min"] = (
        events["finished_at_utc"] - events["started_at_utc"]
    ).dt.total_seconds() / 60.0
    events["is_long_alert"] = events["duration_min"] > config.LONG_ALERT_HOURS * 60
    events["date_kyiv"] = events["started_at_kyiv"].dt.date
    events["hour_kyiv"] = events["started_at_kyiv"].dt.hour
    events["dow_kyiv"] = events["started_at_kyiv"].dt.dayofweek

    events = events.rename(columns={"oblast": "region"})
    cols = [
        "region", "raion", "hromada", "level", "source",
        "started_at_utc", "finished_at_utc", "started_at_kyiv", "finished_at_kyiv",
        "duration_min", "is_censored", "is_long_alert", "source_event_count",
        "date_kyiv", "hour_kyiv", "dow_kyiv",
    ]
    events = events[cols].sort_values("started_at_utc").reset_index(drop=True)

    report["events_final"] = len(events)
    report["regions_count"] = int(events["region"].nunique())
    report["long_alert_count"] = int(events["is_long_alert"].sum())
    if report["censored_count"]:
        cd = events.loc[events["is_censored"], "started_at_kyiv"]
        report["censored_date_min"] = str(cd.min().date())
        report["censored_date_max"] = str(cd.max().date())
        report["dataset_date_max"] = str(events["started_at_kyiv"].max().date())
        report["censored_regions"] = sorted(events.loc[events["is_censored"], "region"].unique())

    return events, report
