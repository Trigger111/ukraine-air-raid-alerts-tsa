"""Phase 1/2 pipeline: raw snapshot -> canonical event table + daily aggregates.

Run::

    py -m src.pipeline

Writes ``data/processed/events_canonical.csv`` and ``data/processed/daily_<region>.csv``
and prints a QA summary (shape, columns, date range, region checks, censored /
overlap / merge / duplicate counts).
"""
from __future__ import annotations

import pandas as pd

from . import aggregate, clean, config, data_loader


def _print_qa(events: pd.DataFrame, daily: pd.DataFrame, report: dict) -> None:
    line = "-" * 64
    print(line)
    print("CLEANING REPORT")
    print(line)
    for k in [
        "total_raw", "dropped_bad_start", "negative_duration_rows",
        "exact_duplicates_removed", "overlap_groups", "rows_merged_into_overlaps",
        "kyiv_city_overlap_groups", "censored_count", "long_alert_count",
        "events_final", "regions_count",
    ]:
        print(f"  {k:<28} {report.get(k)}")

    if report.get("overlaps_by_region"):
        print("\n  overlaps by region (top 10):")
        for reg, n in list(report["overlaps_by_region"].items())[:10]:
            print(f"    {reg:<24} {n}")

    if report.get("overlap_examples"):
        print("\n  overlap examples (raw intervals that were merged):")
        for ex in report["overlap_examples"][:3]:
            unit = ex["exact_unit"].iloc[0]
            print(f"    unit: {unit}")
            for r in ex.itertuples(index=False):
                print(f"      {r.started_at_utc}  ->  {r.finished_at_utc}")

    if report.get("censored_count"):
        print("\n  censored events:")
        print(f"    date range : {report.get('censored_date_min')} .. {report.get('censored_date_max')}")
        print(f"    dataset max: {report.get('dataset_date_max')}")
        print(f"    regions    : {report.get('censored_regions')}")

    print("\n" + line)
    print("CANONICAL EVENT TABLE")
    print(line)
    print(f"  shape   : {events.shape}")
    print(f"  columns : {list(events.columns)}")
    print(f"  UTC  range : {events['started_at_utc'].min()} .. {events['started_at_utc'].max()}")
    print(f"  Kyiv range : {events['started_at_kyiv'].min()} .. {events['started_at_kyiv'].max()}")
    nonlong = events.loc[~events["is_long_alert"], "duration_min"]
    print(f"  duration_min (excl. {report.get('long_alert_count')} long outliers): "
          f"min={nonlong.min():.1f} median={nonlong.median():.1f} max={nonlong.max():.1f}")

    print("\n  Kyiv City vs Kyivska oblast (distinct units):")
    for reg in config.COMPARISON_REGIONS:
        sub = events[events["region"] == reg]
        print(f"    {reg:<16} events={len(sub):>7}  censored={int(sub['is_censored'].sum())}"
              f"  long={int(sub['is_long_alert'].sum())}")

    print("\n" + line)
    print(f"DAILY TABLE -- {config.TARGET_REGION}")
    print(line)
    print(f"  shape   : {daily.shape}")
    print(f"  columns : {list(daily.columns)}")
    print(f"  date range : {daily.index.min().date()} .. {daily.index.max().date()}")
    base = daily["has_alert"].mean()
    print(f"  has_alert base rate : {base:.3f}  ({int(daily['has_alert'].sum())} / {len(daily)} days)")
    print(f"  alert_minutes/day   : mean={daily['alert_minutes'].mean():.1f} "
          f"max={daily['alert_minutes'].max():.1f}")
    print(f"  days with censored  : {int(daily['has_censored_event'].sum())}")
    print(f"  last (incomplete) day, excluded at modeling time: {daily.index.max().date()}")


def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    raw = data_loader.load_raw()
    events, report = clean.to_event_table(raw)
    events.to_csv(config.DATA_PROCESSED / "events_canonical.csv", index=False)

    daily = aggregate.daily_region(events, config.TARGET_REGION)
    slug = config.TARGET_REGION.lower().replace(" ", "_")
    daily.to_csv(config.DATA_PROCESSED / f"daily_{slug}.csv")

    _print_qa(events, daily, report)
    print("\nWrote:")
    print(f"  {config.DATA_PROCESSED / 'events_canonical.csv'}")
    print(f"  {config.DATA_PROCESSED / ('daily_' + slug + '.csv')}")


if __name__ == "__main__":
    main()
