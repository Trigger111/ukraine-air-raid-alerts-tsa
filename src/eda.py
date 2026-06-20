"""Phase 3B - static EDA artifact. Generates PNG figures into ``figures/``.

Run::

    python -m src.pipeline   # build data/processed/*.csv first (if needed)
    python -m src.eda

Exploratory only: this describes historical alert *declarations* and alert
*activity*, NOT attacks/shelling, and is not a forecasting or safety tool.

Methodology note (national comparison):
  * ``alert episode count`` = number of cleaned alert intervals.
  * ``all-level row-sum`` of hours can OVERCOUNT, because parallel raion/hromada
    alerts multiply hours; it is avoided here.
  * ``oblast union time`` = wall-clock time during which at least one alert is
    active in the oblast (overlaps across levels merged). This is the default
    metric for comparing regions, and the only one comparable across the
    Dec-2025 oblast->raion granularity shift. For the same reason, the monthly
    trend uses oblast-level episodes only.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import aggregate, config  # noqa: E402

DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
RED, ORANGE, NAVY, BLUE = "#c0392b", "#e67e22", "#2c3e50", "#2980b9"
GRANULARITY_SHIFT = pd.Timestamp("2025-12-01")

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.titlesize": 12, "font.size": 10})


def _coerce_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def load_events() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PROCESSED / "events_canonical.csv",
                     parse_dates=["started_at_utc", "finished_at_utc"])
    for c in ["is_censored", "is_long_alert"]:
        df[c] = _coerce_bool(df[c])
    df["date_kyiv"] = pd.to_datetime(df["date_kyiv"])
    df["month"] = df["date_kyiv"].dt.to_period("M").dt.to_timestamp()
    return df


def load_daily() -> pd.DataFrame:
    slug = config.TARGET_REGION.lower().replace(" ", "_")
    return pd.read_csv(config.DATA_PROCESSED / f"daily_{slug}.csv",
                       parse_dates=["date"], index_col="date")


def _save(fig, name: str) -> str:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return name


# --- National ------------------------------------------------------------
def national_count_oblast_only(ev: pd.DataFrame) -> str:
    cnt = ev[ev["level"] == "oblast"]["region"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(cnt.index, cnt.values, color=RED)
    ax.set_xlabel("number of alert episodes (level == oblast)")
    ax.set_title("Oblast-wide alert episodes by region")
    ax.grid(axis="y")
    return _save(fig, "national_alert_episode_count_oblast_only.png")


def national_union_hours(ev: pd.DataFrame) -> str:
    hrs = aggregate.region_hours(ev, "union").sort_values()
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(hrs.index, hrs.values, color=ORANGE)
    ax.set_xlabel("oblast union time (hours; any active alert, overlaps merged)")
    ax.set_title("Time under alert by region (oblast union time)")
    ax.grid(axis="y")
    return _save(fig, "national_union_alert_hours.png")


def national_monthly_trend(ev: pd.DataFrame) -> str:
    # Robust activity (granularity-invariant): distinct region-days with an alert.
    rad = ev.drop_duplicates(["region", "date_kyiv"]).groupby("month").size()
    # Declaration practice: oblast-wide episodes (collapses as alerts go raion-level).
    obl = ev[ev["level"] == "oblast"].groupby("month").size().reindex(rad.index, fill_value=0)
    rad, obl = rad.iloc[:-1], obl.iloc[:-1]  # drop final incomplete month

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.plot(rad.index, rad.values, color=NAVY)
    ax1.set_ylabel("region-alert-days / month")
    ax1.set_ylim(0, None)
    ax1.set_title("Alert activity is stable; only the declaration granularity changed")
    ax1.text(0.01, 0.88, "robust: any-alert region-days (granularity-invariant)",
             transform=ax1.transAxes, fontsize=8, color=NAVY)

    ax2.plot(obl.index, obl.values, color=RED)
    ax2.set_ylabel("oblast-wide episodes / month")
    ax2.set_ylim(0, None)
    ax2.text(0.01, 0.88, "oblast-wide declarations (collapse from ~Aug 2025)",
             transform=ax2.transAxes, fontsize=8, color=RED)

    for ax in (ax1, ax2):
        ax.axvline(GRANULARITY_SHIFT, color="grey", ls="--", lw=1)
    ax2.text(GRANULARITY_SHIFT, ax2.get_ylim()[1] * 0.9, " raion-level shift",
             color="grey", fontsize=8, va="top")
    return _save(fig, "national_monthly_alert_trend.png")


# --- Kyiv City -----------------------------------------------------------
def kc_daily_minutes(daily: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily.index, daily["alert_minutes"], color=RED, lw=0.6, alpha=0.5,
            label="daily")
    ax.plot(daily.index, daily["alert_minutes"].rolling(7).mean(), color=NAVY,
            lw=1.5, label="7-day mean")
    ax.set_ylabel("alert minutes / day")
    ax.set_title(f"{config.TARGET_REGION}: daily alert minutes")
    ax.legend()
    return _save(fig, "kyiv_city_daily_alert_minutes.png")


def kc_starts_by_hour(kc: pd.DataFrame) -> str:
    by_hour = kc["hour_kyiv"].value_counts().reindex(range(24), fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(by_hour.index, by_hour.values, color=RED)
    ax.set_xlabel("hour of day (Europe/Kyiv)")
    ax.set_ylabel("alert starts")
    ax.set_xticks(range(0, 24))
    ax.set_title(f"{config.TARGET_REGION}: alert starts by hour of day")
    return _save(fig, "kyiv_city_alert_starts_by_hour.png")


def kc_hour_weekday_heatmap(kc: pd.DataFrame) -> str:
    piv = (kc.groupby(["dow_kyiv", "hour_kyiv"]).size()
           .unstack(fill_value=0)
           .reindex(index=range(7), columns=range(24), fill_value=0))
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(piv.values, aspect="auto", cmap="magma")
    ax.set_yticks(range(7)); ax.set_yticklabels(DOW_LABELS)
    ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels(range(0, 24, 2))
    ax.set_xlabel("hour (Europe/Kyiv)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="alert starts")
    ax.set_title(f"{config.TARGET_REGION}: alert starts, hour x day-of-week")
    return _save(fig, "kyiv_city_hour_weekday_heatmap.png")


def kc_duration_distribution(kc: pd.DataFrame) -> str:
    dur = kc["duration_min"].dropna()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(dur, bins=50, color=ORANGE)
    ax.axvline(dur.median(), color=NAVY, ls="--", lw=1.5,
               label=f"median {dur.median():.0f} min")
    ax.set_xlabel("alert duration (minutes)")
    ax.set_ylabel("number of episodes")
    ax.set_title(f"{config.TARGET_REGION}: alert duration distribution")
    ax.legend()
    return _save(fig, "kyiv_city_duration_distribution.png")


# --- Kyiv City vs Kyivska oblast -----------------------------------------
def city_vs_oblast_monthly(ev: pd.DataFrame) -> str:
    sub = ev[ev["region"].isin(["Kyiv City", "Kyivska oblast"])]
    monthly = sub.groupby(["month", "region"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 4))
    for reg, color in [("Kyiv City", RED), ("Kyivska oblast", BLUE)]:
        if reg in monthly.columns:
            ax.plot(monthly.index, monthly[reg], label=reg, color=color)
    ax.set_ylabel("alert episodes / month")
    ax.set_title("Kyiv City vs Kyivska oblast: monthly alert episodes")
    ax.legend()
    return _save(fig, "kyiv_city_vs_kyivska_oblast_monthly.png")


def _print_findings(ev: pd.DataFrame, daily: pd.DataFrame) -> None:
    kc = ev[ev["region"] == config.TARGET_REGION]
    union = aggregate.region_hours(ev, "union")
    obl_cnt = ev[ev["level"] == "oblast"]["region"].value_counts()
    by_hour = kc["hour_kyiv"].value_counts()
    night = (kc["hour_kyiv"] < 6).mean()
    print("\n--- key numbers for EDA findings ---")
    print("top union-hours:", {k: round(v) for k, v in union.head(3).items()})
    print("top oblast-only episodes:", obl_cnt.head(3).to_dict())
    print(f"Kyiv City busiest start hour: {by_hour.idxmax()}:00 ({by_hour.max()} starts)")
    print(f"Kyiv City night starts (00-06) share: {night:.2f}")
    print(f"Kyiv City daily has_alert base rate: {daily['has_alert'].mean():.3f}")
    print(f"Kyiv City median duration (min): {kc['duration_min'].median():.1f}")


def main() -> None:
    ev = load_events()
    daily = load_daily()
    kc = ev[ev["region"] == config.TARGET_REGION]
    saved = [
        national_count_oblast_only(ev),
        national_union_hours(ev),
        national_monthly_trend(ev),
        kc_daily_minutes(daily),
        kc_starts_by_hour(kc),
        kc_hour_weekday_heatmap(kc),
        kc_duration_distribution(kc),
        city_vs_oblast_monthly(ev),
    ]
    print(f"Saved {len(saved)} figures to {config.FIGURES}:")
    for name in saved:
        print(f"  {name}")
    _print_findings(ev, daily)


if __name__ == "__main__":
    main()
