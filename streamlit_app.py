"""Exploratory Streamlit dashboard for the air-raid alerts dataset.

EXPLORATORY tool only - not a final product, not an official alerting system,
not attack prediction. No model / baselines here.

Build the processed tables first, then launch::

    python -m src.pipeline
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import aggregate, config

EVENTS_PATH = config.DATA_PROCESSED / "events_canonical.csv"
DAILY_SLUG = config.TARGET_REGION.lower().replace(" ", "_")
DAILY_PATH = config.DATA_PROCESSED / f"daily_{DAILY_SLUG}.csv"
DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LEVELS = ["oblast", "raion", "hromada"]
EPISODE_DEF = (
    "An **alert episode** is one cleaned alert interval (start-end) for one exact "
    "unit, after removing exact duplicates and merging in-unit overlaps."
)

st.set_page_config(page_title="UA Air-Raid Alerts - Explorer", layout="wide")


def _coerce_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


@st.cache_data
def load_events() -> pd.DataFrame:
    df = pd.read_csv(EVENTS_PATH, parse_dates=["started_at_utc", "finished_at_utc"])
    for c in ["is_censored", "is_long_alert"]:
        df[c] = _coerce_bool(df[c])
    df["date_kyiv"] = pd.to_datetime(df["date_kyiv"])
    df["month"] = df["date_kyiv"].dt.to_period("M").dt.to_timestamp()
    return df


@st.cache_data
def load_daily() -> pd.DataFrame:
    return pd.read_csv(DAILY_PATH, parse_dates=["date"], index_col="date")


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


# --- header + guard ------------------------------------------------------
st.title("Ukraine Air-Raid Alerts - Data Explorer")
st.caption(
    "Exploratory analysis of historical alert *declarations*. Not an official "
    "alerting system, not attack prediction, not for safety decisions."
)

if not EVENTS_PATH.exists() or not DAILY_PATH.exists():
    st.error("Processed data not found. Build it first, then reload this page:")
    st.code("python -m src.pipeline", language="bash")
    st.stop()

events = load_events()
daily = load_daily()
min_day, max_day = events["date_kyiv"].dt.date.min(), events["date_kyiv"].dt.date.max()
all_regions = sorted(events["region"].unique())

# --- sidebar filters -----------------------------------------------------
st.sidebar.header("Filters")
dr = st.sidebar.date_input("Date range", (min_day, max_day), min_value=min_day, max_value=max_day)
if isinstance(dr, tuple) and len(dr) == 2:
    start_day, end_day = dr
else:
    start_day = end_day = dr[0] if isinstance(dr, tuple) else dr
levels_sel = st.sidebar.multiselect("Alert level", LEVELS, default=LEVELS)
regions_sel = st.sidebar.multiselect("Regions (Overview / National)", all_regions, default=all_regions)
top_n = st.sidebar.slider("Top N regions", 5, 25, 15)
st.sidebar.caption(EPISODE_DEF)
st.sidebar.caption(f"Pinned snapshot: {config.DATASET_COMMIT[:10]}")

# global filter (date + level), applied to every tab
dmask = (events["date_kyiv"].dt.date >= start_day) & (events["date_kyiv"].dt.date <= end_day)
f = events[dmask & events["level"].isin(levels_sel)]
f_reg = f[f["region"].isin(regions_sel)]  # + region filter for Overview/National

tab_ov, tab_nat, tab_kc, tab_cmp = st.tabs(
    ["Overview", "National", config.TARGET_REGION, "City vs Oblast"]
)

# --- Overview ------------------------------------------------------------
with tab_ov:
    st.subheader("Overview")
    st.caption(EPISODE_DEF + " Metrics below reflect the current filters.")
    kc = f_reg[f_reg["region"] == "Kyiv City"]
    ko = f_reg[f_reg["region"] == "Kyivska oblast"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alert episodes", f"{len(f_reg):,}")
    c2.metric("Regions", f_reg["region"].nunique())
    c3.metric("Kyiv City episodes", f"{len(kc):,}")
    c4.metric("Kyivska oblast episodes", f"{len(ko):,}")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Date from", str(start_day))
    c6.metric("Date to", str(end_day))
    c7.metric("Long alerts (>24h)", int(f_reg["is_long_alert"].sum()))
    c8.metric("Censored", int(f_reg["is_censored"].sum()))

# --- National ------------------------------------------------------------
with tab_nat:
    st.subheader("National")
    st.caption(f"Levels included: **{', '.join(levels_sel) or 'none'}**. "
               "Counts and row-sum hours add up episodes across the selected levels.")

    cnt = f_reg["region"].value_counts().head(top_n).sort_values()
    if len(cnt):
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(cnt))))
        ax.barh(cnt.index, cnt.values, color="#c0392b")
        ax.set_xlabel("number of alert episodes")
        ax.set_title(f"Top {min(top_n, len(cnt))} regions by number of alert episodes")
        show(fig)

    st.markdown("**Time under alert**")
    mode_labels = [
        "Oblast union time (any active alert) - recommended",
        "Oblast-level only",
        "All-levels row-sum (episode-hours)",
    ]
    mode_keys = ["union", "oblast_only", "row_sum"]
    choice = st.radio("Aggregation", mode_labels, index=0)
    mode = mode_keys[mode_labels.index(choice)]

    warn = {
        "union": "Union = wall-clock time at least one alert was active anywhere in "
                 "the region. Recommended: comparable across the oblast->raion shift.",
        "oblast_only": "Only oblast-wide declarations - **undercounts** regions where "
                       "alerts are mostly raion-level (especially after Dec 2025).",
        "row_sum": "Sum of every episode across levels - **overcounts** regions with "
                   "many simultaneous local (raion/hromada) alerts (~1.8-2.5x vs union).",
    }[mode]
    (st.info if mode == "union" else st.warning)(warn)

    hours = aggregate.region_hours(f, mode)
    hours = hours[hours.index.isin(regions_sel)].head(top_n).sort_values()
    if len(hours):
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(hours))))
        ax.barh(hours.index, hours.values, color="#e67e22")
        ax.set_xlabel("hours under alert")
        title_mode = {"union": "oblast union time", "oblast_only": "oblast-level only",
                      "row_sum": "all-levels row-sum"}[mode]
        ax.set_title(f"Top regions by alert hours ({title_mode})")
        show(fig)

    monthly = f_reg.groupby("month").size()
    if len(monthly):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(monthly.index, monthly.values, color="#2c3e50")
        ax.set_ylabel("episodes / month")
        ax.set_title("Monthly alert episodes (selected levels)")
        show(fig)

# --- Kyiv City -----------------------------------------------------------
with tab_kc:
    st.subheader(config.TARGET_REGION)
    st.caption("Kyiv City is a single oblast-level unit (no raion/hromada), so the "
               "level-aggregation issue does not apply here.")
    kc = f[f["region"] == config.TARGET_REGION]
    if kc.empty:
        st.info("No Kyiv City episodes for the current filters (check the level filter).")
    else:
        st.markdown("**Daily alert minutes**")
        st.line_chart(daily.loc[str(start_day):str(end_day), "alert_minutes"], height=240)

        col1, col2 = st.columns(2)
        with col1:
            by_hour = kc["hour_kyiv"].value_counts().reindex(range(24), fill_value=0).sort_index()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(by_hour.index, by_hour.values, color="#c0392b")
            ax.set_xlabel("hour of day (Kyiv)")
            ax.set_ylabel("alert starts")
            ax.set_title("Alert starts by hour")
            show(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(kc["duration_min"].clip(upper=600).dropna(), bins=40, color="#e67e22")
            ax.set_xlabel("duration (min, capped at 600)")
            ax.set_ylabel("count")
            ax.set_title("Alert duration distribution")
            show(fig)

        st.markdown("**Heatmap: hour x day-of-week (alert starts)**")
        piv = (
            kc.groupby(["dow_kyiv", "hour_kyiv"]).size()
            .unstack(fill_value=0)
            .reindex(index=range(7), columns=range(24), fill_value=0)
        )
        fig, ax = plt.subplots(figsize=(10, 3.2))
        im = ax.imshow(piv.values, aspect="auto", cmap="magma")
        ax.set_yticks(range(7))
        ax.set_yticklabels(DOW_LABELS)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels(range(0, 24, 2))
        ax.set_xlabel("hour (Kyiv)")
        fig.colorbar(im, ax=ax, label="alert starts")
        show(fig)

        st.markdown("**Inspect episodes** (verify the daily minutes against raw rows)")
        kc_days = kc["date_kyiv"].dt.date
        day = st.date_input("Episodes that started on", value=kc_days.max(),
                            min_value=kc_days.min(), max_value=kc_days.max(), key="kc_day")
        disp = ["started_at_kyiv", "finished_at_kyiv", "duration_min", "level", "source_event_count"]
        day_tbl = kc[kc_days == day][disp]
        st.caption(f"{len(day_tbl)} episode(s) started on {day}")
        st.dataframe(day_tbl, use_container_width=True)
        st.caption("Most recent 15 episodes:")
        st.dataframe(kc.sort_values("started_at_utc").tail(15)[disp], use_container_width=True)

# --- City vs Oblast ------------------------------------------------------
with tab_cmp:
    st.subheader("Kyiv City vs Kyivska oblast (distinct units)")
    st.caption("Two independent units: an alert in one does not imply an alert in the other.")
    sub = f[f["region"].isin(["Kyiv City", "Kyivska oblast"])]
    monthly = sub.groupby(["month", "region"]).size().unstack(fill_value=0)
    if not monthly.empty:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for reg, color in [("Kyiv City", "#c0392b"), ("Kyivska oblast", "#2980b9")]:
            if reg in monthly.columns:
                ax.plot(monthly.index, monthly[reg], label=reg, color=color)
        ax.set_ylabel("episodes / month")
        ax.set_title("Monthly alert episodes")
        ax.legend()
        show(fig)
        st.caption("Last 12 months (episode counts):")
        st.dataframe(monthly.tail(12), use_container_width=True)
