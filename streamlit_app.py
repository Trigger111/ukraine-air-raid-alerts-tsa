"""Exploratory Streamlit dashboard for the air-raid alerts dataset.

EXPLORATORY tool only — not a final product, not an official alerting system,
not attack prediction. No model / baselines here.

Build the processed tables first, then launch::

    python -m src.pipeline
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config

EVENTS_PATH = config.DATA_PROCESSED / "events_canonical.csv"
DAILY_SLUG = config.TARGET_REGION.lower().replace(" ", "_")
DAILY_PATH = config.DATA_PROCESSED / f"daily_{DAILY_SLUG}.csv"
DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.set_page_config(page_title="UA Air-Raid Alerts - Explorer", layout="wide")


def _coerce_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


@st.cache_data
def load_events() -> pd.DataFrame:
    df = pd.read_csv(EVENTS_PATH)
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


# --- guard: processed files must exist -----------------------------------
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

top_n = st.sidebar.slider("Top N regions", 5, 25, 15)
st.sidebar.caption(f"Pinned snapshot: {config.DATASET_COMMIT[:10]}")
st.sidebar.caption("Exploratory dashboard (Phase 3A). No model here.")

tab_ov, tab_nat, tab_kc, tab_cmp = st.tabs(
    ["Overview", "National", config.TARGET_REGION, "City vs Oblast"]
)

# --- Overview ------------------------------------------------------------
with tab_ov:
    st.subheader("Overview")
    kc = events[events["region"] == "Kyiv City"]
    ko = events[events["region"] == "Kyivska oblast"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", f"{len(events):,}")
    c2.metric("Regions", events["region"].nunique())
    c3.metric("Kyiv City events", f"{len(kc):,}")
    c4.metric("Kyivska oblast events", f"{len(ko):,}")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Date from", str(events["date_kyiv"].min().date()))
    c6.metric("Date to", str(events["date_kyiv"].max().date()))
    c7.metric("Long alerts (>24h)", int(events["is_long_alert"].sum()))
    c8.metric("Censored", int(events["is_censored"].sum()))

# --- National ------------------------------------------------------------
with tab_nat:
    st.subheader("National")

    cnt = events["region"].value_counts().head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(cnt))))
    ax.barh(cnt.index, cnt.values, color="#c0392b")
    ax.set_xlabel("event count")
    ax.set_title(f"Top {top_n} regions by event count")
    show(fig)

    mins = (
        events.loc[~events["is_long_alert"]]
        .groupby("region")["duration_min"].sum().div(60)
        .sort_values(ascending=False).head(top_n).sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(mins))))
    ax.barh(mins.index, mins.values, color="#e67e22")
    ax.set_xlabel("total alert hours (excluding is_long_alert outliers)")
    ax.set_title(f"Top {top_n} regions by total alert hours")
    show(fig)

    monthly = events.groupby("month").size()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(monthly.index, monthly.values, color="#2c3e50")
    ax.set_ylabel("events / month")
    ax.set_title("National monthly event count")
    show(fig)

# --- Kyiv City -----------------------------------------------------------
with tab_kc:
    st.subheader(config.TARGET_REGION)
    kc = events[events["region"] == config.TARGET_REGION]

    st.markdown("**Daily alert minutes**")
    st.line_chart(daily["alert_minutes"], height=240)

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
        dur = kc["duration_min"].clip(upper=600)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(dur.dropna(), bins=40, color="#e67e22")
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

# --- City vs Oblast ------------------------------------------------------
with tab_cmp:
    st.subheader("Kyiv City vs Kyivska oblast (distinct units)")
    sub = events[events["region"].isin(["Kyiv City", "Kyivska oblast"])]
    monthly = sub.groupby(["month", "region"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for reg, color in [("Kyiv City", "#c0392b"), ("Kyivska oblast", "#2980b9")]:
        if reg in monthly.columns:
            ax.plot(monthly.index, monthly[reg], label=reg, color=color)
    ax.set_ylabel("events / month")
    ax.set_title("Monthly alert events")
    ax.legend()
    show(fig)
    st.caption("Last 12 months (event counts):")
    st.dataframe(monthly.tail(12))
