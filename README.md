# Ukraine Air-Raid Alerts — Time Series Analysis

> ⚠️ **Disclaimer.** This is a **research / educational** project that analyses
> *historical patterns* of air-raid **alert declarations** in Ukraine. It is
> **NOT** an official alerting system and **must not** be used for operational,
> safety, or evacuation decisions. It does **not** predict attacks. An *alert*
> (тривога) is an administrative warning signal — **not** a physical strike;
> attacks are adversarial and inherently unpredictable. Always rely on official
> channels (e.g. official government apps, local authorities).

## What this is / is not

- ✅ Retrospective EDA of alert patterns (hour-of-day, weekday, seasonality, duration).
- ✅ A transparent, **baseline-first** forecasting experiment of *alert activity*.
- ❌ Not an early-warning system. ❌ Not attack prediction. ❌ Not safety-critical.

## Scope

The current forecasting MVP focuses on **Kyiv City** (a clean, well-balanced
unit). EDA is performed at the **national level across all available Ukrainian
regions**, with an explicit comparison between **Kyiv City** and **Kyiv Oblast**
as distinct units. The pipeline is **parameterised** (`config.TARGET_REGION`)
and **can be extended to other regions**; multi-region modelling is planned as
**future work**.

*Why Kyiv City for the model?* The primary target is binary (`has_alert_next_day`).
At the national level "any alert somewhere in Ukraine tomorrow" is almost always
*yes* (degenerate); for quiet regions it is almost always *no*. Kyiv City sits in
the class-balance sweet spot, so the classification task is meaningful.

## Data

- **Source:** [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset),
  file `datasets/official_data_en.csv` (**official** source only).
- **Pinned commit:** `dac507f80fe59ef62d67e80be1e6b9558f126b33`, retrieved 2026-06-20.
- **Size:** 271,160 alert events · 2022-03-15 → 2026-06-20 · 25 top-level regions.
- **Columns:** `oblast, raion, hromada, level, started_at, finished_at, source`.
- **Timezone:** timestamps are stored in **UTC**; converted to **Europe/Kyiv**
  (with DST) for all local-time analysis.
- **License:** not stated upstream; attributed to the dataset author and used
  *as-is* for research/education.
- **Volunteer data is intentionally excluded** from the MVP: it has a different
  schema (region-level only) and an imputed `naive` flag — `naive=True` records
  use a placeholder end time of *start + 30 min* (the real end was not observed),
  which would corrupt duration analysis. (Note: this dataset `naive` flag is
  unrelated to the *naive baselines* used in modelling — see Methodology.)

## Reproducibility

- Python ≥ 3.11 (uses the stdlib `zoneinfo`). Install deps:
  `pip install -r requirements.txt`.
- The raw snapshot is **not committed**. `src/data_loader.py` fetches it from the
  pinned commit and verifies it against the SHA-256 recorded in
  [`data/meta.json`](data/meta.json) (`a36eb2fa…ab7d58`).
- Fixed `RANDOM_SEED = 42` (`src/config.py`).
- Smoke test the data layer: `py -m src.data_loader`.

## Project structure

```
.
├── README.md
├── requirements.txt
├── LICENSE                  # MIT (project code); dataset license: see Data
├── data/
│   ├── raw/                 # pinned snapshot (git-ignored, fetched by loader)
│   ├── processed/           # generated event/daily/hourly tables (git-ignored)
│   └── meta.json            # source, pinned commit, SHA-256, row count
├── src/
│   ├── config.py            # single source of truth (regions, paths, TZ, seed)
│   ├── data_loader.py       # fetch + verify pinned snapshot (no analysis)
│   ├── clean.py             # raw -> canonical event-level table
│   ├── aggregate.py         # event -> daily / hourly tables (per region)
│   ├── features.py          # past-only supervised features (post-split)
│   ├── baselines.py         # persistence / rolling-mean / seasonal-naive
│   └── evaluate.py          # chronological split + metrics
├── notebooks/               # 01_eda, 02_modeling
└── figures/                 # saved plots
```

## Methodology

- **EDA:** national overview (all regions) + Kyiv City hourly patterns
  (hour-of-day, `hour × weekday` heatmap, durations) + Kyiv City vs Kyiv Oblast.
- **Target (daily, Kyiv City):** primary `has_alert_next_day` (classification);
  optional `alert_minutes_next_day` (regression).
- **Naive baselines:** persistence (*yesterday*), 7-day rolling mean,
  same-weekday-last-week.
- **Models:** `LogisticRegression` / `HistGradientBoosting` (deliberately simple).
- **Evaluation:** chronological split / `TimeSeriesSplit`.
  Classification — F1, ROC-AUC, Brier (+ base rate). Regression — MAE, RMSE.
  Always reported against the baselines.
- **Leakage controls:** chronological (never random) split; past-only features
  built *after* the split; `finished_at`/duration never used as a next-day feature.

## Results

_TBD — populated after modelling._

## Limitations

- Alerts ≠ attacks (administrative signal).
- Non-stationarity: war intensity and tactics change over time.
- Right-censoring of alerts still ongoing at the snapshot boundary.
- Upstream ingestion delay — the dataset is **not real-time**.
- Granularity shift: increasingly raion-level alerts since Dec 2025.

## Future work

- Multi-region modelling (pipeline already parameterised via `config.TARGET_REGION`).
- Hourly classification (class imbalance + block-aware CV).
- Incorporate the volunteer dataset (after handling `naive` imputation) for
  earlier coverage (Feb–Mar 2022) and cross-source validation.
- Automation: scheduled refresh of the pinned snapshot + a live dashboard.

## References

- Related work: *Predictive Analytics of Air Alerts in the Russian-Ukrainian War*
  ([arXiv:2411.14625](https://arxiv.org/abs/2411.14625)).

## Development

Built using AI (Claude) as the primary engineering tool; the full AI interaction
log is part of the submission, as required by the assignment.

## License

Project code: **MIT** (see [`LICENSE`](LICENSE)). Dataset: © its authors
(see the **Data** section).
