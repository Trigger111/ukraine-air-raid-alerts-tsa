"""Phase 2/3 — event-level table -> regular time series, parameterised by region.

``daily_region`` builds a gap-free daily index (missing days filled with 0) with:
    alert_count, total_alert_min, max_alert_min, any_alert, night_alert_count.

``hourly_region`` builds an overlap-based hourly table — used for EDA in the MVP;
an hourly *model* is future work.
"""
from __future__ import annotations

import pandas as pd

from . import config


def daily_region(events: pd.DataFrame, region: str = config.TARGET_REGION) -> pd.DataFrame:
    """Daily aggregate series for a single region (default: config.TARGET_REGION)."""
    raise NotImplementedError("Phase 2/3: daily aggregation")


def hourly_region(events: pd.DataFrame, region: str = config.TARGET_REGION) -> pd.DataFrame:
    """Hourly overlap-based table for a single region (EDA; model = future work)."""
    raise NotImplementedError("EDA/future work: hourly aggregation")
