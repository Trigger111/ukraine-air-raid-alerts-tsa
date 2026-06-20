"""Phase 5/6 — time-aware evaluation.

Splitting: chronological hold-out + ``TimeSeriesSplit`` (never random/k-fold —
that would leak future into past via autocorrelation).

Metrics:
    classification (has_alert_next_day): F1, ROC-AUC, Brier  (+ base rate)
    regression     (alert_minutes_next_day): MAE, RMSE

Model metrics are always reported next to the naive baselines (:mod:`baselines`).
"""
from __future__ import annotations

import pandas as pd

from . import config  # noqa: F401  (used once implemented)


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """Split a time-indexed frame into past (train) and most-recent (test)."""
    raise NotImplementedError("Phase 5: chronological split")


def classification_metrics(y_true, y_prob) -> dict:
    """Return F1, ROC-AUC, Brier and base rate."""
    raise NotImplementedError("Phase 5: classification metrics")


def regression_metrics(y_true, y_pred) -> dict:
    """Return MAE and RMSE."""
    raise NotImplementedError("Phase 6: regression metrics")
