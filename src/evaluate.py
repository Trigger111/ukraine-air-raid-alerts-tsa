"""Phase 5 - chronological evaluation harness for the baselines.

Single chronological hold-out (no shuffle): train = first 80%, test = last 20%.
Probabilistic metrics (ROC-AUC, Brier) use the probability outputs; hard metrics
(accuracy / precision / recall / F1) use a fixed 0.5 threshold. Every baseline is
judged by SKILL over majority / base-rate / persistence, not absolute accuracy.

Run::

    python -m src.evaluate
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)

from . import baselines, config

THRESHOLD = 0.5
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc", "brier"]


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """Split a time-indexed frame into earliest train and most-recent test."""
    df = df.sort_index()
    n_test = int(round(len(df) * test_fraction))
    return df.iloc[:-n_test], df.iloc[-n_test:]


def classification_metrics(y_true, prob, threshold: float = THRESHOLD) -> dict:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "brier": brier_score_loss(y_true, prob),
    }
    # ROC-AUC only where the score actually varies (a real probability).
    if np.unique(prob).size > 1 and np.unique(y_true).size > 1:
        out["roc_auc"] = roc_auc_score(y_true, prob)
    else:
        out["roc_auc"] = np.nan
    return out


def regression_metrics(y_true, y_pred) -> dict:
    """MAE / RMSE (for the optional alert_minutes target; unused in Phase 5)."""
    err = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean()))}


def _daily_has(daily: pd.DataFrame) -> pd.Series:
    h = daily["has_alert"]
    if h.dtype != bool:
        h = h.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    return h.astype(int)


def main() -> None:
    slug = config.TARGET_REGION.lower().replace(" ", "_")
    sup = pd.read_csv(config.DATA_PROCESSED / f"supervised_{slug}.csv",
                      parse_dates=["date"], index_col="date")
    daily = pd.read_csv(config.DATA_PROCESSED / f"daily_{slug}.csv",
                        parse_dates=["date"], index_col="date")
    daily_has = _daily_has(daily)

    train, test = chronological_split(sup, 0.2)
    y_test = test["has_alert_next_day"].to_numpy()

    preds = baselines.predict_all(train, test, daily_has)
    table = pd.DataFrame(
        {name: classification_metrics(y_test, prob.to_numpy()) for name, prob in preds.items()}
    ).T[METRIC_COLS].round(3)

    out_path = config.DATA_PROCESSED / "baseline_metrics.csv"
    table.to_csv(out_path)

    print("=" * 74)
    print(f"BASELINES - {config.TARGET_REGION}  (chronological 80/20 hold-out)")
    print("=" * 74)
    print(f"train: {len(train)} rows  {train.index.min().date()}..{train.index.max().date()}  "
          f"base rate {train['has_alert_next_day'].mean():.3f}")
    print(f"test : {len(test)} rows  {test.index.min().date()}..{test.index.max().date()}  "
          f"base rate {test['has_alert_next_day'].mean():.3f}")
    print("-" * 74)
    print(table.to_string())
    print("-" * 74)
    print("Read as SKILL: a useful model must beat majority/base-rate AND persistence")
    print("on F1 / ROC-AUC / Brier together (not on accuracy alone).")
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
