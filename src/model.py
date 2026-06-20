"""Phase 6 - one lean, explainable model vs the Phase 5 baselines.

LogisticRegression (standardised numeric + one-hot calendar) on the SAME
chronological 80/20 split and the SAME metrics as the baselines. The scaler and
encoder are fit on TRAIN only (no leakage). "Success" = beat the recency
baselines on ROC-AUC AND base-rate-constant on Brier, together.

Deliberately simple: no neural nets, no hourly/multi-region model, no test-set
tuning. Run::

    python -m src.model
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import baselines, config, evaluate
from .features import FEATURES

NUM = ["has_alert_today", "has_alert_lag1", "has_alert_lag6", "has_alert_lag7",
       "has_alert_roll7", "alert_minutes_today", "alert_minutes_roll7", "next_is_weekend"]
CAT = ["next_dow", "next_month"]


def build_model() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT),
    ])
    clf = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def _coef_table(model: Pipeline) -> pd.Series:
    names = [n.split("__", 1)[-1] for n in model.named_steps["pre"].get_feature_names_out()]
    coef = model.named_steps["clf"].coef_[0]
    return pd.Series(coef, index=names).sort_values(key=np.abs, ascending=False)


def main() -> None:
    slug = config.TARGET_REGION.lower().replace(" ", "_")
    sup = pd.read_csv(config.DATA_PROCESSED / f"supervised_{slug}.csv",
                      parse_dates=["date"], index_col="date")
    daily = pd.read_csv(config.DATA_PROCESSED / f"daily_{slug}.csv",
                        parse_dates=["date"], index_col="date")
    daily_has = evaluate._daily_has(daily)

    train, test = evaluate.chronological_split(sup, 0.2)
    y_train, y_test = train["has_alert_next_day"], test["has_alert_next_day"].to_numpy()

    model = build_model()
    model.fit(train[FEATURES], y_train)  # scaler + encoder fit on TRAIN only
    prob = model.predict_proba(test[FEATURES])[:, 1]

    rows = {name: evaluate.classification_metrics(y_test, p.to_numpy())
            for name, p in baselines.predict_all(train, test, daily_has).items()}
    rows["logreg_model"] = evaluate.classification_metrics(y_test, prob)
    table = pd.DataFrame(rows).T[evaluate.METRIC_COLS].round(3)
    table.to_csv(config.DATA_PROCESSED / "model_vs_baselines.csv")

    print("=" * 74)
    print(f"MODEL vs BASELINES - {config.TARGET_REGION}  (chronological 80/20)")
    print("=" * 74)
    print(table.to_string())
    print("-" * 74)

    best_recency_auc = max(table.loc["rolling_7d", "roc_auc"], table.loc["rolling_30d", "roc_auc"])
    base_brier = table.loc["base_rate_constant", "brier"]
    m_auc, m_brier = table.loc["logreg_model", "roc_auc"], table.loc["logreg_model", "brier"]
    beats_auc = m_auc > best_recency_auc
    beats_brier = m_brier < base_brier
    print(f"ROC-AUC: model {m_auc:.3f} vs best recency {best_recency_auc:.3f} -> "
          f"{'BEATS' if beats_auc else 'does NOT beat'}")
    print(f"Brier  : model {m_brier:.3f} vs base-rate {base_brier:.3f} -> "
          f"{'BEATS' if beats_brier else 'does NOT beat'}")
    print(f"VERDICT: {'useful (beats both)' if beats_auc and beats_brier else 'NOT a clear win - report honestly'}")
    print("-" * 74)
    print("Top coefficients (standardised; + raises tomorrow's alert probability):")
    print(_coef_table(model).head(8).round(3).to_string())


if __name__ == "__main__":
    main()
