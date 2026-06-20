"""Phase 6 - lean explainable model (LogReg) + Phase 6B stronger sanity-check
(HistGradientBoosting) vs the Phase 5 baselines.

Same chronological 80/20 split, same features/target/metrics. No grid search, no
test-set tuning. Goal of 6B: check whether extra model capacity (non-linearities
and interactions) finds any signal beyond the naive baselines. If even a gradient
boosting model does not beat them, the limited predictability is real, not an
artefact of using too simple a model.

Run::

    python -m src.model
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import baselines, config, evaluate
from .features import FEATURES

NUM = ["has_alert_today", "has_alert_lag1", "has_alert_lag6", "has_alert_lag7",
       "has_alert_roll7", "alert_minutes_today", "alert_minutes_roll7", "next_is_weekend"]
CAT = ["next_dow", "next_month"]


def build_logreg() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED))])


def build_gbm() -> HistGradientBoostingClassifier:
    # Default hyper-parameters on purpose (no grid search, no test tuning).
    return HistGradientBoostingClassifier(random_state=config.RANDOM_SEED)


def _logreg_coefs(model: Pipeline) -> pd.Series:
    names = [n.split("__", 1)[-1] for n in model.named_steps["pre"].get_feature_names_out()]
    return pd.Series(model.named_steps["clf"].coef_[0], index=names).sort_values(key=np.abs, ascending=False)


def _gbm_importance(model, X_test, y_test) -> pd.Series:
    r = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                               n_repeats=5, random_state=config.RANDOM_SEED)
    return pd.Series(r.importances_mean, index=FEATURES).sort_values(key=np.abs, ascending=False)


def _verdict(name, table, best_recency_auc, base_brier):
    a, b = table.loc[name, "roc_auc"], table.loc[name, "brier"]
    beats_auc, beats_brier = a > best_recency_auc, b < base_brier
    tag = "useful (beats both)" if beats_auc and beats_brier else "NOT a clear win"
    print(f"  {name:<14} ROC-AUC {a:.3f} ({'beats' if beats_auc else 'no'} vs {best_recency_auc:.3f}) | "
          f"Brier {b:.3f} ({'beats' if beats_brier else 'no'} vs {base_brier:.3f}) -> {tag}")


def main() -> None:
    slug = config.TARGET_REGION.lower().replace(" ", "_")
    sup = pd.read_csv(config.DATA_PROCESSED / f"supervised_{slug}.csv",
                      parse_dates=["date"], index_col="date")
    daily = pd.read_csv(config.DATA_PROCESSED / f"daily_{slug}.csv",
                        parse_dates=["date"], index_col="date")
    daily_has = evaluate._daily_has(daily)

    train, test = evaluate.chronological_split(sup, 0.2)
    y_train, y_test = train["has_alert_next_day"], test["has_alert_next_day"].to_numpy()

    logreg = build_logreg().fit(train[FEATURES], y_train)
    gbm = build_gbm().fit(train[FEATURES], y_train)
    p_lr = logreg.predict_proba(test[FEATURES])[:, 1]
    p_gb = gbm.predict_proba(test[FEATURES])[:, 1]

    rows = {name: evaluate.classification_metrics(y_test, p.to_numpy())
            for name, p in baselines.predict_all(train, test, daily_has).items()}
    rows["logreg_model"] = evaluate.classification_metrics(y_test, p_lr)
    rows["histgbm_model"] = evaluate.classification_metrics(y_test, p_gb)
    table = pd.DataFrame(rows).T[evaluate.METRIC_COLS].round(3)
    table.to_csv(config.DATA_PROCESSED / "model_vs_baselines.csv")

    best_recency_auc = max(table.loc["rolling_7d", "roc_auc"], table.loc["rolling_30d", "roc_auc"])
    base_brier = table.loc["base_rate_constant", "brier"]

    print("=" * 74)
    print(f"MODEL vs BASELINES - {config.TARGET_REGION}  (chronological 80/20)")
    print("=" * 74)
    print(table.to_string())
    print("-" * 74)
    print(f"success rule: beat recency ROC-AUC (>{best_recency_auc:.3f}) AND base-rate Brier (<{base_brier:.3f})")
    _verdict("logreg_model", table, best_recency_auc, base_brier)
    _verdict("histgbm_model", table, best_recency_auc, base_brier)
    print("-" * 74)
    print("LogReg top coefficients (standardised; + raises tomorrow's alert prob):")
    print(_logreg_coefs(logreg).head(6).round(3).to_string())
    print("\nHistGBM permutation importance (drop in test ROC-AUC when shuffled):")
    print(_gbm_importance(gbm, test[FEATURES], y_test).head(6).round(4).to_string())


if __name__ == "__main__":
    main()
