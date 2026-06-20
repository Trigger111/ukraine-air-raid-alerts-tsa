# Reflection — KSE Stage 2

Project: Time Series Analysis of air raid alerts in Ukraine

**What went wrong?** My first analytical metrics misled me. Summing alert
durations across oblast/raion/hromada overcounted regional time, and an
oblast-level monthly trend implied alerts had collapsed in late 2025 — when in
fact only the declaration *granularity* changed (a structural break already
present in the source data, alongside ~42% duplicate rows).

**How did I adjust?** I verified each finding against the raw data instead of
trusting the first chart, switched to granularity-invariant metrics (oblast
union time, region-alert-days), pinned the dataset by commit hash, and built a
leakage-safe, baseline-first evaluation on a chronological split.

**Why is the final version better?** It is reproducible and honest: neither a
simple model nor a stronger gradient-boosting one beats the naive baselines —
the stronger one overfits and does worse — so I report that limit instead of
inflating a number.
