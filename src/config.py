"""Central configuration — the single source of truth.

No region name, path, or data-source detail is hardcoded anywhere outside this
module. Switching the forecasting unit to another region is a one-line change
(``TARGET_REGION``); the rest of the pipeline is parameterised and reads from
here.
"""
from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

# --- Pinned data source (reproducibility) ---------------------------------
DATASET_REPO = "Vadimkin/ukrainian-air-raid-sirens-dataset"
DATASET_COMMIT = "dac507f80fe59ef62d67e80be1e6b9558f126b33"
DATASET_PATH = "datasets/official_data_en.csv"
RAW_URL = (
    f"https://raw.githubusercontent.com/{DATASET_REPO}/"
    f"{DATASET_COMMIT}/{DATASET_PATH}"
)
# SHA-256 of the pinned snapshot; data_loader verifies any download against this.
RAW_SHA256 = "a36eb2fac7606765b67967f4b70be0a2c93d15710972c01e27427be128ab7d58"

# --- Timezone policy: store UTC, analyse in Kyiv local time ----------------
TZ_UTC = ZoneInfo("UTC")
TZ_LOCAL = ZoneInfo("Europe/Kyiv")

# --- Regions (SINGLE SOURCE OF TRUTH) -------------------------------------
TARGET_REGION = "Kyiv City"                            # forecasting MVP unit
COMPARISON_REGIONS = ["Kyiv City", "Kyivska oblast"]   # distinct-unit comparison
EDA_REGIONS: list[str] | None = None                   # None -> all regions (national EDA)

# --- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RAW_FILE = DATA_RAW / "official_data_en.csv"
META_FILE = ROOT / "data" / "meta.json"

# --- Reproducibility ------------------------------------------------------
RANDOM_SEED = 42
