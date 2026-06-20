"""Load the pinned official dataset snapshot.

Reproducibility contract:
  * Always read the snapshot pinned in :mod:`config` (commit + SHA-256).
  * If a local copy exists and its hash matches, use it (offline-friendly).
  * Otherwise download from the pinned commit URL, verify the hash, cache it.

This module is data *infrastructure*, not analysis — it performs no cleaning.
Run directly for a quick smoke test::

    py -m src.data_loader
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import requests

from . import config


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_raw(force: bool = False) -> Path:
    """Return the path to the verified local snapshot, downloading if needed."""
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    path = config.RAW_FILE

    if path.exists() and not force:
        digest = _sha256(path)
        if config.RAW_SHA256 is None or digest == config.RAW_SHA256:
            return path
        raise ValueError(
            f"Local snapshot hash mismatch for {path.name}: {digest} "
            f"!= pinned {config.RAW_SHA256}. Delete the file or call "
            f"ensure_raw(force=True) to re-download the pinned commit."
        )

    resp = requests.get(config.RAW_URL, timeout=120)
    resp.raise_for_status()
    path.write_bytes(resp.content)

    digest = _sha256(path)
    if config.RAW_SHA256 is not None and digest != config.RAW_SHA256:
        raise ValueError(
            f"Downloaded snapshot hash {digest} != pinned {config.RAW_SHA256}."
        )
    return path


def load_raw(force: bool = False) -> pd.DataFrame:
    """Load the raw official alerts snapshot (timestamps parsed as UTC-aware)."""
    path = ensure_raw(force=force)
    return pd.read_csv(path, parse_dates=["started_at", "finished_at"])


if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {len(df):,} rows; columns: {list(df.columns)}")
    print(df.head())
