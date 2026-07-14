"""
pipeline.py

A batch ETL pipeline built on REAL data: the UCI SECOM semiconductor
manufacturing dataset (McCann, M. & Johnston, A., 2008 -- UCI Machine
Learning Repository, https://doi.org/10.24432/C54305).

This is real production data from an actual wafer fabrication line: 1,567
manufacturing runs recorded between Jan-Dec 2008, each with real sensor
readings and a real pass/fail outcome. The dataset's 590 sensor columns are
anonymized (numbered, not named) in the original release to protect
proprietary process details -- a common real-world constraint, not
something introduced here.

Why only 12 of the 590 sensor columns are used: with genuinely anonymous
sensor IDs and no documented meaning, using all 590 would add noise without
adding clarity for a portfolio project. The 12 kept here were selected by
an explicit, reproducible rule: lowest missing-data percentage (<2% missing)
and highest variance (most informative signal) -- see
`select_sensor_columns()` below. This mirrors real sensor-feature-selection
practice, which is in fact one of the stated goals of the original SECOM
research.

Design choice: ETL, not ELT (see README for the full reasoning) -- raw data
is cleaned and validated before being loaded into the shared database, since
this data has real missing values and irregular timestamps that must be
resolved before anything downstream reads it.

Pipeline stages:
    1. EXTRACT  - read the real SECOM CSV, select the 12 best sensor columns
    2. VALIDATE - real data quality checks: missing values (already present
                  in the raw data, not injected), timestamp ordering/
                  duplicate checks, freshness gaps (real gaps up to 28 days
                  exist in this dataset), statistical outlier detection
                  (no physical sensor bounds exist since sensors are
                  anonymized, so an IQR-based statistical check is used
                  instead)
    3. TRANSFORM- interpolate missing values, engineer rolling time-series
                  features (rolling mean/std, rate of change)
    4. LOAD     - write into a star-schema SQLite database:
                  fact_process_readings (fact table)
                  dim_sensor            (dimension table: which sensors kept & why)
                  data_quality_log      (pipeline observability)
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "raw_data" / "secom_real.csv"
DB_PATH = BASE_DIR / "output" / "industrial.db"
DB_PATH.parent.mkdir(exist_ok=True)

N_SENSORS = 12
MAX_MISSING_PCT = 0.02          # keep only sensors with <2% missing in raw data
FRESHNESS_GAP_THRESHOLD = pd.Timedelta(hours=6)   # flag any gap longer than this

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# 1. EXTRACT
# ---------------------------------------------------------------------------
def select_sensor_columns(df: pd.DataFrame) -> list[str]:
    """Reproducible rule for picking which of the 590 anonymous sensors to
    keep: lowest missing-data %, then highest variance among those."""
    sensor_cols = [c for c in df.columns if c not in ("Time", "Pass/Fail")]
    missing_pct = df[sensor_cols].isna().mean()
    variance = df[sensor_cols].var(numeric_only=True)
    low_missing = [c for c in missing_pct[missing_pct < MAX_MISSING_PCT].index
                   if variance.get(c, 0) > 0]
    top = variance[low_missing].sort_values(ascending=False).head(N_SENSORS)
    return top.index.tolist()


def extract() -> tuple[pd.DataFrame, list[str]]:
    log.info(f"Extracting real SECOM data from {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, parse_dates=["Time"])
    sensor_cols = select_sensor_columns(df)
    log.info(f"Selected {len(sensor_cols)} of {df.shape[1] - 2} raw sensor columns "
              f"(rule: <{MAX_MISSING_PCT:.0%} missing, highest variance)")

    keep = ["Time"] + sensor_cols + ["Pass/Fail"]
    df = df[keep].rename(columns={c: f"sensor_{i+1:02d}" for i, c in enumerate(sensor_cols)})
    df = df.rename(columns={"Time": "timestamp", "Pass/Fail": "pass_fail_raw"})
    df["passed"] = (df["pass_fail_raw"] == -1).astype(int)  # dataset encodes -1=pass, 1=fail
    df = df.drop(columns=["pass_fail_raw"])

    log.info(f"Extracted {len(df)} real manufacturing process records "
              f"({df['passed'].eq(0).sum()} real failures, {df['passed'].eq(1).sum()} passes)")
    return df, [f"sensor_{i+1:02d}" for i in range(len(sensor_cols))]


# ---------------------------------------------------------------------------
# 2. VALIDATE
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame, sensor_cols: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    quality_log = []

    # --- Duplicate timestamps (real production data can log two runs at once) ---
    n_dupes = int(df.duplicated(subset=["timestamp"]).sum())
    if n_dupes:
        quality_log.append({"check": "duplicate_timestamps", "severity": "warning",
                             "count": n_dupes, "action": "kept, both are real distinct runs"})

    # --- Missing values (these are REAL, already in the raw data) ---
    for col in sensor_cols:
        n_null = int(df[col].isna().sum())
        if n_null:
            quality_log.append({"check": f"null_{col}", "severity": "warning",
                                 "count": n_null, "action": "flagged for interpolation"})

    # --- Statistical outlier check (IQR-based, since sensors are anonymized
    #     and no physical/spec bounds are known -- a real constraint of
    #     working with anonymized industrial data) ---
    for col in sensor_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr   # 3x IQR = conservative extreme-outlier bound
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n_out:
            quality_log.append({"check": f"outlier_{col}", "severity": "warning",
                                 "count": n_out, "action": "flagged (kept, not removed -- "
                                 "real process extremes can be meaningful for failure prediction)"})

    # --- Freshness / gap checks (real gaps exist in this data, up to 28 days) ---
    df_sorted = df.sort_values("timestamp")
    gaps = df_sorted["timestamp"].diff()
    big_gaps = gaps[gaps > FRESHNESS_GAP_THRESHOLD]
    if len(big_gaps):
        quality_log.append({"check": "freshness_gaps", "severity": "warning",
                             "count": len(big_gaps),
                             "action": f"largest real gap: {big_gaps.max()}"})

    for entry in quality_log:
        log.info(f"  [{entry['severity'].upper()}] {entry['check']}: {entry['count']} -> {entry['action']}")

    return df, quality_log


# ---------------------------------------------------------------------------
# 3. TRANSFORM
# ---------------------------------------------------------------------------
def transform(df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    df = df.sort_values("timestamp").set_index("timestamp")

    # Interpolate real missing sensor values (time-aware)
    for col in sensor_cols:
        df[col] = df[col].interpolate(method="time").ffill().bfill()

    # --- Feature engineering used by downstream ML models ---
    for col in sensor_cols[:4]:  # engineer features for the top 4 highest-variance sensors
        df[f"{col}_roll_mean_10"] = df[col].rolling(10, min_periods=1).mean()
        df[f"{col}_rate_of_change"] = df[col].diff()

    result = df.reset_index()
    log.info(f"Transformed {len(result)} rows; engineered rolling/derived features on top 4 sensors")
    return result


# ---------------------------------------------------------------------------
# 4. LOAD
# ---------------------------------------------------------------------------
def load(df: pd.DataFrame, sensor_cols: list[str], quality_log: list[dict]):
    conn = sqlite3.connect(DB_PATH)
    try:
        dim_sensor = pd.DataFrame({
            "sensor_id": sensor_cols,
            "selection_rule": "lowest missing % (<2%) among 590 raw sensors, then highest variance",
            "note": "original sensor identity anonymized by UCI SECOM dataset for confidentiality",
        })
        dim_sensor.to_sql("dim_sensor", conn, if_exists="replace", index=False)
        df.to_sql("fact_process_readings", conn, if_exists="replace", index=False)
        pd.DataFrame(quality_log).to_sql("data_quality_log", conn, if_exists="replace", index=False)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_fact_time ON fact_process_readings(timestamp)")
        conn.commit()
        log.info(f"Loaded schema into {DB_PATH}")
        log.info("  Tables: dim_sensor, fact_process_readings, data_quality_log")
    finally:
        conn.close()


def run():
    log.info("=== Starting pipeline run (real SECOM data) ===")
    df, sensor_cols = extract()
    df, quality_log = validate(df, sensor_cols)
    df = transform(df, sensor_cols)
    load(df, sensor_cols, quality_log)
    log.info("=== Pipeline run complete ===")


if __name__ == "__main__":
    run()
