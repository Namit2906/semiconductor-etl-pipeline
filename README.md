# Semiconductor Manufacturing — Real-Data Predictive Maintenance Pipeline

A single end-to-end batch data pipeline built entirely on **real** industrial
data — no simulated or synthetic data anywhere in this project.

One dataset, five steps, start to finish:
**Extract → Validate → Transform → Load → Analyze/Visualize**

## The dataset

**UCI SECOM Semiconductor Manufacturing Dataset**
McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C54305

This is real production data from an actual wafer fabrication line:
**1,567 real manufacturing runs**, recorded between January and December
2008, each with real sensor readings (590 raw sensor columns) and a real
recorded pass/fail outcome (104 real failures, a realistic ~6.6% failure
rate). Licensed CC BY 4.0 for open use.

The 590 sensor columns are anonymized (numbered, not named) in the original
dataset release to protect proprietary manufacturing process details — a
real-world confidentiality constraint, not something introduced by this
project.

## Why only 12 of the 590 sensor columns are used

Using all 590 anonymous, undocumented sensor columns would add noise
without adding clarity for a readable portfolio pipeline. Instead, 12
sensors were selected using an explicit, reproducible rule (see
`select_sensor_columns()` in `src/pipeline.py`):

1. Keep only sensors with **<2% missing values** in the raw data
2. Among those, keep the **12 with the highest variance** (most informative
   signal)

This mirrors real feature-selection practice — reducing a high-dimensional
sensor set to the most useful signals is in fact one of the stated research
goals of the original SECOM dataset.

## The 5 steps

```
raw_data/secom_real.csv (real, downloaded, unmodified)
     │
     ▼
1. EXTRACT    (pipeline.py)  — read the real CSV, select the 12 best sensors
     ▼
2. VALIDATE   (pipeline.py)  — check real missing values, real duplicate
                                timestamps, real statistical outliers,
                                real freshness gaps (up to 28 days between
                                some production runs); every issue logged
     ▼
3. TRANSFORM  (pipeline.py)  — interpolate real missing values, engineer
                                rolling time-series features (10-run
                                rolling mean, rate of change)
     ▼
4. LOAD       (pipeline.py)  — write into a star-schema SQLite database
                                (output/industrial.db)
     ▼
5. ANALYZE    (visualize_sensor_data.py) — query with SQL, generate charts
```

**Database schema (star schema):**
- `dim_sensor` — which 12 sensors were kept and the exact rule used to
  select them (dimension table)
- `fact_process_readings` — timestamp-indexed real sensor readings, real
  pass/fail outcome, engineered features (fact table)
- `data_quality_log` — every real validation issue caught and how it was
  resolved (nothing fails silently)

## Design decision: ETL vs ELT

This pipeline is **ETL** (transform before load), not ELT. The raw data has
real missing values and irregular timestamps that need resolving before
anything downstream — a dashboard, an ML feature pipeline — reads it.
Loading unvalidated data into a shared table first (the ELT pattern) risks
other consumers reading incomplete data before a transform job gets around
to cleaning it. At larger scale, or with a cloud warehouse (Snowflake /
BigQuery / Databricks), ELT + a tool like dbt is often preferred instead,
since the warehouse has the compute power to transform cheaply at scale.
Which pattern to use is a scale/architecture tradeoff, not a fixed rule.

## Running it

```bash
pip install pandas numpy matplotlib

# raw_data/secom_real.csv must already be present (real data, downloaded once)
python src/pipeline.py               # Steps 1-4: Extract, Validate, Transform, Load
python src/visualize_sensor_data.py  # Step 5: Analyze + generate charts

sqlite3 output/industrial.db < sql/analysis_queries.sql   # explore with SQL
```

If `raw_data/secom_real.csv` is missing, download it from the UCI ML
Repository page for "SECOM" (id 179), or via:
```python
from ucimlrepo import fetch_ucirepo
secom = fetch_ucirepo(id=179)
```

## SQL analysis layer (`sql/analysis_queries.sql`)

Six queries on the real, cleaned manufacturing data:
1. Join to `dim_sensor` — which sensors were kept and why
2. `LAG()` window function — run-over-run sensor change in real
   chronological production order
3. Rolling window function in pure SQL (cross-checks the Python-computed
   rolling feature)
4. CTE + `strftime()` — real monthly failure rate across 2008 production
5. CTE + `RANK()` — ranking real failed runs by sensor extremity
6. Data quality summary — real issues the pipeline caught (missing values,
   duplicate timestamps, a real 28-day gap between production runs)

Query 4 shows genuine month-by-month failure rates directly from the real
2008 production data — not simulated numbers.

## Charts (`output/charts/`, from `visualize_sensor_data.py`)

- `sensor_trend_by_outcome.png` — sensor_01 readings across the real 2008
  timeline, colored by real pass/fail outcome
- `sensor_correlation_with_failure.png` — which of the 12 real sensors
  correlates most strongly with real failure
- `failure_timeline.png` — real monthly failure rate across the actual
  production year
- `data_quality_summary.png` — real data quality issues caught, by category

## Honest limitations

- The sensor columns are anonymized by the original dataset, so physical
  units and plausibility bounds (e.g. "temperature can't exceed X°C") are
  unknown — validation uses statistical (IQR-based) outlier detection
  instead of physical range checks.
- With only 104 real failures out of 1,567 runs, this is a naturally
  imbalanced dataset — worth mentioning if asked about building a
  predictive model on top of this data (a real class-imbalance problem to
  handle, e.g. via resampling or class weighting).

## Possible extensions

- Orchestrate with Airflow (currently a single manual run)
- Swap SQLite for a cloud warehouse (DuckDB → Snowflake/BigQuery migration
  path is very close to a 1:1 swap)
- Build an actual classification model (logistic regression / random
  forest) on top of the engineered features to predict failure —
  natural next step given the real labeled outcome column
- dbt-based transformation layer for an ELT variant of this same pipeline
