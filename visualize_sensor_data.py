"""
visualize_sensor_data.py

STEP 5: Analysis & Visualization -- on the SAME real SECOM data that went
through steps 1-4 in pipeline.py. Produces charts answering real questions:
    1. Does sensor_01 (highest-variance real sensor) trend differently
       before failed vs passed runs?
    2. Which of the 12 selected sensors correlates most with real failure?
    3. Real process timeline: failures over the actual 2008 production year
    4. Data quality summary: what the pipeline actually caught in real data
"""

import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "output" / "industrial.db"
CHART_DIR = BASE_DIR / "output" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def load_data():
    conn = sqlite3.connect(DB_PATH)
    readings = pd.read_sql("SELECT * FROM fact_process_readings", conn, parse_dates=["timestamp"])
    quality_log = pd.read_sql("SELECT * FROM data_quality_log", conn)
    conn.close()
    return readings, quality_log


def chart_sensor_trend_by_outcome(readings: pd.DataFrame):
    plt.figure()
    for outcome, grp in readings.groupby("passed"):
        label = "Passed" if outcome == 1 else "Failed"
        grp_sorted = grp.sort_values("timestamp")
        plt.scatter(grp_sorted["timestamp"], grp_sorted["sensor_01"],
                    alpha=0.5, s=12, label=label)
    plt.xlabel("Date (real production timeline, 2008)")
    plt.ylabel("sensor_01 reading (highest-variance real sensor)")
    plt.title("Sensor 01 Readings Over Real Production Timeline, by Outcome")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "sensor_trend_by_outcome.png", dpi=120)
    plt.close()
    print("Saved: sensor_trend_by_outcome.png")


def chart_sensor_correlation_with_failure(readings: pd.DataFrame):
    sensor_cols = [c for c in readings.columns if c.startswith("sensor_") and "_roll_" not in c and "_rate_" not in c]
    corr = readings[sensor_cols + ["passed"]].corr()["passed"].drop("passed").sort_values()

    plt.figure()
    colors = ["#c0392b" if abs(v) == abs(corr).max() else "#2980b9" for v in corr.values]
    corr.plot(kind="barh", color=colors)
    plt.xlabel("Correlation with passing (negative = associated with real failure)")
    plt.title("Which Real Sensors Correlate Most With Failure")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "sensor_correlation_with_failure.png", dpi=120)
    plt.close()
    print("Saved: sensor_correlation_with_failure.png")


def chart_failure_timeline(readings: pd.DataFrame):
    monthly = readings.set_index("timestamp").resample("ME")["passed"].agg(["count", "sum"])
    monthly["failure_rate_pct"] = 100 * (1 - monthly["sum"] / monthly["count"])

    plt.figure()
    monthly["failure_rate_pct"].plot(kind="bar", color="#c0392b")
    plt.ylabel("Real failure rate (%)")
    plt.xlabel("Month (2008, real production data)")
    plt.title("Real Monthly Failure Rate — Actual Wafer Fabrication Line")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "failure_timeline.png", dpi=120)
    plt.close()
    print("Saved: failure_timeline.png")


def chart_data_quality_summary(quality_log: pd.DataFrame):
    plt.figure()
    quality_log["category"] = quality_log["check"].str.extract(r"^([a-z]+)")
    by_category = quality_log.groupby("category")["count"].sum().sort_values()
    by_category.plot(kind="barh", color="#f39c12")
    plt.xlabel("Number of real records affected")
    plt.title("Real Data Quality Issues Caught by the Pipeline")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "data_quality_summary.png", dpi=120)
    plt.close()
    print("Saved: data_quality_summary.png")


def run():
    readings, quality_log = load_data()
    print(f"Loaded {len(readings)} real cleaned process records "
          f"({readings['passed'].eq(0).sum()} real failures)\n")

    chart_sensor_trend_by_outcome(readings)
    chart_sensor_correlation_with_failure(readings)
    chart_failure_timeline(readings)
    chart_data_quality_summary(quality_log)

    print(f"\nAll 4 charts saved to {CHART_DIR}")


if __name__ == "__main__":
    run()
