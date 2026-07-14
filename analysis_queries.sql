-- ============================================================================
-- analysis_queries.sql
-- Analytical queries over REAL semiconductor manufacturing data
-- (UCI SECOM dataset, 1,567 real production runs, 2008)
-- Tables: fact_process_readings (fact), dim_sensor (dimension)
-- ============================================================================


-- ----------------------------------------------------------------------------
-- Q1. JOIN: which sensors are in this analysis, and why they were selected
--     (dim_sensor documents the real selection rule used in pipeline.py)
-- ----------------------------------------------------------------------------
SELECT sensor_id, selection_rule, note
FROM dim_sensor
ORDER BY sensor_id;


-- ----------------------------------------------------------------------------
-- Q2. WINDOW FUNCTION - LAG: run-over-run change in sensor_01, in real
--     chronological production order
-- ----------------------------------------------------------------------------
SELECT
    timestamp,
    passed,
    sensor_01,
    LAG(sensor_01) OVER (ORDER BY timestamp) AS prev_sensor_01,
    sensor_01 - LAG(sensor_01) OVER (ORDER BY timestamp) AS sensor_01_delta
FROM fact_process_readings
ORDER BY timestamp
LIMIT 20;


-- ----------------------------------------------------------------------------
-- Q3. WINDOW FUNCTION: rolling 10-run average computed in pure SQL
--     (cross-checks the Python-computed rolling feature from the pipeline)
-- ----------------------------------------------------------------------------
SELECT
    timestamp,
    sensor_01,
    AVG(sensor_01) OVER (
        ORDER BY timestamp
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) AS sensor_01_rolling_10_sql
FROM fact_process_readings
ORDER BY timestamp
LIMIT 20;


-- ----------------------------------------------------------------------------
-- Q4. CTE + failure rate by month: real seasonal/temporal pattern in actual
--     production failures across 2008
-- ----------------------------------------------------------------------------
WITH monthly AS (
    SELECT
        strftime('%Y-%m', timestamp) AS month,
        COUNT(*) AS total_runs,
        SUM(CASE WHEN passed = 0 THEN 1 ELSE 0 END) AS real_failures
    FROM fact_process_readings
    GROUP BY month
)
SELECT
    month,
    total_runs,
    real_failures,
    ROUND(100.0 * real_failures / total_runs, 2) AS failure_rate_pct
FROM monthly
ORDER BY month;


-- ----------------------------------------------------------------------------
-- Q5. CTE + RANK: within failed runs only, rank by sensor_01 to see which
--     failures had the most extreme sensor_01 readings
-- ----------------------------------------------------------------------------
WITH failed_runs AS (
    SELECT timestamp, sensor_01, sensor_02, sensor_03
    FROM fact_process_readings
    WHERE passed = 0
)
SELECT
    timestamp,
    sensor_01,
    sensor_02,
    sensor_03,
    RANK() OVER (ORDER BY sensor_01 DESC) AS sensor_01_rank_among_failures
FROM failed_runs
ORDER BY sensor_01_rank_among_failures
LIMIT 15;


-- ----------------------------------------------------------------------------
-- Q6. Data quality summary -- what the pipeline actually caught in real data
--     (real missing values, real duplicate timestamps, real 28-day gap)
-- ----------------------------------------------------------------------------
SELECT * FROM data_quality_log ORDER BY severity, "check";
