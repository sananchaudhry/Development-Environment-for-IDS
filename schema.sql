-- ===========================================
-- Simplified IDS-ML Schema (v3)
-- ===========================================

DROP TABLE IF EXISTS Metrics;
DROP TABLE IF EXISTS Runs;
DROP TABLE IF EXISTS Models;

-- Store available model types
CREATE TABLE IF NOT EXISTS Models (
    model_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name   TEXT NOT NULL UNIQUE,
    description  TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Store each model run
CREATE TABLE IF NOT EXISTS Runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id     INTEGER,
    dataset_name TEXT,
    status       TEXT DEFAULT 'completed',
    runtime_sec  REAL,
    timestamp    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes        TEXT,
    FOREIGN KEY(model_id) REFERENCES Models(model_id)
);

-- Store metrics for each run
CREATE TABLE IF NOT EXISTS Metrics (
    metric_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER,
    metric_name  TEXT,
    metric_value REAL,
    FOREIGN KEY(run_id) REFERENCES Runs(run_id)
);

