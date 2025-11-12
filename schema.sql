-- ===========================================
-- IDS-ML Research Platform Database Schema (v2)
-- ===========================================

-- ====== CORE TABLES ======
CREATE TABLE IF NOT EXISTS Users (
    user_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT NOT NULL UNIQUE,
    email           TEXT,
    organization    TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS Models (
    model_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT NOT NULL UNIQUE,
    description     TEXT,
    algorithm_type  TEXT,
    version         TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS Datasets (
    dataset_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name    TEXT NOT NULL,
    source_url      TEXT,
    num_records     INTEGER,
    features_count  INTEGER,
    preprocessing   TEXT,          -- e.g., normalization, feature selection
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS Runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER REFERENCES Users(user_id) ON DELETE SET NULL,
    model_id        INTEGER REFERENCES Models(model_id) ON DELETE CASCADE,
    dataset_id      INTEGER REFERENCES Datasets(dataset_id) ON DELETE SET NULL,
    run_timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status          TEXT DEFAULT 'pending',   -- pending, running, completed, failed
    runtime_seconds REAL,
    notes           TEXT
);

-- ====== PARAMETERS ======
-- Stores user-input hyperparameters for each run
CREATE TABLE IF NOT EXISTS Parameters (
    param_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES Runs(run_id) ON DELETE CASCADE,
    param_name      TEXT NOT NULL,
    param_value     TEXT,
    data_type       TEXT,
    source          TEXT DEFAULT 'user',  -- user, system, default
    UNIQUE(run_id, param_name)
);

-- ====== METRICS ======
-- Accuracy, Precision, Recall, F1, AUC, etc.
CREATE TABLE IF NOT EXISTS Metrics (
    metric_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES Runs(run_id) ON DELETE CASCADE,
    metric_name     TEXT NOT NULL,
    metric_value    REAL,
    UNIQUE(run_id, metric_name)
);

-- ====== CONFUSION MATRIX ======
CREATE TABLE IF NOT EXISTS ConfusionMatrix (
    cm_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES Runs(run_id) ON DELETE CASCADE,
    true_positive   INTEGER,
    false_positive  INTEGER,
    true_negative   INTEGER,
    false_negative  INTEGER
);

-- ====== FEATURE IMPORTANCE ======
CREATE TABLE IF NOT EXISTS FeatureImportances (
    importance_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES Runs(run_id) ON DELETE CASCADE,
    feature_name    TEXT,
    importance_value REAL
);

-- ====== ARTIFACTS ======
-- Plots, logs, model files, etc.
CREATE TABLE IF NOT EXISTS Artifacts (
    artifact_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES Runs(run_id) ON DELETE CASCADE,
    artifact_type   TEXT,             -- e.g., plot, log, model, report
    file_path       TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
