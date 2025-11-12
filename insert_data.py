import sqlite3
from datetime import datetime
from LCCDE_IDS_GlobeCom22 import run_lccde_model

DB = "ids_ml.db"

def get_conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize schema if not exists
with open("schema.sql") as f:
    conn = get_conn()
    conn.executescript(f.read())
    conn.commit()
    conn.close()
    print("Database schema initialized.")

# Ensure LCCDE model exists
conn = get_conn()
cur = conn.cursor()
cur.execute("INSERT OR IGNORE INTO Models (model_name, description) VALUES (?, ?);",
            ("LCCDE", "Leader Class and Confidence Decision Ensemble IDS"))
conn.commit()

# Run model on dataset
print("Running LCCDE model...")
metrics = run_lccde_model(dataset_path="/data/CICIDS2017_sample_km.csv")

# Insert a run entry
cur.execute("""
    INSERT INTO Runs (model_id, dataset_name, status, runtime_sec, notes)
    VALUES ((SELECT model_id FROM Models WHERE model_name='LCCDE'),
            ?, 'completed', 0, 'Run from insert_data.py');
""", ("CICIDS2017_sample_km.csv",))
run_id = cur.lastrowid

# Insert metrics
for name, value in metrics.items():
    if name != 'confusion_matrix':
        cur.execute("INSERT INTO Metrics (run_id, metric_name, metric_value) VALUES (?, ?, ?);",
                    (run_id, name, value))

conn.commit()
conn.close()

print("Data inserted for run ID:", run_id)
