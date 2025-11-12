import sqlite3

conn = sqlite3.connect("ids_ml.db")
cursor = conn.cursor()

# Add model
cursor.execute("""
INSERT INTO Models (model_name, description, algorithm_type, version)
VALUES ('LCCDE', 'Lightweight Cooperative Coevolution Differential Evolution IDS', 'Evolutionary', '1.0');
""")

# Add dataset
cursor.execute("""
INSERT INTO Datasets (dataset_name, source_url, num_records, features_count, preprocessing)
VALUES ('NSL-KDD', 'https://www.unb.ca/cic/datasets/nsl.html', 125973, 41, 'Normalized + Feature selection');
""")

# Add run
cursor.execute("""
INSERT INTO Runs (user_id, model_id, dataset_id, status, runtime_seconds, notes)
VALUES (NULL, 1, 1, 'completed', 4.82, 'Run triggered from UI, baseline LCCDE');
""")

# Insert parameters (user input)
params = {
    'population_size': '50',
    'generations': '100',
    'mutation_factor': '0.8',
    'crossover_rate': '0.9'
}
for name, val in params.items():
    cursor.execute("""
    INSERT INTO Parameters (run_id, param_name, param_value, data_type)
    VALUES (1, ?, ?, 'float');
    """, (name, val))

# Insert metrics (results)
metrics = {
    'accuracy': 0.974,
    'precision': 0.977,
    'recall': 0.965,
    'f1_score': 0.971,
    'roc_auc': 0.982
}
for name, val in metrics.items():
    cursor.execute("""
    INSERT INTO Metrics (run_id, metric_name, metric_value)
    VALUES (1, ?, ?);
    """, (name, val))

# Insert confusion matrix
cursor.execute("""
INSERT INTO ConfusionMatrix (run_id, true_positive, false_positive, true_negative, false_negative)
VALUES (1, 5123, 132, 4981, 164);
""")

conn.commit()
conn.close()
print("All IDS-ML experiment data stored successfully.")
