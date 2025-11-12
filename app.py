import sqlite3
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

DATABASE = 'ids_ml.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def home():
    return redirect(url_for('run_model'))

@app.route('/run', methods=['GET', 'POST'])
def run_model():
    if request.method == 'POST':
        model = request.form['model']
        dataset = request.form['dataset'] or "CICIDS2017_sample_km.csv"
        params = request.form['parameters']

        print(f"Run button clicked\nModel: {model}\nDataset: {dataset}\nParams: {params}")

        # 1️⃣ Import and run model dynamically (lazy load to avoid slow startup)
        from LCCDE_IDS_GlobeCom22 import run_lccde_model
        results = run_lccde_model(dataset_path=f"./data/{dataset}")

        # 2️⃣ Store run info + metrics in SQL
        conn = get_db_connection()
        cur = conn.cursor()

        # Ensure model exists
        cur.execute("INSERT OR IGNORE INTO Models (model_name, description) VALUES (?, ?);",
                    (model, "Leader Class and Confidence Decision Ensemble IDS"))

        # Insert run record
        cur.execute("""
            INSERT INTO Runs (model_id, dataset_name, status, runtime_sec, notes)
            VALUES ((SELECT model_id FROM Models WHERE model_name=?),
                    ?, 'completed', 0, 'Executed from web interface');
        """, (model, dataset))
        run_id = cur.lastrowid

        # Insert metrics (skip confusion matrix)
        for name, value in results.items():
            if name != 'confusion_matrix':
                cur.execute("INSERT INTO Metrics (run_id, metric_name, metric_value) VALUES (?, ?, ?);",
                            (run_id, name, float(value)))

        conn.commit()
        conn.close()
        print(f"Stored results for run ID: {run_id}")

        return redirect(url_for('history'))

    return render_template('run_model.html')

@app.route('/history')
def history():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT r.run_id AS id,
               m.model_name AS model,
               r.dataset_name AS dataset,
               r.timestamp AS timestamp,
               (SELECT metric_value FROM Metrics WHERE run_id = r.run_id AND metric_name='accuracy') AS accuracy
        FROM Runs r
        JOIN Models m ON r.model_id = m.model_id
        ORDER BY r.timestamp DESC;
    """)
    runs = cursor.fetchall()
    conn.close()
    return render_template('history.html', runs=runs)

@app.route('/compare')
def compare():
    return render_template('compare.html')

if __name__ == '__main__':
    app.run(debug=True)
