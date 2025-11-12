from flask import Flask, render_template, request, redirect, url_for
import sqlite3


DATABASE = 'ids_ml.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # lets you access columns by name
    return conn


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('history.html')

@app.route('/run', methods=['GET', 'POST'])
def run_model():
    if request.method == 'POST':
        # Run the LCCDE model
        from LCCDE_IDS_GlobeCom22 import run_lccde_modelz`
        metrics = run_lccde_model()

        # Connect to database
        conn = get_db_connection()
        cur = conn.cursor()

        # Insert run metadata
        cur.execute("""
            INSERT INTO Runs (model_id, dataset_id, status, runtime_seconds, notes)
            VALUES (1, 1, 'completed', 0, 'Executed LCCDE from Flask');
        """)
        run_id = cur.lastrowid

        # Insert metrics into table
        for name, value in metrics.items():
            if name != 'confusion_matrix':
                cur.execute("""
                    INSERT INTO Metrics (run_id, metric_name, metric_value)
                    VALUES (?, ?, ?);
                """, (run_id, name, value))

        conn.commit()
        conn.close()

        return redirect(url_for('history'))

    return render_template('run_model.html')


@app.route('/history')
def history():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch runs with their model name, dataset, timestamp, and accuracy
    cursor.execute("""
        SELECT 
            r.run_id AS id,
            m.model_name AS model,
            d.dataset_name AS dataset,
            r.run_timestamp AS timestamp,
            (
                SELECT metric_value 
                FROM Metrics 
                WHERE run_id = r.run_id AND metric_name = 'accuracy'
            ) AS accuracy
        FROM Runs r
        JOIN Models m ON r.model_id = m.model_id
        LEFT JOIN Datasets d ON r.dataset_id = d.dataset_id
        ORDER BY r.run_timestamp DESC;
    """)
    runs = cursor.fetchall()
    conn.close()

    return render_template('history.html', runs=runs)


@app.route('/compare')
def compare():
    return render_template('compare.html')

if __name__ == '__main__':
    app.run(debug=True)
