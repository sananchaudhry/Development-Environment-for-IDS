import json
import os
import sqlite3
from pathlib import Path
from difflib import get_close_matches

from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'ids-ml-secret-key'

DATABASE = 'ids_ml.db'
FIGURE_DIR = Path("static/figures")

# Canonical parameter schema for the LCCDE model
LCCDE_PARAM_SCHEMA = {
    "train_size": {"type": float, "default": 0.8, "min": 0.1, "max": 0.95},
    "random_state": {"type": int, "default": 0},
    "smote_minority_2": {"type": int, "default": 1000, "min": 1},
    "smote_minority_4": {"type": int, "default": 1000, "min": 1},
    "lgb_num_leaves": {"type": int, "default": 31, "min": 2},
    "lgb_learning_rate": {"type": float, "default": 0.1, "min": 0.0001, "max": 1.0},
    "lgb_n_estimators": {"type": int, "default": 100, "min": 10},
    "xgb_n_estimators": {"type": int, "default": 100, "min": 10},
    "xgb_max_depth": {"type": int, "default": 6, "min": 1},
    "xgb_learning_rate": {"type": float, "default": 0.1, "min": 0.0001, "max": 1.0},
    "cb_depth": {"type": int, "default": 6, "min": 1},
    "cb_iterations": {"type": int, "default": 200, "min": 10},
    "cb_learning_rate": {"type": float, "default": 0.1, "min": 0.0001, "max": 1.0},
}

LCCDE_PARAM_ALIASES = {
    "train_split": "train_size",
    "smote_2": "smote_minority_2",
    "smote_4": "smote_minority_4",
    "num_leaves": "lgb_num_leaves",
    "learning_rate": "lgb_learning_rate",
    "lgb_lr": "lgb_learning_rate",
    "xgb_depth": "xgb_max_depth",
    "cb_lr": "cb_learning_rate",
    "cb_iter": "cb_iterations",
}

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_extended_schema(conn):
    """Create auxiliary tables for parameters and artifacts if they do not exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS RunParameters (
            param_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            param_name TEXT,
            param_value TEXT,
            FOREIGN KEY(run_id) REFERENCES Runs(run_id)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS RunArtifacts (
            artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            artifact_type TEXT,
            label TEXT,
            path TEXT,
            FOREIGN KEY(run_id) REFERENCES Runs(run_id)
        );
        """
    )


def canonical_lccde_defaults():
    return {k: v["default"] for k, v in LCCDE_PARAM_SCHEMA.items()}


def normalize_key(key: str) -> str:
    key_lower = key.strip().lower()
    if key_lower in LCCDE_PARAM_ALIASES:
        return LCCDE_PARAM_ALIASES[key_lower]
    return key_lower


def parse_parameter_block(raw_params):
    """Parse parameters from either a mapping (preferred) or a raw string."""
    if isinstance(raw_params, dict):
        return {k: v for k, v in raw_params.items() if v not in (None, "")}

    raw_params = (raw_params or "").strip()
    if not raw_params:
        return {}
    # Try JSON first
    try:
        obj = json.loads(raw_params)
        if isinstance(obj, dict):
            return {str(k): obj[k] for k in obj}
    except json.JSONDecodeError:
        pass

    parsed = {}
    # Support key=value separated by commas or newlines
    for chunk in raw_params.replace("\r", "").replace(",", "\n").split("\n"):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def validate_lccde_params(raw_params, base_params=None):
    """Validate user-supplied params and detect typos/incomplete entries."""
    parsed = parse_parameter_block(raw_params)
    errors, warnings = [], []
    normalized = dict(base_params) if base_params else canonical_lccde_defaults()

    for key, value in parsed.items():
        canonical_key = normalize_key(key)
        if canonical_key not in LCCDE_PARAM_SCHEMA:
            suggestion = get_close_matches(canonical_key, LCCDE_PARAM_SCHEMA.keys(), n=1)
            if suggestion:
                errors.append(f"Unknown parameter '{key}'. Did you mean '{suggestion[0]}'?")
            else:
                errors.append(f"Unknown parameter '{key}' was ignored.")
            continue

        schema = LCCDE_PARAM_SCHEMA[canonical_key]
        target_type = schema["type"]
        try:
            cast_value = target_type(value)
        except (TypeError, ValueError):
            errors.append(f"Parameter '{key}' expects {target_type.__name__} but received '{value}'.")
            continue

        if "min" in schema and cast_value < schema["min"]:
            errors.append(f"Parameter '{canonical_key}' must be >= {schema['min']}.")
            continue
        if "max" in schema and cast_value > schema["max"]:
            errors.append(f"Parameter '{canonical_key}' must be <= {schema['max']}.")
            continue

        normalized[canonical_key] = cast_value

    # Ensure train_size sanity
    if normalized["train_size"] <= 0 or normalized["train_size"] >= 1:
        errors.append("'train_size' must be between 0 and 1 (exclusive).")

    return normalized, errors, warnings

@app.route('/')
def home():
    return redirect(url_for('run_model'))


def store_run_results(model_name, dataset, metrics, params, artifacts, notes="Executed from web interface"):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_db_connection()
    ensure_extended_schema(conn)
    cur = conn.cursor()

    cur.execute("INSERT OR IGNORE INTO Models (model_name, description) VALUES (?, ?);",
                (model_name, "Leader Class and Confidence Decision Ensemble IDS"))

    cur.execute(
        """
        INSERT INTO Runs (model_id, dataset_name, status, runtime_sec, notes)
        VALUES ((SELECT model_id FROM Models WHERE model_name=?), ?, 'completed', 0, ?);
        """,
        (model_name, dataset, notes),
    )
    run_id = cur.lastrowid

    for name, value in metrics.items():
        cur.execute(
            "INSERT INTO Metrics (run_id, metric_name, metric_value) VALUES (?, ?, ?);",
            (run_id, name, float(value)),
        )

    for name, value in params.items():
        cur.execute(
            "INSERT INTO RunParameters (run_id, param_name, param_value) VALUES (?, ?, ?);",
            (run_id, name, json.dumps(value)),
        )

    for artifact in artifacts:
        cur.execute(
            """
            INSERT INTO RunArtifacts (run_id, artifact_type, label, path)
            VALUES (?, ?, ?, ?);
            """,
            (run_id, artifact.get("type"), artifact.get("label"), artifact.get("path")),
        )

    conn.commit()
    conn.close()
    return run_id


def fetch_run_details(run_id):
    conn = get_db_connection()
    ensure_extended_schema(conn)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT r.run_id AS id,
               m.model_name AS model,
               r.dataset_name AS dataset,
               r.timestamp AS timestamp,
               r.status AS status,
               r.notes AS notes
        FROM Runs r
        JOIN Models m ON r.model_id = m.model_id
        WHERE r.run_id = ?;
        """,
        (run_id,),
    )
    run = cursor.fetchone()
    if not run:
        conn.close()
        return None

    cursor.execute(
        "SELECT metric_name, metric_value FROM Metrics WHERE run_id = ?;", (run_id,)
    )
    metrics = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.execute(
        "SELECT param_name, param_value FROM RunParameters WHERE run_id = ?;", (run_id,)
    )
    params = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

    cursor.execute(
        "SELECT artifact_type, label, path FROM RunArtifacts WHERE run_id = ?;", (run_id,)
    )
    artifacts = [
        {"type": row[0], "label": row[1], "path": row[2]}
        for row in cursor.fetchall()
    ]

    conn.close()
    return {
        "run": run,
        "metrics": metrics,
        "params": params,
        "artifacts": artifacts,
    }


def fetch_run_summaries():
    conn = get_db_connection()
    ensure_extended_schema(conn)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT r.run_id AS id,
               m.model_name AS model,
               r.dataset_name AS dataset,
               r.timestamp AS timestamp,
               (SELECT metric_value FROM Metrics WHERE run_id = r.run_id AND metric_name='accuracy') AS accuracy
        FROM Runs r
        JOIN Models m ON r.model_id = m.model_id
        ORDER BY r.timestamp DESC;
        """
    )
    runs = cursor.fetchall()
    conn.close()
    return runs

@app.route('/run', methods=['GET', 'POST'])
def run_model():
    selected_model = request.form.get('model', 'LCCDE') if request.method == 'POST' else 'LCCDE'
    dataset_value = request.form.get('dataset', '') if request.method == 'POST' else ''
    param_entries = (
        {key: request.form.get(key, '') for key in LCCDE_PARAM_SCHEMA}
        if request.method == 'POST'
        else canonical_lccde_defaults()
    )
    errors = []
    warnings = []

    if request.method == 'POST':
        dataset = dataset_value or "CICIDS2017_sample_km.csv"
        parsed_params, errors, warnings = validate_lccde_params(param_entries)

        if selected_model != 'LCCDE':
            errors.append("Only the LCCDE model is currently supported in this interface.")

        dataset_path = f"./data/{dataset}"
        if not os.path.exists(dataset_path):
            errors.append(f"Dataset '{dataset}' was not found in the data/ directory.")

        if errors:
            return render_template(
                'run_model.html',
                errors=errors,
                warnings=warnings,
                selected_model=selected_model,
                dataset_value=dataset_value,
                param_values=param_entries,
                defaults=canonical_lccde_defaults(),
                param_schema=LCCDE_PARAM_SCHEMA,
            )

        from LCCDE_IDS_GlobeCom22 import run_lccde_model

        results = run_lccde_model(dataset_path=dataset_path, params=parsed_params, artifact_dir=FIGURE_DIR)
        metrics = results.get('metrics', {})
        artifacts = results.get('artifacts', [])

        run_id = store_run_results(selected_model, dataset, metrics, parsed_params, artifacts)
        print(f"Stored results for run ID: {run_id}")

        flash(f"Run {run_id} completed successfully.", "success")
        return redirect(url_for('history', run_id=run_id))

    return render_template(
        'run_model.html',
        errors=errors,
        warnings=warnings,
        selected_model=selected_model,
        dataset_value=dataset_value,
        param_values=param_entries or canonical_lccde_defaults(),
        defaults=canonical_lccde_defaults(),
        param_schema=LCCDE_PARAM_SCHEMA,
    )

@app.route('/history')
def history():
    run_id = request.args.get('run_id', type=int)
    runs = fetch_run_summaries()
    selected_details = fetch_run_details(run_id) if run_id else None

    return render_template('history.html', runs=runs, selected_details=selected_details)


@app.route('/view/<int:run_id>')
def view_run(run_id):
    return redirect(url_for('history', run_id=run_id))


@app.route('/delete/<int:run_id>')
def delete_run(run_id):
    conn = get_db_connection()
    ensure_extended_schema(conn)
    cur = conn.cursor()
    cur.execute("DELETE FROM Metrics WHERE run_id=?", (run_id,))
    cur.execute("DELETE FROM RunParameters WHERE run_id=?", (run_id,))
    cur.execute("DELETE FROM RunArtifacts WHERE run_id=?", (run_id,))
    cur.execute("DELETE FROM Runs WHERE run_id=?", (run_id,))
    conn.commit()
    conn.close()
    flash(f"Run {run_id} deleted.", "info")
    return redirect(url_for('history'))

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    runs = fetch_run_summaries()
    baseline_id = request.args.get('baseline_id', type=int)
    comparison = None
    baseline_details = fetch_run_details(baseline_id) if baseline_id else None
    errors = []
    warnings = []
    dataset_value = ''
    form_params = baseline_details["params"] if baseline_details and baseline_details.get("params") else canonical_lccde_defaults()

    if request.method == 'POST':
        baseline_id = request.form.get('baseline_id', type=int)
        baseline_details = fetch_run_details(baseline_id) if baseline_id else None
        dataset_value = request.form.get('dataset', '')
        form_params = {key: request.form.get(key, '') for key in LCCDE_PARAM_SCHEMA}

        if not baseline_details:
            errors.append("A valid baseline run must be selected for comparison.")
        else:
            base_params = baseline_details["params"] if baseline_details and baseline_details.get("params") else canonical_lccde_defaults()
            merged_params, errors, warnings = validate_lccde_params(form_params, base_params=base_params)

            dataset = dataset_value or baseline_details["run"]["dataset"]
            dataset_path = f"./data/{dataset}"
            if not os.path.exists(dataset_path):
                errors.append(f"Dataset '{dataset}' was not found in the data/ directory.")

            if not errors:
                from LCCDE_IDS_GlobeCom22 import run_lccde_model

                results = run_lccde_model(dataset_path=dataset_path, params=merged_params, artifact_dir=FIGURE_DIR)
                metrics = results.get('metrics', {})
                artifacts = results.get('artifacts', [])
                new_run_id = store_run_results('LCCDE', dataset, metrics, merged_params, artifacts, notes=f"Comparison rerun of {baseline_id}")
                comparison = {
                    'baseline': baseline_details,
                    'new': fetch_run_details(new_run_id)
                }

                flash(f"Comparison run {new_run_id} completed.", "success")

    return render_template(
        'compare.html',
        runs=runs,
        baseline=baseline_details,
        comparison=comparison,
        errors=errors,
        warnings=warnings,
        form_params=form_params,
        dataset_value=dataset_value,
        defaults=canonical_lccde_defaults(),
        param_schema=LCCDE_PARAM_SCHEMA,
    )

if __name__ == '__main__':
    app.run(debug=True)
