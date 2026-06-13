from flask import Flask, render_template, request, send_file
import joblib
import json
import os
import pandas as pd
import numpy as np
import io

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "marathon_model.pkl")
FEATURES_PATH = os.path.join("model", "feature_columns.json")

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r") as f:
    feature_columns = json.load(f)


# ── helpers ──────────────────────────────────────────────────────────────────

REF_DATA_PATH = os.path.join("model", "reference_data.parquet")
ref_df = pd.read_parquet(REF_DATA_PATH)

# Age group mapping (form values → data values which may differ)
AGE_MAP = {}
for ag in ref_df["age_group"].unique():
    AGE_MAP[ag] = ag
    AGE_MAP[ag.replace(" ", "")] = ag
    AGE_MAP[ag.replace(" - ", "-")] = ag

MIN_COHORT = 50


def find_cohort(target_minutes, gender, age_group):
    """Find similar runners with adaptive window and progressive demographic relaxation."""
    mapped_ag = AGE_MAP.get(age_group, age_group)

    for half_window in [5, 10, 15, 30, 45]:
        lo = target_minutes - half_window
        hi = target_minutes + half_window
        pool = ref_df[(ref_df["duration"] >= lo) & (ref_df["duration"] <= hi)]

        # Try gender + age group
        subset = pool[(pool["gender"] == gender) & (pool["age_group"] == mapped_ag)]
        if len(subset) >= MIN_COHORT:
            return subset, half_window, f"{gender}, {mapped_ag}"

        # Relax to gender only
        subset = pool[pool["gender"] == gender]
        if len(subset) >= MIN_COHORT:
            return subset, half_window, f"{gender}, all ages"

        # Relax to all demographics
        if len(pool) >= MIN_COHORT:
            return pool, half_window, "all runners"

    return ref_df, None, "full dataset (not enough similar runners)"


WEEKLY_FEATURES = [
    "total_weekly_mileage", "total_weekly_duration", "num_runs",
    "longest_run_mileage", "longest_run_duration", "longest_run_pace",
    "fastest_run_mileage", "fastest_run_duration", "fastest_run_pace",
    "pace_diff", "pace_ratio",
]


def cohort_medians(cohort):
    """Compute median of each per-week feature across the cohort."""
    medians = {}
    for feat in WEEKLY_FEATURES:
        for wk in range(1, 17):
            col = f"{feat}_wk{wk}"
            if col in cohort.columns:
                medians[col] = cohort[col].median()
    medians["prev_marathon_time"] = cohort["prev_marathon_time"].median()
    medians["marathon_num"] = cohort["marathon_num"].median()
    return medians


def predict_from_row(row, gender, age_group):
    row["gender"] = gender
    row["age_group"] = age_group
    X = pd.DataFrame([row])[feature_columns]
    for col in feature_columns:
        if col not in ("gender", "age_group"):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return float(model.predict(X)[0])


def generate_schedule(target_minutes, gender, age_group, marathon_num,
                      prev_marathon_time):
    """Generate a 16-week plan from the median training patterns of
    real runners who finished near the target time."""
    cohort, window, cohort_desc = find_cohort(target_minutes, gender, age_group)

    medians = cohort_medians(cohort)
    if prev_marathon_time is not None:
        medians["prev_marathon_time"] = prev_marathon_time
    medians["marathon_num"] = marathon_num

    def fmt_pace(p):
        if pd.isna(p) or p <= 0:
            return "--"
        total_secs = int(round(p * 60))
        mins = total_secs // 60
        secs = total_secs % 60
        return f"{mins}:{secs:02d} /mi"

    # Data uses wk1 = taper (1 week before race), wk16 = start of training.
    # Flip so the display reads Week 1 = start of training → Week 16 = race week.
    weeks = []
    for display_week in range(1, 17):
        data_wk = 17 - display_week  # display 1 → data wk16, display 16 → data wk1

        wm = medians.get(f"total_weekly_mileage_wk{data_wk}", 0)
        lr = medians.get(f"longest_run_mileage_wk{data_wk}", 0)
        lr_pace = medians.get(f"longest_run_pace_wk{data_wk}", 0)
        nr = medians.get(f"num_runs_wk{data_wk}", 4)
        dur = medians.get(f"total_weekly_duration_wk{data_wk}", 0)

        nr_int = int(round(nr))
        remaining = max(wm - lr, 0)
        easy_runs = max(nr_int - 1, 1)
        easy_per_run = remaining / easy_runs
        easy_pace = dur / wm if wm > 0 else 0

        weeks.append({
            "Week": display_week,
            "Phase": _phase(display_week),
            "Total Miles": round(wm, 1),
            "Runs/Week": nr_int,
            "Long Run (mi)": round(lr, 1),
            "Long Pace": fmt_pace(lr_pace),
            "Easy Run (mi)": round(easy_per_run, 1),
            "Easy Pace": fmt_pace(easy_pace),
        })

    modeled_pred = predict_from_row(dict(medians), gender, age_group)
    cohort_info = {
        "count": len(cohort),
        "window": f"±{window} min" if window else "full dataset",
        "desc": cohort_desc,
    }

    return weeks, round(modeled_pred, 1), cohort_info


def _phase(display_week):
    if display_week <= 4:
        return "Base"
    if display_week <= 10:
        return "Build"
    if display_week <= 14:
        return "Peak"
    return "Taper"


def fmt_minutes(mins):
    h = int(mins) // 60
    m = int(mins) % 60
    return f"{h}h {m:02d}m"


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/resume")
def resume():
    return render_template("resume.html")


@app.route("/projects")
def projects():
    return render_template("projects.html")


@app.route("/projects/marathon/template")
def download_template():
    template_df = pd.DataFrame([[""] * len(feature_columns)], columns=feature_columns)
    buffer = io.StringIO()
    template_df.to_csv(buffer, index=False)
    output = io.BytesIO()
    output.write(buffer.getvalue().encode("utf-8"))
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True,
                     download_name="marathon_input_template.csv")


@app.route("/projects/marathon", methods=["GET", "POST"])
def marathon_project():
    prediction_table = None
    error = None

    if request.method == "POST":
        try:
            if "file" not in request.files:
                raise ValueError("No file uploaded.")
            file = request.files["file"]
            if file.filename == "":
                raise ValueError("Please choose a CSV file.")
            if not file.filename.lower().endswith(".csv"):
                raise ValueError("Please upload a CSV file.")

            uploaded_df = pd.read_csv(file)
            missing_cols = [c for c in feature_columns if c not in uploaded_df.columns]
            if missing_cols:
                raise ValueError("Missing columns: " + ", ".join(missing_cols))

            X_input = uploaded_df[feature_columns].copy()
            for col in feature_columns:
                if col not in ("gender", "age_group"):
                    X_input[col] = pd.to_numeric(X_input[col], errors="coerce")

            preds = model.predict(X_input)
            result_df = uploaded_df.copy()
            result_df["predicted_finish_time_minutes"] = np.round(preds, 2)
            prediction_table = result_df.to_html(index=False, classes="results-table", border=0)

        except Exception as e:
            error = str(e)

    return render_template("marathon_project.html",
                           prediction_table=prediction_table, error=error)


@app.route("/projects/marathon/schedule", methods=["GET", "POST"])
def marathon_schedule():
    schedule = None
    modeled_time = None
    error = None
    form_values = {}

    if request.method == "POST":
        try:
            form_values = request.form.to_dict()

            # Parse target time
            target_raw = request.form.get("target_time", "").strip()
            if ":" in target_raw:
                parts = target_raw.split(":")
                if len(parts) == 2:
                    target_minutes = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:
                    target_minutes = int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
                else:
                    raise ValueError("Enter target time as H:MM or HH:MM:SS")
            else:
                target_minutes = float(target_raw)

            if target_minutes < 120 or target_minutes > 420:
                raise ValueError("Target time must be between 2:00 and 7:00.")

            gender = request.form.get("gender", "M")
            age_group = request.form.get("age_group", "18-39")
            marathon_num = int(request.form.get("marathon_num", 1))

            prev_raw = request.form.get("prev_marathon_time", "").strip()
            prev_marathon_time = None
            if prev_raw:
                if ":" in prev_raw:
                    parts = prev_raw.split(":")
                    if len(parts) == 2:
                        prev_marathon_time = int(parts[0]) * 60 + int(parts[1])
                    elif len(parts) == 3:
                        prev_marathon_time = int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
                else:
                    prev_marathon_time = float(prev_raw)

            schedule, modeled_pred_raw, cohort_info = generate_schedule(
                target_minutes, gender, age_group,
                marathon_num, prev_marathon_time
            )
            modeled_time = fmt_minutes(modeled_pred_raw)

        except Exception as e:
            error = str(e)

    return render_template("marathon_schedule.html",
                           schedule=schedule, modeled_time=modeled_time,
                           error=error, form_values=form_values,
                           cohort_info=cohort_info if schedule else None)


if __name__ == "__main__":
    app.run(debug=True)
