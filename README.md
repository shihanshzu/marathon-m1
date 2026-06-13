# Marathon ML

A machine learning project analyzing 22,000+ marathon training cycles to predict finish times and generate data-driven training plans.

## Features

### Finish Time Predictor
Upload 16 weeks of training data (CSV) and get a finish time prediction from a Ridge Regression model trained on real marathon runners.

### Training Schedule Generator
Enter a goal finish time, gender, and age group. The app finds real runners who finished near your target and builds a 16-week plan from the **median training patterns** of that cohort — not from hardcoded rules or model inversion.

- Adaptive cohort windowing (±5 → ±10 → ±15 → ±30 → ±45 min) with progressive demographic relaxation
- Minimum cohort size of 50 runners ensures statistical reliability
- Displays long run, easy run distances and paces per week across Base → Build → Peak → Taper phases

## Key Findings

1. **Pace differentiation matters.** The gap between easy-run pace and fast-run pace is one of the strongest independent predictors — runners who keep slow runs slow and fast runs fast finish faster, even at the same weekly mileage.

2. **Baseline fitness outweighs peak-week volume.** Early-training paces (weeks 1–4) carry more predictive weight than peak-week mileage. The fitness you bring into the training block matters more than how hard you push during it.

## Model

| Detail | Value |
|---|---|
| Algorithm | Ridge Regression (α = 51.79) |
| Features | 157 (per-week mileage, pace, long-run & fastest-run metrics × 16 weeks + demographics) |
| CV | GroupKFold on athlete ID (no leakage between a runner's training and race result) |
| RMSE | ~29 min |
| R² | 0.614 |

Ridge was chosen over Random Forest (R²=0.68) and XGBoost (R²=0.69) for interpretability and stable generalization.

## Data

Garmin/Strava training logs (2019–2020). Raw data is in kilometers; the pipeline converts to miles. Reference dataset: 22,785 runners (17,698 M / 5,087 F), finish times 131–360 min.

## Running Locally

```bash
pip install flask scikit-learn pandas numpy joblib pyarrow
python app.py
```
Then open [http://localhost:5000](http://localhost:5000).

**Note:** The schedule generator requires `model/reference_data.parquet` (excluded from git due to size). Run `build_reference_data.py` to generate it from raw training data.

## Project Structure

```
├── app.py                      # Flask app (predictor + schedule generator)
├── build_reference_data.py     # Builds reference_data.parquet from raw data
├── Capstone.ipynb              # Model training & EDA notebook
├── model/
│   ├── marathon_model.pkl      # Trained Ridge pipeline
│   ├── feature_columns.json    # Expected feature column names
│   └── reference_data.parquet  # (generated, not in git)
├── templates/
│   ├── base.html
│   ├── home.html               # Insight-first landing page
│   ├── marathon_project.html   # CSV predictor
│   └── marathon_schedule.html  # Schedule generator
└── static/
    └── style.css
```

## Tech

Python, scikit-learn, pandas, Flask, Jinja2.
