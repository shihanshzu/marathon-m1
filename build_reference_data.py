"""
Replicate the notebook's feature engineering pipeline from raw parquet data,
then save the final processed DataFrame as model/reference_data.parquet.

Run once:  python build_reference_data.py
"""
import pandas as pd
import numpy as np
import os

DATA_FILES = ["run_ww_2019_d.parquet", "run_ww_2020_d.parquet"]
OUTPUT_PATH = os.path.join("model", "reference_data.parquet")


def main():
    # ── Load raw data ────────────────────────────────────────────────────
    frames = [pd.read_parquet(f) for f in DATA_FILES if os.path.exists(f)]
    if not frames:
        raise FileNotFoundError("No parquet data files found")
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["distance"] = df["distance"] * 0.621371  # km → miles
    df["pace"] = df["duration"] / df["distance"]  # min/mile
    print(f"Loaded {len(df):,} raw runs")

    # ── Marathon race records (any athlete who ran 26.2–27 mi) ──────────
    df_races = df[
        (df["distance"] >= 26.2) & (df["distance"] < 27)
    ].copy()
    df_races = df_races.sort_values(["athlete", "datetime"])
    df_races["marathon_num"] = df_races.groupby("athlete").cumcount() + 1
    df_races["prev_marathon_time"] = df_races.groupby("athlete")["duration"].shift(1)
    print(f"Marathon race records: {len(df_races):,}")

    # ── 16-week training availability (checked on ALL athletes' runs) ───
    df_active = df[df["distance"] > 0].copy()
    df_active["week"] = df_active["datetime"].dt.to_period("W")

    weekly_check = (
        df_active.groupby(["athlete", "week"])["distance"].sum().reset_index()
    )
    athlete_week_sets = weekly_check.groupby("athlete")["week"].apply(set).to_dict()

    def has_16_weeks(row):
        aws = athlete_week_sets.get(row["athlete"], set())
        rw = row["datetime"].to_period("W")
        return all((rw - i) in aws for i in range(1, 17))

    df_races["has_16wk"] = df_races.apply(has_16_weeks, axis=1)
    print(f"Races with 16 weeks training: {df_races['has_16wk'].sum():,}")

    # Gender-specific time filters AFTER 16-week check (matches notebook order)
    valid_races = df_races[
        df_races["has_16wk"]
        & (df_races["duration"] <= 360)
        & (
            ((df_races["gender"] == "M") & (df_races["duration"] >= 130))
            | ((df_races["gender"] == "F") & (df_races["duration"] >= 140))
        )
    ].copy()
    valid_athletes = valid_races["athlete"].unique()
    print(f"After time/gender filter: {len(valid_races):,}")

    # ── Build race metadata ──────────────────────────────────────────────
    race_meta = valid_races.copy()
    race_meta["race_date"] = race_meta["datetime"]
    race_meta["race_id"] = (
        race_meta["athlete"].astype(str) + "_" + race_meta["race_date"].astype(str)
    )
    race_meta = race_meta[
        [
            "race_id", "athlete", "race_date",
            "gender", "age_group",
            "distance", "duration", "pace",
            "marathon_num", "prev_marathon_time",
        ]
    ].copy()

    # ── Weekly training features ─────────────────────────────────────────
    df_active = df[(df["distance"] > 0) & (df["athlete"].isin(valid_athletes))].copy()
    df_active["datetime"] = pd.to_datetime(df_active["datetime"])
    df_active["pace"] = df_active["duration"] / df_active["distance"]
    df_active["week"] = df_active["datetime"].dt.to_period("W")

    weekly_base = df_active.groupby(["athlete", "week"], as_index=False).agg(
        total_weekly_mileage=("distance", "sum"),
        total_weekly_duration=("duration", "sum"),
        num_runs=("distance", "count"),
    )

    longest_idx = df_active.groupby(["athlete", "week"])["distance"].idxmax()
    longest = df_active.loc[
        longest_idx, ["athlete", "week", "distance", "duration", "pace"]
    ].rename(
        columns={
            "distance": "longest_run_mileage",
            "duration": "longest_run_duration",
            "pace": "longest_run_pace",
        }
    )

    fastest_df = df_active[df_active["pace"].notna()].copy()
    fastest_idx = fastest_df.groupby(["athlete", "week"])["pace"].idxmin()
    fastest = fastest_df.loc[
        fastest_idx, ["athlete", "week", "distance", "duration", "pace"]
    ].rename(
        columns={
            "distance": "fastest_run_mileage",
            "duration": "fastest_run_duration",
            "pace": "fastest_run_pace",
        }
    )

    weekly = (
        weekly_base.merge(longest, on=["athlete", "week"], how="left")
        .merge(fastest, on=["athlete", "week"], how="left")
    )
    weekly["pace_diff"] = weekly["longest_run_pace"] - weekly["fastest_run_pace"]
    weekly["pace_ratio"] = np.where(
        weekly["fastest_run_pace"] > 0,
        weekly["longest_run_pace"] / weekly["fastest_run_pace"],
        np.nan,
    )

    # ── Map each race to its 16 training weeks ──────────────────────────
    race_weeks = race_meta[["race_id", "athlete", "race_date"]].copy()
    race_weeks["race_week"] = race_weeks["race_date"].dt.to_period("W")
    race_weeks = race_weeks.loc[race_weeks.index.repeat(16)].reset_index(drop=True)
    race_weeks["week_num"] = race_weeks.groupby("race_id").cumcount() + 1
    race_weeks["train_week"] = race_weeks["race_week"] - race_weeks["week_num"]

    race_week_features = race_weeks.merge(
        weekly,
        left_on=["athlete", "train_week"],
        right_on=["athlete", "week"],
        how="left",
    )
    race_week_features = race_week_features.drop(
        columns=["week", "race_week", "train_week"]
    )

    # ── Pivot to wide format ─────────────────────────────────────────────
    feature_cols = [
        "total_weekly_mileage", "total_weekly_duration", "num_runs",
        "longest_run_mileage", "longest_run_duration", "longest_run_pace",
        "fastest_run_mileage", "fastest_run_duration", "fastest_run_pace",
        "pace_diff", "pace_ratio",
    ]
    wide = race_week_features.pivot(
        index="race_id", columns="week_num", values=feature_cols
    )
    wide.columns = [f"{feat}_wk{wk}" for feat, wk in wide.columns]
    wide = wide.reset_index()

    # ── Merge metadata ───────────────────────────────────────────────────
    final_df = wide.merge(
        race_meta[
            [
                "race_id", "gender", "age_group",
                "duration", "marathon_num", "prev_marathon_time",
            ]
        ],
        on="race_id",
        how="left",
    )

    # Pace consistency filter
    pace_cols = [c for c in final_df.columns if "longest_run_pace_wk" in c]
    final_df["pace_std"] = final_df[pace_cols].std(axis=1)
    final_df = final_df[final_df["pace_std"] <= 15].copy()
    final_df = final_df.drop(columns=["pace_std", "race_id"])

    print(f"Final reference dataset: {final_df.shape}")
    print(f"Gender distribution:\n{final_df['gender'].value_counts()}")
    print(f"\nAge group distribution:\n{final_df['age_group'].value_counts()}")
    print(f"\nFinish time range: {final_df['duration'].min():.0f} - {final_df['duration'].max():.0f} min")

    final_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
