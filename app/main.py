import os
import glob
import json
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import dump

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTROL_PREFIX = "SDLC"
CONTROL_COLS = [f"{CONTROL_PREFIX}{i}" for i in range(1, 25)]


def load_monthly_files(pattern: str = "SDLC_*.json") -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {DATA_DIR}")
    frames = []
    for fp in files:
        with open(fp, "r") as f:
            frames.append(pd.DataFrame(json.load(f)))
    df = pd.concat(frames, ignore_index=True)
    df["Month"] = pd.to_datetime(df["Month"], format="%Y%m")
    return df.sort_values(["applicationID", "Month"]).reset_index(drop=True)


def build_per_control_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    for c in CONTROL_COLS:
        df[f"{c}_next"] = df.groupby("applicationID")[c].shift(-1)
    mask_unlabeled = df[[f"{c}_next" for c in CONTROL_COLS]].isna().all(axis=1)
    unlabeled_df = df[mask_unlabeled].copy()
    labeled_df = df[~mask_unlabeled].copy()
    for c in CONTROL_COLS:
        labeled_df[f"{c}_next"] = labeled_df[f"{c}_next"].astype(int)
    if unlabeled_df.empty:
        warnings.warn("No unlabeled (current) month detected; predictions will not be produced.")
    else:
        print(f"Unlabeled (current) month detected: {unlabeled_df['Month'].iloc[0].strftime('%Y-%m')} "
              f"rows={len(unlabeled_df)}")
    return labeled_df, unlabeled_df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    for c in CONTROL_COLS:
        df[f"{c}_prev"] = df.groupby("applicationID")[c].shift(1)
        df[f"{c}_delta"] = df[c] - df[f"{c}_prev"]
    df["gap_count"] = df[CONTROL_COLS].sum(axis=1)
    df["gap_count_prev"] = df.groupby("applicationID")["gap_count"].shift(1)
    df["gap_count_delta"] = df["gap_count"] - df["gap_count_prev"]
    df["gap_count_roll3"] = (
        df.groupby("applicationID")["gap_count"]
        .rolling(3, min_periods=1).sum()
        .reset_index(level=0, drop=True)
    )
    feature_cols = (
        CONTROL_COLS +
        [f"{c}_prev" for c in CONTROL_COLS] +
        [f"{c}_delta" for c in CONTROL_COLS] +
        ["gap_count", "gap_count_prev", "gap_count_delta", "gap_count_roll3"]
    )
    df[feature_cols] = df[feature_cols].fillna(0)
    return df, feature_cols


def temporal_split(labeled_df: pd.DataFrame):
    months = sorted(labeled_df["Month"].unique())
    val_month = months[-1]
    val_mask = labeled_df["Month"] == val_month
    train_mask = ~val_mask
    train_idx = labeled_df[train_mask].index
    val_idx = labeled_df[val_mask].index
    return train_idx, val_idx, val_month


def build_multi_output_model() -> Pipeline:
    base_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf = MultiOutputClassifier(base_rf, n_jobs=-1)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("clf", clf)
    ])
    return pipe


def evaluate_per_control(model: Pipeline,
                         X_val: pd.DataFrame,
                         y_val: pd.DataFrame):
    print("\nPer-Control Validation Metrics (predicting next-month gap = 1):")
    probs_list = model.predict_proba(X_val)
    summary_rows = []
    for i, c in enumerate(CONTROL_COLS):
        y_true = y_val[f"{c}_next"].values
        if probs_list[i].shape[1] == 2:
            p1 = probs_list[i][:, 1]
        else:
            p1 = np.zeros(len(y_true))
        unique = np.unique(y_true)
        if len(unique) < 2:
            roc = "NA"
            pr = "NA"
        else:
            roc = f"{roc_auc_score(y_true, p1):.3f}"
            pr = f"{average_precision_score(y_true, p1):.3f}"
        prevalence = y_true.mean()
        summary_rows.append((c, prevalence, roc, pr))
    print(f"{'Control':8s}  {'Prev':>5s}  {'ROC_AUC':>7s}  {'PR_AUC':>7s}")
    for c, prev, roc, pr in summary_rows:
        print(f"{c:8s}  {prev:5.2f}  {roc:>7s}  {pr:>7s}")


def save_control_model(model: Pipeline, feature_cols: List[str]):
    path = os.path.join(MODEL_DIR, "rf_per_control_multioutput.joblib")
    dump({"model": model, "features": feature_cols, "controls": CONTROL_COLS}, path)
    print(f"\nPer-control model saved: {path}")


def score_current_month(model: Pipeline,
                        unlabeled_df: pd.DataFrame,
                        feature_cols: List[str]):
    if unlabeled_df.empty:
        print("\nNo current (unlabeled) month to score.")
        return
    X_cur = unlabeled_df[feature_cols]
    probs_list = model.predict_proba(X_cur)
    control_prob_cols = {}
    for i, c in enumerate(CONTROL_COLS):
        arr = probs_list[i]
        p1 = arr[:, 1] if arr.shape[1] == 2 else np.zeros(arr.shape[0])
        control_prob_cols[f"pred_next_gap_prob_{c}"] = p1
    out = unlabeled_df[["applicationID", "Month"]].copy()
    out.rename(columns={"Month": "feature_month"}, inplace=True)
    next_month = (out["feature_month"].iloc[0] + pd.offsets.MonthBegin(1))
    out["predicts_for_month"] = next_month
    for k, v in control_prob_cols.items():
        out[k] = v
    fname = f"per_control_predictions_current_{out['feature_month'].iloc[0].strftime('%Y%m')}.csv"
    csv_path = os.path.join(OUTPUT_DIR, fname)
    out.to_csv(csv_path, index=False)
    print(f"\nPer-control next-month predictions saved: {csv_path}")


def main():
    print("Loading data...")
    raw = load_monthly_files()
    print(f"Loaded {len(raw)} rows across months: {[m.strftime('%Y-%m') for m in sorted(raw['Month'].unique())]}")
    labeled_df, unlabeled_df = build_per_control_labels(raw)
    combined = pd.concat([labeled_df, unlabeled_df], ignore_index=True)
    combined, feature_cols = engineer_features(combined)
    combined.set_index(["applicationID", "Month"], inplace=True)
    labeled_idx = labeled_df.set_index(["applicationID", "Month"]).index
    unlabeled_idx = unlabeled_df.set_index(["applicationID", "Month"]).index
    labeled_feat = combined.loc[labeled_idx].reset_index()
    unlabeled_feat = combined.loc[unlabeled_idx].reset_index()
    y_cols = [f"{c}_next" for c in CONTROL_COLS]
    X_labeled = labeled_feat[feature_cols]
    Y_labeled = labeled_feat[y_cols]
    train_idx, val_idx, val_month = temporal_split(labeled_feat)
    X_train, X_val = X_labeled.loc[train_idx], X_labeled.loc[val_idx]
    Y_train, Y_val = Y_labeled.loc[train_idx], Y_labeled.loc[val_idx]
    print(f"Train samples: {len(X_train)}  Val samples: {len(X_val)}  Val month: {val_month.strftime('%Y-%m')}")
    model = build_multi_output_model()
    model.fit(X_train, Y_train)
    evaluate_per_control(model, X_val, Y_val)
    save_control_model(model, feature_cols)
    final_model = build_multi_output_model()
    final_model.fit(X_labeled, Y_labeled)
    score_current_month(final_model, unlabeled_feat, feature_cols)
    print("\nDone.")


if __name__ == "__main__":
    main()
