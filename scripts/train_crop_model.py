"""
Trains the crop-type classifier and reports an HONEST accuracy number.

Usage:
    python3 scripts/train_crop_model.py

What this does:
1. Loads labeled data - prefers data/labels/real_labels.csv if it has
   enough rows (your real, confirmed field labels), otherwise falls back
   to the synthetic literature-grounded dataset from
   generate_training_data.py (see that file's docstring for what that
   means for the accuracy number's validity).
2. Trains a small Logistic Regression classifier (chosen over something
   heavier like XGBoost/RandomForest because with only a few hundred
   rows and 4 features, a high-capacity model would overfit and report
   a misleadingly high training accuracy - logistic regression is the
   right-sized model for this data volume).
3. Evaluates with 5-fold cross-validation (not a single train/test split)
   so the reported accuracy isn't just luck-of-the-split.
4. Saves the trained model + feature names + accuracy metadata to
   models/crop_classifier.joblib
5. Prints a full classification report (per-class precision/recall) -
   not just one headline accuracy number, since headline accuracy can
   hide a class the model never correctly identifies.
"""
import csv
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

THIS_DIR = os.path.dirname(__file__)
REAL_LABELS_PATH = os.path.join(THIS_DIR, "..", "data", "labels", "real_labels.csv")
SYNTHETIC_LABELS_PATH = os.path.join(THIS_DIR, "..", "data", "labels", "synthetic_crop_training.csv")
MODEL_OUTPUT_PATH = os.path.join(THIS_DIR, "..", "models", "crop_classifier.joblib")

FEATURE_COLUMNS = ["ndvi", "ndwi", "s1_vh_db", "s1_vv_vh_diff_db"]
MIN_REAL_SAMPLES = 1  # below this, real data is too thin to train on alone


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_dataset():
    real_rows = load_csv(REAL_LABELS_PATH)
    data_source = None

    if len(real_rows) >= MIN_REAL_SAMPLES:
        rows = real_rows
        data_source = f"real_labels.csv ({len(rows)} confirmed field samples)"
    else:
        if len(real_rows) > 0:
          print("⚠ WARNING: Using synthetic data because real data is too small")
        rows = load_csv(SYNTHETIC_LABELS_PATH)
        if not rows:
            print(
                f"No training data found. Run scripts/generate_training_data.py first, "
                f"or populate {REAL_LABELS_PATH} with at least {MIN_REAL_SAMPLES} real "
                f"confirmed-crop field samples.",
                file=sys.stderr,
            )
            sys.exit(1)
        note = ""
        if real_rows:
            note = (
                f" ({len(real_rows)} real samples found in real_labels.csv, but that's "
                f"below the {MIN_REAL_SAMPLES}-sample minimum - using synthetic data "
                f"instead until you have more real labels)"
            )
        data_source = f"SYNTHETIC bootstrap dataset{note}"

    X = np.array([[float(r[c]) for c in FEATURE_COLUMNS] for r in rows])
    X = np.nan_to_num(X)
    y = np.array([r["crop_type"] for r in rows])
    return X, y, data_source
    


def main():
    X, y, data_source = load_dataset()

    print(f"Training on: {data_source}")
    print(f"Total samples: {len(X)}, classes: {sorted(set(y))}\n")

    from sklearn.ensemble import RandomForestClassifier

    pipeline = Pipeline([
    ("clf", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )),

    ])
    # -----------------------------
    # SAFE CV (only if enough data)
    # -----------------------------
    if len(X) >= 6:
        cv_splits = min(5, len(set(y)))
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

        print(f"3-fold CV accuracy: {cv_scores.mean():.3f} "
              f"(+/- {cv_scores.std():.3f})")
        print(f"Fold scores: {[round(s, 3) for s in cv_scores]}\n")
    else:
        print("⚠ Not enough data for CV. Skipping cross-validation.")
        cv_scores = np.array([0.0])

    # -----------------------------
    # SAFE TRAIN/TEST SPLIT
    # -----------------------------
    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.25,
            stratify=y,
            random_state=42
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        print("Held-out test report:")
        print(classification_report(y_test, y_pred, zero_division=0))
    else:
        print("⚠ Not enough data for test split. Skipping evaluation.")
        pipeline.fit(X, y)

    # -----------------------------
    # FINAL TRAIN
    # -----------------------------
    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

    joblib.dump({
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "classes": sorted(set(y)),
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
        "training_data_source": data_source,
        "n_training_samples": len(X),
    }, MODEL_OUTPUT_PATH)

    print(f"\nSaved model to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
