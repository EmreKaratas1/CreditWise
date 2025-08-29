# src/inference.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib


REPO_ROOT = Path(__file__).resolve().parents[1]  # .../CreditWise
MODELS_DIR = REPO_ROOT / "models"

# Artifact paths to be loaded
PATH_ENCODERS = MODELS_DIR / "label_encoders.pkl"
PATH_SCALER   = MODELS_DIR / "standard_scaler.pkl"
PATH_MODEL    = MODELS_DIR / "kmeans_model.pkl"
PATH_FEATURES = MODELS_DIR / "feature_cols.json"
PATH_SEGNAMES = MODELS_DIR / "segment_names.json"


def load_artifacts():
    encoders = joblib.load(PATH_ENCODERS)
    scaler   = joblib.load(PATH_SCALER)
    model    = joblib.load(PATH_MODEL)

    with open(PATH_FEATURES, "r") as f:
        feature_cols = json.load(f)
    with open(PATH_SEGNAMES, "r") as f:
        segment_names = json.load(f)

    return encoders, scaler, model, feature_cols, segment_names


def _safe_encode(value: Any, le) -> int:
    """Safe encoding for LabelEncoder:
    If an unseen category appears, fall back to 'unknown' if it exists; otherwise raise ValueError."""
    value = str(value)
    classes = le.classes_.tolist()
    if value not in classes:
        if "unknown" in classes:
            value = "unknown"
        else:
            raise ValueError(f"Unknown category '{value}' and no 'unknown' class in encoder.")
    return int(le.transform([value])[0])


def transform_row(
    row: Dict[str, Any],
    encoders: Dict[str, Any],
    scaler,
    feature_cols: list[str]
) -> np.ndarray:
    """Single line (dict) -> Model input (1, n_features)"""
    # Working copy
    r = dict(row)

    # Encode categoricals
    for col, le in encoders.items():
        if col not in r:
            raise KeyError(f"Missing required field: {col}")
        r[col] = _safe_encode(r[col], le)

    # Convert to DataFrame and guarantee column order
    X = pd.DataFrame([r])[feature_cols].copy()

    # Scale only numerical features
    num_cols = ["Age", "Duration", "Credit amount"]
    X[num_cols] = scaler.transform(X[num_cols])

    return X.values

def predict_row(row: Dict[str, Any]) -> Tuple[int, str]:
    """# Return cluster id and cluster name for a single record."""
    encoders, scaler, model, feature_cols, segment_names = load_artifacts()
    X = transform_row(row, encoders, scaler, feature_cols)
    cluster_id = int(model.predict(X)[0])
    cluster_name = segment_names.get(str(cluster_id)) or segment_names.get(cluster_id) or f"Cluster {cluster_id}"
    return cluster_id, cluster_name

# Test
if __name__ == "__main__":
    from pprint import pprint
    scenarios = [
        (
            "S1 • Younger / Short-Duration (expected: seg-1)",
            {
                "Age": 25,
                "Duration": 10,
                "Credit amount": 1200,
                "Sex": "male",
                "Job": 1,
                "Housing": "rent",
                "Saving accounts": "little",
                "Checking account": "moderate",
                "Purpose": "radio/TV",
            },
        ),
    ]

    for title, row in scenarios:
        cid, cname = predict_row(row)
        print(f"{title}\n→ Predicted: {cid} {cname}\n")
