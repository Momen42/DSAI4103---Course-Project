
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "churn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
META_PATH = os.path.join(BASE_DIR, "feature_metadata.json")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

feature_names = metadata["feature_names"]
numeric_features = metadata["numeric_features"]

def assign_risk_tier(p):
    if p < 0.30:
        return "Low"
    elif p < 0.60:
        return "Medium"
    return "High"

def preprocess_input(df):
    X = df.copy()

    # Remove ID if present
    if "CustomerID" in X.columns:
        X = X.drop(columns=["CustomerID"])

    # Feature engineering
    X["RevenueSegment"] = pd.cut(
        X["MonthlyCharges"],
        bins=[0, 35, 65, 200],
        labels=["Low", "Mid", "High"]
    )

    X["HasSupportService"] = (
        (X["OnlineBackup"] == "Yes") | (X["TechSupport"] == "Yes")
    ).astype(int)

    X["ChargesPerMonth"] = (
        X["TotalCharges"] / X["Tenure"].clip(lower=1)
    ).round(2)

    X["IsLongTermCustomer"] = (X["Tenure"] > 24).astype(int)
    X["HighComplaintFlag"] = (X["NumComplaints"] >= 4).astype(int)

    # Binary mapping
    binary_maps = {
        "Gender": {"Female": 0, "Male": 1},
        "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1},
        "PhoneService": {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1},
    }

    for col, mapping in binary_maps.items():
        X[col] = X[col].map(mapping)

    # One-hot encoding
    ohe_cols = [
        "InternetService", "OnlineBackup", "TechSupport",
        "StreamingTV", "Contract", "PaymentMethod", "RevenueSegment"
    ]
    X = pd.get_dummies(X, columns=ohe_cols, drop_first=True)

    # Convert bools to int
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    # Add missing columns expected by the model
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    # Keep exact training order
    X = X[feature_names].copy()

    # Numeric conversion and scaling
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X[numeric_features] = scaler.transform(X[numeric_features])

    return X

def score(input_path, output_path):
    df = pd.read_csv(input_path)

    customer_ids = df["CustomerID"] if "CustomerID" in df.columns else None

    X = preprocess_input(df)

    churn_prob = model.predict_proba(X)[:, 1]
    churn_pred = (churn_prob >= 0.50).astype(int)
    risk_tier = [assign_risk_tier(p) for p in churn_prob]

    result = pd.DataFrame({
        "ChurnProb": np.round(churn_prob, 4),
        "ChurnPred": churn_pred,
        "RiskTier": risk_tier
    })

    if customer_ids is not None:
        result.insert(0, "CustomerID", customer_ids.values)

    result.to_csv(output_path, index=False)
    print(f"Scoring complete. Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    args = parser.parse_args()

    score(args.input, args.output)
