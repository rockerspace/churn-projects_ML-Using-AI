#!/usr/bin/env python3
"""
predict.py — Generate churn probability predictions using saved model artifacts.
"""
import os
import joblib
import pandas as pd
import argparse
import lightgbm as lgb

def load_artifacts(model_dir="models"):
    preproc_path = os.path.join(model_dir, "preprocessor.joblib")
    model_path = os.path.join(model_dir, "lgb_model.txt")
    if not os.path.exists(preproc_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Missing model artifacts. Run src/models/train.py first.")
    preprocessor = joblib.load(preproc_path)
    model = lgb.Booster(model_file=model_path)
    return preprocessor, model

def predict_churn(df, preprocessor, model):
    X_t = preprocessor.transform(df)
    probs = model.predict(X_t)
    return probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default=None, help="Path to input CSV for prediction")
    parser.add_argument("--model_dir", default="models", help="Directory containing model artifacts")
    args = parser.parse_args()

    preprocessor, model = load_artifacts(model_dir=args.model_dir)

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
    else:
        # Minimal demo input row
        df = pd.DataFrame([{
            "customerID": "SAMPLE001",
            "tenure": 5,
            "MonthlyCharges": 75.0,
            "TotalCharges": 375.0,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check",
            "gender": "Male",
            "SeniorCitizen": "No",
            "avg_charge": 75.0
        }])

    # Drop target if present
    if "churn" in df.columns:
        df = df.drop(columns=["churn"])

    probs = predict_churn(df, preprocessor, model)
    for i, p in enumerate(probs):
        cid = df.iloc[i].get("customerID", f"row{i}")
        print(f"{cid}\tchurn_prob={p:.4f}")
