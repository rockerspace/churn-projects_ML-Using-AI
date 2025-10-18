#!/usr/bin/env python3
"""
Robust train.py for starter churn project.
Auto-detects LightGBM capabilities and uses correct callbacks for early stopping and logging.
"""
import os
import joblib
import pandas as pd
import argparse
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for col in ("customerID",):
        if col in cat_cols: cat_cols.remove(col)
        if col in num_cols: num_cols.remove(col)
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer([("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)], remainder="drop")
    return preprocessor

def train(save_dir="models", data_path="data/processed/churn.csv", seed=42, early_stop_rounds=30, num_boost_round=500, log_period=50):
    print("Running train.py from:", os.path.abspath(__file__))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path} - run src/data/preprocess.py first.")
    df = pd.read_csv(data_path)
    if "churn" not in df.columns:
        raise ValueError("Target column 'churn' not found in processed data.")
    y = df["churn"]
    X = df.drop(columns=["churn"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    dtrain = lgb.Dataset(X_train_t, label=y_train)
    dval = lgb.Dataset(X_val_t, label=y_val)
    params = {"objective": "binary", "metric": "auc", "verbosity": -1, "boosting_type": "gbdt", "seed": seed}
    print("Detected lightgbm version:", lgb.__version__)
    major = int(lgb.__version__.split(".")[0])
    # Build callbacks list depending on available API
    callbacks = []
    # early stopping
    if hasattr(lgb, "early_stopping"):
        try:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stop_rounds))
        except TypeError:
            # older signature fallback (rare)
            callbacks.append(lgb.callback.early_stopping(early_stop_rounds))
    # logging / evaluation
    if hasattr(lgb, "log_evaluation"):
        try:
            callbacks.append(lgb.log_evaluation(period=log_period))
        except TypeError:
            # fall back to alternative location
            try:
                callbacks.append(lgb.callback.log_evaluation(period=log_period))
            except Exception:
                pass
    # train with callbacks (works for >=4.x), older versions will ignore callbacks gracefully above
    model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=num_boost_round, callbacks=callbacks if callbacks else None)
    preds = model.predict(X_val_t)
    roc = roc_auc_score(y_val, preds)
    precision, recall, _ = precision_recall_curve(y_val, preds)
    pr_auc = auc(recall, precision)
    print(f"\nValidation ROC AUC: {roc:.4f}")
    print(f"Validation PR AUC: {pr_auc:.4f}")
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(save_dir, "preprocessor.joblib"))
    model.save_model(os.path.join(save_dir, "lgb_model.txt"))
    joblib.dump({"roc_auc": float(roc), "pr_auc": float(pr_auc)}, os.path.join(save_dir, "metrics.joblib"))
    print(f"Saved artifacts to {save_dir}")
    return model, preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/churn.csv")
    parser.add_argument("--save_dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=30)
    parser.add_argument("--num_boost", type=int, default=500)
    parser.add_argument("--log_period", type=int, default=50)
    args = parser.parse_args()
    train(save_dir=args.save_dir, data_path=args.data, seed=args.seed, early_stop_rounds=args.early_stop, num_boost_round=args.num_boost, log_period=args.log_period)
