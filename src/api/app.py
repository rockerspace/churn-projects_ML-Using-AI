from fastapi import FastAPI, HTTPException, Body
from typing import List, Dict, Any
import joblib
import os
import pandas as pd
import lightgbm as lgb

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
PREPROCESSOR = None
MODEL = None

app = FastAPI(title="Churn Scoring API (safe)")

@app.on_event("startup")
def load_artifacts():
    global PREPROCESSOR, MODEL
    preproc_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    model_path = os.path.join(MODEL_DIR, "lgb_model.txt")
    if not os.path.exists(preproc_path) or not os.path.exists(model_path):
        print("Warning: model artifacts missing at startup:", preproc_path, model_path)
        PREPROCESSOR = None
        MODEL = None
        return
    PREPROCESSOR = joblib.load(preproc_path)
    MODEL = lgb.Booster(model_file=model_path)
    print("Loaded model artifacts.")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": PREPROCESSOR is not None and MODEL is not None}

@app.post("/predict")
def predict_batch(payload: Dict[str, Any] = Body(...)):
    """
    Expects JSON: { "records": [ {...}, {...} ] }
    Returns: { "predictions": [ {"customerID": ..., "churn_prob": ...}, ... ] }
    """
    if PREPROCESSOR is None or MODEL is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Run training.")

    records = payload.get("records")
    if not isinstance(records, list) or len(records) == 0:
        raise HTTPException(status_code=400, detail="Request JSON must include a non-empty 'records' list.")

    df = pd.DataFrame(records)
    if df.empty:
        raise HTTPException(status_code=400, detail="Empty records list.")

    if "churn" in df.columns:
        df = df.drop(columns=["churn"])

    # transform and predict
    try:
        X_t = PREPROCESSOR.transform(df)
        probs = MODEL.predict(X_t).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    ids = df.get("customerID", pd.Series([None] * len(df))).tolist()
    results = [{"customerID": cid, "churn_prob": float(p)} for cid, p in zip(ids, probs)]
    return {"predictions": results}

@app.post("/predict_one")
def predict_one(record: Dict[str, Any] = Body(...)):
    """
    Accepts a single JSON object with feature names (and optional customerID).
    Example: { "customerID":"C1", "tenure": 5, ... }
    """
    # reuse batch endpoint logic
    return predict_batch({"records": [record]})
