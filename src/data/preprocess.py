# src/data/preprocess.py
import pandas as pd
import numpy as np
from typing import Tuple

def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def build_duration_event(df: pd.DataFrame,
                         id_col='customerID',
                         churn_col='churn',
                         tenure_col='tenure') -> pd.DataFrame:
    """
    For Telco-like datasets with 'tenure' measured in months:
      - duration = tenure
      - event = churn (1 if churned, else 0)
    If you have timestamped events, compute difference between start and churn or last observed date.
    """
    df = df.copy()
    # sanitize churn to 0/1
    df[churn_col] = df[churn_col].astype(int)
    # duration (use tenure if available)
    if tenure_col in df.columns:
        df['duration'] = pd.to_numeric(df[tenure_col], errors='coerce').fillna(0).astype(float)
    else:
        # fallback: derive from MonthlyCharges & TotalCharges as a heuristic
        df['duration'] = np.where(df['MonthlyCharges'] > 0,
                                  df['TotalCharges'] / (df['MonthlyCharges'] + 1e-9),
                                  0.0)
    df['event'] = df[churn_col].astype(int)
    return df

def select_features(df: pd.DataFrame, drop_cols=None):
    df = df.copy()
    if drop_cols is None:
        drop_cols = ['customerID','churn','tenure','duration','event']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X

def preprocess_for_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = build_duration_event(df)
    y_time = df['duration']
    y_event = df['event']
    X = select_features(df)
    # simple encoding of categoricals
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)
    return X, y_time, y_event

if __name__ == "__main__":
    df = load_raw("data/raw/telco_synthetic.csv")
    X, t, e = preprocess_for_model(df)
    df_out = pd.concat([X, t.rename('duration'), e.rename('event')], axis=1)
    df_out.to_csv("data/processed/churn_survival.csv", index=False)
    print("Wrote data/processed/churn_survival.csv")
