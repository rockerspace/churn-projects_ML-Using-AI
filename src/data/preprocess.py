import pandas as pd
import os

def load_raw(path="data/raw/telco_synthetic.csv"):
    """Load raw CSV produced by fetch_data.py"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run src/data/fetch_data.py first.")
    return pd.read_csv(path)

def preprocess(df: pd.DataFrame):
    """Simple cleansing and feature engineering for the starter project."""
    df = df.copy()
    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Ensure numeric columns are numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # If churn is Yes/No (string), map to 1/0
    if df["churn"].dtype == object:
        mapping = {"Yes": 1, "No": 0}
        if set(df["churn"].unique()) & set(mapping.keys()):
            df["churn"] = df["churn"].map(mapping).fillna(df["churn"])
    # Ensure churn is integer 0/1
    if df["churn"].dtype != "int64" and df["churn"].dtype != "int32":
        try:
            df["churn"] = df["churn"].astype(int)
        except Exception:
            pass

    # Basic feature engineering
    if ("TotalCharges" in df.columns) and ("tenure" in df.columns):
        df["avg_charge"] = df["TotalCharges"] / df["tenure"].replace({0: 1})

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/churn.csv"
    df.to_csv(out_path, index=False)
    print(f"Processed data saved to {out_path} (shape={df.shape})")
    return df

def main():
    df_raw = load_raw()
    df_processed = preprocess(df_raw)
    print("Columns:", df_processed.columns.tolist())
    print("dtypes:")
    print(df_processed.dtypes)
    print("Value counts for churn:")
    print(df_processed["churn"].value_counts(dropna=False))

if __name__ == "__main__":
    main()

