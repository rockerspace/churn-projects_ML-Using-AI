import pandas as pd
import numpy as np
import os

def generate_synthetic_telco(path="data/raw/telco_synthetic.csv", n=2000, random_state=42):
    np.random.seed(random_state)
    cust_id = [f"CUST{100000+i}" for i in range(n)]
    tenure = np.random.exponential(scale=12, size=n).astype(int)
    monthly_charges = np.round(np.random.normal(70, 30, n).clip(10, 200), 2)
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 20, n), 2)
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.6, 0.25, 0.15])
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.35, 0.45, 0.2])
    payment_method = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n)
    gender = np.random.choice(["Male", "Female"], size=n)
    senior = np.random.choice(["Yes", "No"], size=n, p=[0.12, 0.88])
    churn_prob = (
        0.25 * (contract == "Month-to-month").astype(float)
        + 0.20 * (internet_service == "Fiber optic").astype(float)
        + 0.15 * (payment_method == "Electronic check").astype(float)
        - 0.01 * tenure
        + np.random.normal(0, 0.05, n)
    )
    churn_prob = (churn_prob - churn_prob.min()) / (churn_prob.max() - churn_prob.min())  # 0..1
    churn = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customerID": cust_id,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "InternetService": internet_service,
            "PaymentMethod": payment_method,
            "gender": gender,
            "SeniorCitizen": senior,
            "churn": churn,
        }
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Synthetic dataset written to {path}")
    return df

if __name__ == "__main__":
    generate_synthetic_telco()

