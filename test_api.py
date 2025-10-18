import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# --- Test /health ---
resp = requests.get(f"{BASE_URL}/health")
print("Health check:", resp.status_code, resp.json())

# --- Test /predict_one (single record) ---
single_record = {
    "customerID": "CUST123",
    "tenure": 10,
    "MonthlyCharges": 60.5,
    "TotalCharges": 605.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "gender": "Male",
    "SeniorCitizen": "No"
}

resp = requests.post(f"{BASE_URL}/predict_one", json=single_record)
print("\nSingle record prediction:", resp.status_code)
print(json.dumps(resp.json(), indent=2))

# --- Test /predict (batch records) ---
batch_records = {
    "records": [
        {
            "customerID": "C1",
            "tenure": 1,
            "MonthlyCharges": 80,
            "TotalCharges": 80,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check",
            "gender": "Female",
            "SeniorCitizen": "No"
        },
        {
            "customerID": "C2",
            "tenure": 24,
            "MonthlyCharges": 55,
            "TotalCharges": 1320,
            "Contract": "One year",
            "InternetService": "DSL",
            "PaymentMethod": "Mailed check",
            "gender": "Male",
            "SeniorCitizen": "No"
        }
    ]
}

resp = requests.post(f"{BASE_URL}/predict", json=batch_records)
print("\nBatch prediction:", resp.status_code)
print(json.dumps(resp.json(), indent=2))

