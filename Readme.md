# Time-aware / Survival Analysis Churn

1. Create virtualenv, install requirements:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Preprocess:
   python src/data/preprocess.py   # will write data/processed/churn_survival.csv

3. Train:
   python src/models/train_survival.py

4. Predict:
   python src/models/predict_survival.py

5. Docker:
   docker build -t survival-churn .
   docker run -p 8001:8000 survival-churn
