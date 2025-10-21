import pandas as pd
from src.data.preprocess import build_duration_event

def test_build_duration_event():
    df = pd.DataFrame({"customerID":["a","b"], "tenure":[3,10], "churn":[1,0]})
    out = build_duration_event(df)
    assert "duration" in out.columns
    assert "event" in out.columns
    assert out.loc[0,"event"] == 1
