import pandas as pd
import joblib
import os

model = joblib.load("model_artifacts/model.pkl")
scaler = joblib.load("model_artifacts/scaler.pkl")

def predict(data: dict) -> dict:
    """
    Expected input:
    {
      "records": [
         {"transaction_amount": 120.5, "policy_code": 1, "other_feature": 0.85},
         ...
      ]
    }
    """
    df = pd.DataFrame(data["records"])
    X_scaled = scaler.transform(df)
    preds = model.predict(X_scaled)          # 1 = normal, -1 = anomaly
    df["prediction"] = ["normal" if p == 1 else "anomaly" for p in preds]
    return {"predictions": df.to_dict(orient="records")}
