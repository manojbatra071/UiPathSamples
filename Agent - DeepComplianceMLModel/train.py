import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train():
    # AI Center passes the mounted dataset folder in DATASET_PATH env var
    data_path = os.getenv("DATASET_PATH", "synthetic_transactions.csv")
    df = pd.read_csv(data_path)

    X = df[["transaction_amount", "policy_code", "other_feature"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.025, random_state=42)
    model.fit(X_scaled)

    os.makedirs("model_artifacts", exist_ok=True)
    joblib.dump(model, "model_artifacts/model.pkl")
    joblib.dump(scaler, "model_artifacts/scaler.pkl")
    print("✅ Model trained and saved to model_artifacts/")

if __name__ == "__main__":
    train()
