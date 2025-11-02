# prepare_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
model_name = "compliance_tabular_v1"
save_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")

# --- Synthetic training data ---
rng = np.random.RandomState(42)
# normal examples: features are [count, mean, stdev, min, max, median]
normal = np.hstack([
    rng.poisson(5, size=(500,1)),                      # count
    rng.normal(loc=100.0, scale=20.0, size=(500,1)),   # mean
    np.abs(rng.normal(loc=10.0, scale=3.0, size=(500,1))), # stdev
    rng.normal(loc=10.0, scale=5.0, size=(500,1)),     # min
    rng.normal(loc=200.0, scale=30.0, size=(500,1)),   # max
    rng.normal(loc=100.0, scale=20.0, size=(500,1)),   # median
])

# anomalies (outliers)
outliers = np.hstack([
    rng.poisson(50, size=(40,1)),
    rng.normal(loc=1000.0, scale=100.0, size=(40,1)),
    np.abs(rng.normal(loc=200.0, scale=50.0, size=(40,1))),
    rng.normal(loc=5.0, scale=2.0, size=(40,1)),
    rng.normal(loc=5000.0, scale=500.0, size=(40,1)),
    rng.normal(loc=1000.0, scale=100.0, size=(40,1)),
])

X = np.vstack([normal, outliers])
# simple scaler
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
iso.fit(Xs)

# Persist a dict with model, scaler and metadata
record = {"model": iso, "scaler": scaler, "metadata": {"mode":"tabular", "feature_columns":["count","mean","stdev","min","max","median"], "model_name": model_name}}
joblib.dump(record, save_path)
print("Saved model to", save_path)
