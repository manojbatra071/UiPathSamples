# mcp_server_minimal.py
import os
import joblib
import json
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler



MODEL_DIR = os.environ.get("DCA_MODEL_DIR", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

mcp = FastMCP("AnomalyDetectMinimalMCP")

# simple in-memory cache of loaded models
_model_cache = {}

def _ensure_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]
    path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}")
    rec = joblib.load(path)
    # rec expected: {"model": IsolationForest, "scaler": StandardScaler, "metadata": {...}}
    _model_cache[model_name] = rec
    return rec


@mcp.tool()
def train_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trains an Isolation Forest model for anomaly detection using synthetic data,
    scales the data, and persists the model and scaler to the global MODEL_DIR.
    """
    # Use the globally defined MODEL_DIR, which respects the DCA_MODEL_DIR env var.
    # The os.makedirs(MODEL_DIR, exist_ok=True) call is handled globally.
    
    model_name = "compliance_tabular_v1"
    save_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")

    # --- Synthetic training data ---
    rng = np.random.RandomState(42)
    # normal examples: features are [count, mean, stdev, min, max, median]
    normal = np.hstack([
        rng.poisson(5, size=(500,1)),               # count
        rng.normal(loc=100.0, scale=20.0, size=(500,1)), # mean
        np.abs(rng.normal(loc=10.0, scale=3.0, size=(500,1))), # stdev
        rng.normal(loc=10.0, scale=5.0, size=(500,1)),  # min
        rng.normal(loc=200.0, scale=30.0, size=(500,1)), # max
        rng.normal(loc=100.0, scale=20.0, size=(500,1)), # median
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
    
    # Simple scaler (StandardScaler)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Train Isolation Forest model
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(Xs)

    # Persist a dict with model, scaler and metadata
    record = {"model": iso, "scaler": scaler, "metadata": {"mode":"tabular", "feature_columns":["count","mean","stdev","min","max","median"], "model_name": model_name}}
    joblib.dump(record, save_path)
    
    print("Saved model to", save_path)
    return {"status": "model trained and saved", "model_name": model_name, "path": save_path}


@mcp.tool()
def predict_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal predict tool.
    Accepts payload with either {"model_name": "...", "feature_vector": [...]} or {"model_name":"...", "text":"..."}
    For this demo we support only the tabular model saved as joblib. If text is sent but model is tabular, we'll return error.
    """
    try:
        model_name = payload.get("model_name", "compliance_tabular_v1")
        rec = _ensure_model(model_name)
        model = rec["model"]
        scaler = rec.get("scaler", None)
        metadata = rec.get("metadata", {})

        # Build feature matrix
        if "feature_vector" in payload:
            X = np.array(payload["feature_vector"], dtype=float).reshape(1, -1)
        elif "rows" in payload:
            import pandas as pd
            df = pd.DataFrame(payload["rows"])
            cols = payload.get("feature_columns") or metadata.get("feature_columns")
            if not cols:
                return {"error": "feature_columns required for rows input"}
            X = df[cols].astype(float).values
        elif "text" in payload:
            # For this minimal demo we do not support text->embedding conversion
            return {"error": "text input not supported by this minimal MCP. Use feature_vector for demo."}
        else:
            return {"error": "No input provided (feature_vector / rows / text)"}

        Xs = scaler.transform(X) if scaler is not None else X
        df_scores = model.decision_function(Xs)  # sklearn: higher normal, lower anomaly
        raw_labels = model.predict(Xs)           # 1 normal, -1 anomaly
        anomaly_scores = (-1.0 * df_scores).tolist()  # convert so higher -> more anomalous
        # normalize scores to 0..1 roughly (simple min-max cap for demo)
        # Note: for demo we'll clamp and scale but in prod use a consistent mapping
        scaled = []
        for s in anomaly_scores:
            # clamp and loosely scale; keep within [0,1]
            x = max(0.0, min(1.0, float(s)))
            scaled.append(x)
        return {"scores": scaled, "is_anomaly": [bool(int(l) == -1) for l in raw_labels], "raw_labels": raw_labels.tolist()}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # For local debug you can run the mcp serve loop; in MCP packaging the MCP runtime will start this.
    print("Starting minimal MCP server (FastMCP).")
    mcp.run()
