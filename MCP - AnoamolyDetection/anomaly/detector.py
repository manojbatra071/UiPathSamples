# deep_compliance_mcp/anomaly/detector.py
import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# optional: sentence-transformers for text-mode
try:
    from sentence_transformers import SentenceTransformer
    _HAS_S2 = True
except Exception:
    _HAS_S2 = False

MODEL_DIR = os.environ.get("DCA_MODEL_DIR", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

class AnomalyDetector:
    def __init__(self):
        # model registry in-memory: model_name -> dict with model, scaler, metadata
        self.models = {}
        # hold loaded embedder instances if text-mode
        self.embedders = {}

    def _infer_mode_from_rows(self, rows: List[Dict[str, Any]]):
        df = pd.DataFrame(rows)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 1:
            return {"mode": "tabular", "feature_columns": numeric_cols}
        raise ValueError("Couldn't infer features; provide feature_columns, embedding_columns or text_column.")

    def _make_features(self, rows: List[Dict[str, Any]], metadata: Dict[str,Any], model_name: str):
        mode = metadata.get("mode")
        if mode == "tabular":
            cols = metadata["feature_columns"]
            df = pd.DataFrame(rows)
            X = df[cols].astype(float).values
            return X
        elif mode == "embedding":
            cols = metadata["embedding_columns"]
            df = pd.DataFrame(rows)
            X = df[cols].astype(float).values
            return X
        elif mode == "text":
            if not _HAS_S2:
                raise RuntimeError("sentence-transformers not installed; cannot embed text.")
            texts = [r[metadata["text_column"]] for r in rows]
            embedder = self.embedders.get(model_name)
            if embedder is None:
                # load embedder with model specified in metadata or default
                model_id = metadata.get("embedding_model", "all-MiniLM-L6-v2")
                embedder = SentenceTransformer(model_id)
                self.embedders[model_name] = embedder
            X = embedder.encode(texts, show_progress_bar=False)
            return np.asarray(X)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def train(self,
              rows: Optional[List[Dict[str,Any]]] = None,
              csv_path: Optional[str] = None,
              feature_columns: Optional[List[str]] = None,
              embedding_columns: Optional[List[str]] = None,
              text_column: Optional[str] = None,
              model_name: str = "isoforest_default",
              contamination: float = 0.01,
              random_state: int = 42,
              preprocess_scale: bool = True):
        # load rows
        if csv_path:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"{csv_path} not found")
            df = pd.read_csv(csv_path)
            rows = df.to_dict(orient="records")
        if not rows:
            raise ValueError("Provide rows or csv_path for training")

        # decide mode
        metadata = {}
        if feature_columns:
            metadata["mode"] = "tabular"
            metadata["feature_columns"] = feature_columns
        elif embedding_columns:
            metadata["mode"] = "embedding"
            metadata["embedding_columns"] = embedding_columns
        elif text_column:
            metadata["mode"] = "text"
            metadata["text_column"] = text_column
            metadata["embedding_model"] = "all-MiniLM-L6-v2"
            if not _HAS_S2:
                raise RuntimeError("Please install sentence-transformers for text mode.")
            # preload embedder for this model
            self.embedders[model_name] = SentenceTransformer(metadata["embedding_model"])
        else:
            # try auto infer
            metadata = self._infer_mode_from_rows(rows)

        X = self._make_features(rows, metadata, model_name)

        scaler = None
        if preprocess_scale:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
        else:
            Xs = X

        iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
        iso.fit(Xs)

        record = {"model": iso, "scaler": scaler, "metadata": metadata}
        self.models[model_name] = record

        # persist
        save_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        joblib.dump(record, save_path)

        return {"status": "trained", "model_name": model_name, "n_samples": int(X.shape[0]), "mode": metadata.get("mode")}

    def load(self, model_name: str):
        p = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Model file {p} not found")
        rec = joblib.load(p)
        self.models[model_name] = rec
        # if text-mode, ensure embedder exists
        meta = rec.get("metadata", {})
        if meta.get("mode") == "text":
            if not _HAS_S2:
                raise RuntimeError("Model requires sentence-transformers but it's not installed.")
            embed_model = meta.get("embedding_model", "all-MiniLM-L6-v2")
            self.embedders[model_name] = SentenceTransformer(embed_model)
        return {"status": "loaded", "model_name": model_name}

    def predict(self,
                model_name: str,
                rows: Optional[List[Dict[str,Any]]] = None,
                feature_vector: Optional[List[float]] = None,
                text: Optional[str] = None):
        if model_name not in self.models:
            self.load(model_name)

        rec = self.models[model_name]
        model: IsolationForest = rec["model"]
        scaler: Optional[StandardScaler] = rec["scaler"]
        metadata = rec["metadata"]

        X = None
        if feature_vector is not None:
            X = np.array(feature_vector, dtype=float).reshape(1, -1)
        elif rows is not None:
            X = self._make_features(rows, metadata, model_name)
        elif text is not None:
            if metadata.get("mode") != "text":
                raise ValueError("Model not configured for text input")
            if not _HAS_S2:
                raise RuntimeError("sentence-transformers not installed")
            embedder = self.embedders.get(model_name)
            if embedder is None:
                embedder = SentenceTransformer(metadata.get("embedding_model", "all-MiniLM-L6-v2"))
                self.embedders[model_name] = embedder
            X = embedder.encode([text])
        else:
            raise ValueError("Provide feature_vector, rows or text for prediction")

        Xs = scaler.transform(X) if scaler is not None else X
        df_scores = model.decision_function(Xs)   # higher -> more normal
        raw_labels = model.predict(Xs)            # 1 normal, -1 anomaly
        anomaly_scores = (-1.0 * df_scores).tolist()
        is_anomaly = [int(l) == -1 for l in raw_labels]
        return {
            "scores": anomaly_scores,
            "is_anomaly": is_anomaly,
            "raw_labels": raw_labels.tolist()
        }
