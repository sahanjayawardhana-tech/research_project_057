import numpy as np
from pathlib import Path
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def _normalize(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array

    minimum = np.nanmin(array)
    maximum = np.nanmax(array)
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum - minimum == 0:
        return np.zeros_like(array, dtype=float)

    return (array - minimum) / (maximum - minimum)


def train_detection_models(features):
    numeric_features = features.drop(columns=["user"]).select_dtypes(include=[np.number]).fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_features)

    iforest = IsolationForest(contamination=0.08, random_state=42)
    iforest.fit(scaled)
    iforest_pred = (iforest.predict(scaled) == -1).astype(int)
    iforest_score = _normalize(-iforest.decision_function(scaled))

    n_components = min(8, scaled.shape[0], scaled.shape[1])
    if n_components < 1:
        lstm_score = np.zeros(scaled.shape[0], dtype=float)
        pca = None
    else:
        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(scaled)
        reconstructed = pca.inverse_transform(transformed)
        reconstruction_error = np.mean((scaled - reconstructed) ** 2, axis=1)
        lstm_score = _normalize(reconstruction_error)

    final_score = 0.6 * iforest_score + 0.4 * lstm_score

    return {
        "iforest_pred": iforest_pred,
        "iforest_score": iforest_score,
        "lstm_score": lstm_score,
        "final_score": final_score,
        "scaler": scaler,
        "iforest": iforest,
        "pca": pca,
    }


def save_models(model_dict, output_dir):
    """Save trained models to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model_dict["scaler"], output_dir / "scaler.joblib")
    joblib.dump(model_dict["iforest"], output_dir / "iforest.joblib")
    if model_dict["pca"] is not None:
        joblib.dump(model_dict["pca"], output_dir / "pca.joblib")
    print(f"✓ Models saved to {output_dir}")


def load_models(output_dir):
    """Load pre-trained models from disk."""
    output_dir = Path(output_dir)
    
    scaler = joblib.load(output_dir / "scaler.joblib")
    iforest = joblib.load(output_dir / "iforest.joblib")
    pca = joblib.load(output_dir / "pca.joblib") if (output_dir / "pca.joblib").exists() else None
    
    return {
        "scaler": scaler,
        "iforest": iforest,
        "pca": pca,
    }


def apply_pretrained_models(features, models):
    """Apply pre-trained models to new features."""
    numeric_features = features.drop(columns=["user"]).select_dtypes(include=[np.number]).fillna(0)
    
    scaler = models["scaler"]
    iforest = models["iforest"]
    pca = models["pca"]
    
    scaled = scaler.transform(numeric_features)
    
    iforest_pred = (iforest.predict(scaled) == -1).astype(int)
    iforest_score = _normalize(-iforest.decision_function(scaled))
    
    if pca is not None:
        transformed = pca.transform(scaled)
        reconstructed = pca.inverse_transform(transformed)
        reconstruction_error = np.mean((scaled - reconstructed) ** 2, axis=1)
        lstm_score = _normalize(reconstruction_error)
    else:
        lstm_score = np.zeros(scaled.shape[0], dtype=float)
    
    final_score = 0.6 * iforest_score + 0.4 * lstm_score
    
    return {
        "iforest_pred": iforest_pred,
        "iforest_score": iforest_score,
        "lstm_score": lstm_score,
        "final_score": final_score,
    }