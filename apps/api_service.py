from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="HEADS CERT r4.2 API", version="1.0.0")
ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / 'data' / 'demo'

class BatchRequest(BaseModel):
    rows: list[dict]

@app.get('/health')
def health():
    return {"status": "ok", "project": "HEADS_CERT_R42"}

@app.get('/top_alerts')
def top_alerts(limit: int = 20):
    path = DEMO / 'alerts.csv'
    if not path.exists():
        return []
    df = pd.read_csv(path).head(limit)
    return df.to_dict(orient='records')

@app.get('/user/{user_id}')
def user_profile(user_id: str):
    path = DEMO / 'user_day_features.csv'
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    user_df = df[df['user'].astype(str).str.lower() == user_id.lower()]
    return {
        "user": user_id,
        "rows": user_df.to_dict(orient='records')[:200]
    }

@app.post('/predict_batch')
def predict_batch(req: BatchRequest):
    df = pd.DataFrame(req.rows)
    if df.empty:
        return []
    score = (df.select_dtypes(include='number').sum(axis=1) / 100.0).clip(0, 1)
    out = df.copy()
    out['malicious_probability'] = score
    out['severity'] = pd.cut(out['malicious_probability'], bins=[-0.01,0.35,0.60,0.80,1.0], labels=['low','medium','high','critical']).astype(str)
    return out.to_dict(orient='records')
