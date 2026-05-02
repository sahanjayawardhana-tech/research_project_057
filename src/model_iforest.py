from sklearn.ensemble import IsolationForest

def train_iforest(X):
    model = IsolationForest(contamination=0.2, random_state=42)
    model.fit(X)
    preds = model.predict(X)
    preds = [1 if p == -1 else 0 for p in preds]
    return preds