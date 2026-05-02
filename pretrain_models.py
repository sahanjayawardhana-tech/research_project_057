#!/usr/bin/env python3
"""
Pre-train and save models to artifacts/ for fast dashboard startup.
Run once: python pretrain_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocess import load_email_data
from features import create_features
from training import train_detection_models, save_models

base_dir = Path(__file__).resolve().parent
email_path = base_dir / "data" / "r4.1" / "email.csv"
artifacts_dir = base_dir / "artifacts"

print("📧 Pre-training Email Threat Detection Models")
print("=" * 50)

print("1️⃣  Loading email data...")
raw = load_email_data(email_path)
print(f"   ✓ Loaded {len(raw):,} emails")

print("2️⃣  Engineering features...")
features = create_features(raw)
print(f"   ✓ Computed {len(features)} user profiles")

print("3️⃣  Training models (IsolationForest + PCA)...")
model_outputs = train_detection_models(features)
print(f"   ✓ Training complete")

print("4️⃣  Saving models to artifacts/...")
save_models(model_outputs, artifacts_dir)

print("\n" + "=" * 50)
print("✨ Pre-training complete!")
print(f"   Dashboard will now start instantly (~5 sec)")
