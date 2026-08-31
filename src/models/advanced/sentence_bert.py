"""Sprint 3 – Sentence-BERT Embedding + Downstream Classifier.

Pipeline:
1. Load data/processed/clean_resumes.csv using the exact split_indices.npz from Sprint 2.
2. Generate dense vector embeddings with sentence-transformers ('all-MiniLM-L6-v2').
3. Train a lightweight Logistic Regression classifier on top of embeddings.
4. Save the trained model and metadata in models_artifacts/.
5. Measure training time, test inference latency, and return predictions.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# Root resolution: src/models/advanced -> src/models -> src -> TraiCV
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "clean_resumes.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "models_artifacts"
SPLIT_FILE    = ARTIFACTS_DIR / "split_indices.npz"

MODEL_NAME = "all-MiniLM-L6-v2"
RANDOM_STATE = 42


def load_data_and_split():
    """Load preprocessed dataset with the exact train/test split from Sprint 2."""
    df = pd.read_csv(PROCESSED_CSV)
    df = df.dropna(subset=["Clean_Resume", "Category"]).reset_index(drop=True)
    
    le = LabelEncoder()
    y_all = le.fit_transform(df["Category"].values)
    class_names = list(le.classes_)
    
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found at {SPLIT_FILE}. Run Sprint 2 first.")
        
    split_data = np.load(SPLIT_FILE)
    idx_tr = split_data["idx_train"]
    idx_te = split_data["idx_test"]
    
    X_train_txt = df["Clean_Resume"].values[idx_tr]
    y_train = y_all[idx_tr]
    X_test_txt  = df["Clean_Resume"].values[idx_te]
    y_test = y_all[idx_te]
    
    return X_train_txt, y_train, X_test_txt, y_test, class_names


def train_sentence_bert_pipeline():
    """Encode texts via SentenceTransformer and train downstream classifier."""
    print("\n" + "=" * 70)
    print("  Sprint 3 | Sentence-BERT (all-MiniLM-L6-v2) + Classifier")
    print("=" * 70)
    
    X_tr_txt, y_tr, X_te_txt, y_te, class_names = load_data_and_split()
    print(f"  Train samples : {len(X_tr_txt)} | Test samples: {len(X_te_txt)}")
    print(f"  Categories    : {len(class_names)}")
    
    # 1. Load Sentence Transformer model
    print(f"\n  Loading Sentence-BERT model: '{MODEL_NAME}'...")
    t0_load = time.perf_counter()
    embedder = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded in {time.perf_counter() - t0_load:.2f}s")
    
    # 2. Generate embeddings
    print("  Generating embeddings for training set (1000 samples)...")
    t0_embed = time.perf_counter()
    X_tr_emb = embedder.encode(X_tr_txt.tolist(), show_progress_bar=False, batch_size=32, normalize_embeddings=True)
    embed_train_time = time.perf_counter() - t0_embed
    print(f"  Train embeddings generated in {embed_train_time:.2f}s (dim: {X_tr_emb.shape[1]})")
    
    # 3. Train downstream classifier
    print("  Training Logistic Regression on embeddings...")
    t0_train = time.perf_counter()
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE, class_weight="balanced")
    clf.fit(X_tr_emb, y_tr)
    clf_train_time = time.perf_counter() - t0_train
    total_train_time = embed_train_time + clf_train_time
    print(f"  Classifier trained in {clf_train_time:.2f}s (Total train time: {total_train_time:.2f}s)")
    
    # 4. Test inference
    print("  Running inference on test set (250 samples)...")
    t0_infer = time.perf_counter()
    X_te_emb = embedder.encode(X_te_txt.tolist(), show_progress_bar=False, batch_size=32, normalize_embeddings=True)
    y_pred = clf.predict(X_te_emb)
    infer_total_sec = time.perf_counter() - t0_infer
    infer_latency_ms = (infer_total_sec / len(X_te_txt)) * 1000.0
    
    # 5. Metrics
    acc  = accuracy_score(y_te, y_pred)
    f1_w = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_te, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    
    print(f"  Test Accuracy   : {acc:.4f}")
    print(f"  Test F1-Weighted: {f1_w:.4f} | F1-Macro: {f1_m:.4f}")
    print(f"  Latency/CV      : {infer_latency_ms:.2f} ms")
    
    # 6. Save artifact
    artifact_payload = {
        "model_name": MODEL_NAME,
        "classifier": clf,
        "class_names": class_names,
    }
    save_path = ARTIFACTS_DIR / "sentence_bert_classifier.joblib"
    joblib.dump(artifact_payload, save_path, compress=3)
    print(f"  Artifact saved  -> {save_path}")
    
    return {
        "model_name": "Sentence-BERT (MiniLM) + LR",
        "type": "Dense Embeddings",
        "accuracy": round(acc, 4),
        "precision_w": round(prec, 4),
        "recall_w": round(rec, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro": round(f1_m, 4),
        "train_time_s": round(total_train_time, 2),
        "infer_latency_ms": round(infer_latency_ms, 2),
        "y_pred": y_pred,
        "y_true": y_te,
        "class_names": class_names,
    }


if __name__ == "__main__":
    train_sentence_bert_pipeline()
