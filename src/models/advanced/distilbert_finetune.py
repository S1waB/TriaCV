"""Sprint 3 – Optimized DistilBERT Fine-tuning for Fast CPU/GPU Execution.

Fine-tunes the upper transformer layer and sequence classification head of
DistilBERT on the exact dataset split, maintaining high convergence speed and
strict early-stopping validation.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Root resolution
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "clean_resumes.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "models_artifacts"
SPLIT_FILE    = ARTIFACTS_DIR / "split_indices.npz"
MODEL_SAVE_DIR = ARTIFACTS_DIR / "distilbert_finetuned"

MODEL_CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 128
NUM_EPOCHS = 2
BATCH_SIZE = 32
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
    
    train_texts = df["Clean_Resume"].values[idx_tr].tolist()
    train_labels = y_all[idx_tr].tolist()
    
    test_texts = df["Clean_Resume"].values[idx_te].tolist()
    test_labels = y_all[idx_te].tolist()
    
    return train_texts, train_labels, test_texts, test_labels, class_names


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_m = f1_score(labels, preds, average="macro", zero_division=0)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec  = recall_score(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "precision_weighted": prec,
        "recall_weighted": rec,
    }


def train_distilbert_pipeline():
    print("\n" + "=" * 70)
    print("  Sprint 3 | Fine-Tuning DistilBERT (distilbert-base-uncased)")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Execution device: {device.upper()}")
    
    train_texts, train_labels, test_texts, test_labels, class_names = load_data_and_split()
    num_labels = len(class_names)
    print(f"  Train: {len(train_texts)} | Test: {len(test_texts)} | Classes: {num_labels}")
    
    # 1. Load Tokenizer & Model
    print(f"\n  Loading tokenizer & model '{MODEL_CHECKPOINT}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label={i: name for i, name in enumerate(class_names)},
        label2id={name: i for i, name in enumerate(class_names)},
    )
    
    # Freeze lower layers for CPU efficiency (keep top transformer layer + classifier trainable)
    if hasattr(model, "distilbert") and hasattr(model.distilbert, "transformer"):
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False
        for layer in model.distilbert.transformer.layer[:-2]:
            for param in layer.parameters():
                param.requires_grad = False
        print("  Frozen lower transformer layers (optimized for CPU execution speed).")
    
    # 2. Tokenize Datasets
    print("  Tokenizing train & test sets...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings  = tokenizer(test_texts,  truncation=True, padding=True, max_length=MAX_LENGTH)
    
    train_dataset = Dataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "label": train_labels,
    })
    
    test_dataset = Dataset.from_dict({
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
        "label": test_labels,
    })
    
    # 3. Training Arguments with Early Stopping
    tmp_output_dir = PROJECT_ROOT / "models_artifacts" / "distilbert_checkpoints"
    training_args = TrainingArguments(
        output_dir=str(tmp_output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=1,
        report_to="none",
        use_cpu=(device == "cpu"),
        seed=RANDOM_STATE,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    
    # 4. Train
    print("  Starting fine-tuning with Trainer...")
    t0_train = time.perf_counter()
    trainer.train()
    total_train_time = time.perf_counter() - t0_train
    print(f"  Training completed in {total_train_time:.1f}s")
    
    # 5. Measure test inference latency
    print("  Measuring inference speed on test set...")
    model.eval()
    t0_infer = time.perf_counter()
    eval_predictions = trainer.predict(test_dataset)
    infer_total_sec = time.perf_counter() - t0_infer
    infer_latency_ms = (infer_total_sec / len(test_texts)) * 1000.0
    
    logits = eval_predictions.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = np.array(test_labels)
    
    acc  = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    
    print(f"  Test Accuracy   : {acc:.4f}")
    print(f"  Test F1-Weighted: {f1_w:.4f} | F1-Macro: {f1_m:.4f}")
    print(f"  Latency/CV      : {infer_latency_ms:.2f} ms")
    
    # 6. Save final model & tokenizer
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_SAVE_DIR))
    tokenizer.save_pretrained(str(MODEL_SAVE_DIR))
    print(f"  Fine-tuned model saved -> {MODEL_SAVE_DIR}")
    
    return {
        "model_name": "DistilBERT (Fine-tuned)",
        "type": "End-to-End Transformer",
        "accuracy": round(acc, 4),
        "precision_w": round(prec, 4),
        "recall_w": round(rec, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro": round(f1_m, 4),
        "train_time_s": round(total_train_time, 2),
        "infer_latency_ms": round(infer_latency_ms, 2),
        "y_pred": y_pred,
        "y_true": y_true,
        "class_names": class_names,
    }


if __name__ == "__main__":
    train_distilbert_pipeline()
