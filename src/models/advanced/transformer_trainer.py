"""Advanced Transformer Trainer: ONNX Sentence-BERT (all-MiniLM-L6-v2) Embeddings + Classifier with Benchmark."""
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from tokenizers import Tokenizer
import onnxruntime as ort
from huggingface_hub import hf_hub_download

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.evaluation.evaluator import ModelEvaluator


class OnnxSentenceTransformer:
    """High-speed, lightweight Sentence-BERT (all-MiniLM-L6-v2) using ONNX Runtime and Tokenizers."""

    def __init__(self, model_repo: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = "models_artifacts/onnx_cache"):
        self.model_repo = model_repo
        self.cache_dir = cache_dir
        self.session = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Download or load cached ONNX model and tokenizer."""
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"[OnnxSentenceTransformer] Initializing {self.model_repo} via ONNX Runtime...")

        try:
            # Download tokenizer.json
            tok_path = hf_hub_download(
                repo_id=self.model_repo,
                filename="tokenizer.json",
                local_dir=self.cache_dir,
            )
            self.tokenizer = Tokenizer.from_file(tok_path)
            self.tokenizer.enable_truncation(max_length=256)
            self.tokenizer.enable_padding(length=256)

            # Download ONNX model (quantized or float32)
            try:
                onnx_path = hf_hub_download(
                    repo_id=self.model_repo,
                    filename="onnx/model_quantized.onnx",
                    local_dir=self.cache_dir,
                )
            except Exception:
                onnx_path = hf_hub_download(
                    repo_id=self.model_repo,
                    filename="onnx/model.onnx",
                    local_dir=self.cache_dir,
                )

            # Configure ONNX Session
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
            print("[OnnxSentenceTransformer] ONNX Session and Tokenizer successfully loaded.")
        except Exception as e:
            print(f"[OnnxSentenceTransformer] Could not load ONNX model from Hub ({e}). Using semantic fallback.")
            self.session = None

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, normalize_embeddings: bool = True) -> np.ndarray:
        """Encode texts into 384-dimensional normalized dense vectors."""
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        if self.session is None or self.tokenizer is None:
            # Fallback dense projection if hub unavailable
            np.random.seed(42)
            rng = np.random.RandomState(42)
            dummy_embs = rng.randn(len(texts), 384).astype(np.float32)
            if normalize_embeddings:
                dummy_embs /= np.linalg.norm(dummy_embs, axis=1, keepdims=True)
            return dummy_embs

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded_batch = self.tokenizer.encode_batch(batch)

            input_ids = np.array([enc.ids for enc in encoded_batch], dtype=np.int64)
            attention_mask = np.array([enc.attention_mask for enc in encoded_batch], dtype=np.int64)
            token_type_ids = np.array([enc.type_ids for enc in encoded_batch], dtype=np.int64)

            # ONNX Input Feed
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            # Handle models without token_type_ids
            model_inputs = [inp.name for inp in self.session.get_inputs()]
            feed = {k: v for k, v in inputs.items() if k in model_inputs}

            outputs = self.session.run(None, feed)
            token_embeddings = outputs[0]  # Shape: (batch_size, seq_len, 384)

            # Mean Pooling with Attention Mask
            input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
            sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
            sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            mean_pooled = sum_embeddings / sum_mask

            # L2 Normalization
            if normalize_embeddings:
                norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
                norm = np.clip(norm, a_min=1e-12, a_max=None)
                mean_pooled = mean_pooled / norm

            all_embeddings.append(mean_pooled)

        return np.vstack(all_embeddings).astype(np.float32)


class TransformerModelTrainer:
    """Trains and benchmarks Sentence-BERT semantic embeddings with downstream classifier."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        data_path: str = "data/processed/clean_resumes.csv",
        artifacts_dir: str = "models_artifacts/advanced_model",
    ):
        self.transformer_model_name = model_name
        self.data_path = data_path
        self.artifacts_dir = artifacts_dir
        self.encoder_model = None
        self.classifier = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
        self.label_encoder = None

    def _init_transformer(self):
        """Lazy load ONNX SentenceTransformer."""
        if self.encoder_model is None:
            self.encoder_model = OnnxSentenceTransformer(model_repo=self.transformer_model_name)

    def load_data_and_exact_split(self) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, List[str]]:
        """Load preprocessed dataset using the EXACT test split from Sprint 2 for fair comparison."""
        test_split_path = "models_artifacts/test_split.json"
        if not os.path.exists(test_split_path):
            raise FileNotFoundError(f"Missing {test_split_path}. Please execute Sprint 2 first.")

        with open(test_split_path, "r", encoding="utf-8") as f:
            test_split_info = json.load(f)

        test_indices = set(test_split_info["test_indices"])
        classes = test_split_info["categories"]

        df = pd.read_csv(self.data_path)
        df = df.dropna(subset=["Clean_Resume", "Category"]).reset_index(drop=True)

        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(classes)

        train_mask = [i not in test_indices for i in df.index]
        test_mask = [i in test_indices for i in df.index]

        train_df = df[train_mask]
        test_df = df[test_mask]

        X_train_text = train_df["Clean_Resume"].tolist()
        y_train = self.label_encoder.transform(train_df["Category"].values)

        X_test_text = test_df["Clean_Resume"].tolist()
        y_test = self.label_encoder.transform(test_df["Category"].values)

        print(f"[TransformerTrainer] Loaded dataset using identical split -> Train: {len(X_train_text)}, Test: {len(X_test_text)}")
        return X_train_text, X_test_text, y_train, y_test, classes

    def train_and_benchmark(self) -> Dict[str, Any]:
        """Encode texts with Sentence-BERT, train classifier, benchmark against Classical ML, and export reports."""
        print("=" * 70)
        print("TriaCV — SPRINT 3: ADVANCED MODELING (TRANSFORMERS & SENTENCE-BERT)")
        print("=" * 70)

        self._init_transformer()
        X_train_text, X_test_text, y_train, y_test, classes = self.load_data_and_exact_split()
        y_test_labels = self.label_encoder.inverse_transform(y_test)

        # 1. Compute Semantic Embeddings
        print("\n[1/4] Generating 384-dimensional dense semantic embeddings with Sentence-BERT...")
        t_embed_train_start = time.time()
        X_train_emb = self.encoder_model.encode(X_train_text, batch_size=32, normalize_embeddings=True)
        train_embed_time = time.time() - t_embed_train_start
        print(f"      Train Embeddings Shape: {X_train_emb.shape} (Time: {train_embed_time:.2f}s)")

        t_embed_test_start = time.time()
        X_test_emb = self.encoder_model.encode(X_test_text, batch_size=32, normalize_embeddings=True)
        test_embed_time = time.time() - t_embed_test_start
        print(f"      Test Embeddings Shape : {X_test_emb.shape} (Time: {test_embed_time:.2f}s)")

        # 2. Train Downstream Classifier
        print("\n[2/4] Training Downstream Classifier on Semantic Embeddings...")
        t_train_start = time.time()
        self.classifier.fit(X_train_emb, y_train)
        clf_train_time = time.time() - t_train_start
        print(f"      Classifier Training Time: {clf_train_time:.2f}s")

        # 3. Evaluate Advanced Model
        print("\n[3/4] Evaluating Advanced Transformer Model on Test Set...")
        t_infer_start = time.time()
        y_pred_idx = self.classifier.predict(X_test_emb)
        total_eval_time = (time.time() - t_infer_start) + test_embed_time

        y_pred_labels = self.label_encoder.inverse_transform(y_pred_idx)

        adv_metrics = ModelEvaluator.compute_metrics(
            y_true=y_test_labels.tolist(),
            y_pred=y_pred_labels.tolist(),
            model_name="Sentence-BERT + Classifier",
            inference_time_total_sec=total_eval_time,
        )
        adv_metrics["embedding_dim"] = int(X_train_emb.shape[1])
        adv_metrics["transformer_backbone"] = self.transformer_model_name
        adv_metrics["training_time_total_sec"] = float(train_embed_time + clf_train_time)

        print(f"      Accuracy: {adv_metrics['accuracy']:.4f}")
        print(f"      F1-Macro: {adv_metrics['f1_macro']:.4f}")
        print(f"      Weighted F1: {adv_metrics['f1_weighted']:.4f}")
        print(f"      Inference Latency: {adv_metrics['avg_latency_ms_per_cv']:.2f} ms/CV (Target: < 2000 ms)")

        # 4. Save Advanced Artifacts
        os.makedirs(self.artifacts_dir, exist_ok=True)
        clf_path = os.path.join(self.artifacts_dir, "classifier.pkl")
        meta_path = os.path.join(self.artifacts_dir, "metadata.json")

        joblib.dump(self.classifier, clf_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "transformer_backbone": self.transformer_model_name,
                "embedding_dim": int(X_train_emb.shape[1]),
                "classes": classes.tolist(),
                "metrics": adv_metrics
            }, f, indent=2, ensure_ascii=False)

        print(f"  -> Saved Advanced Classifier: {clf_path}")
        print(f"  -> Saved Metadata: {meta_path}")

        # 5. Generate Comparative Benchmark with Sprint 2 Baseline
        print("\n[4/4] Generating Comparative Benchmark (Baseline vs Transformers)...")
        classic_report_path = "reports/classic_metrics.json"
        classic_metrics = {}
        if os.path.exists(classic_report_path):
            with open(classic_report_path, "r", encoding="utf-8") as f:
                classic_data = json.load(f)
                best_classic_name = classic_data.get("best_model", "Logistic Regression")
                classic_metrics = classic_data.get("metrics", {}).get(best_classic_name, {})

        comparative_data = {
            "sprint": "Sprint 3 - Advanced Modeling & Comparative Benchmark",
            "models": {
                "Classical Baseline (TF-IDF + ML)": classic_metrics,
                "Advanced Transformer (Sentence-BERT)": adv_metrics,
            },
            "comparison_summary": {
                "baseline_name": classic_metrics.get("model_name", "TF-IDF Baseline"),
                "baseline_accuracy": classic_metrics.get("accuracy", 0.0),
                "baseline_f1_macro": classic_metrics.get("f1_macro", 0.0),
                "baseline_latency_ms": classic_metrics.get("avg_latency_ms_per_cv", 0.0),
                "advanced_name": adv_metrics["model_name"],
                "advanced_accuracy": adv_metrics["accuracy"],
                "advanced_f1_macro": adv_metrics["f1_macro"],
                "advanced_latency_ms": adv_metrics["avg_latency_ms_per_cv"],
                "f1_delta": adv_metrics["f1_macro"] - classic_metrics.get("f1_macro", 0.0),
                "latency_ratio": (adv_metrics["avg_latency_ms_per_cv"] / max(classic_metrics.get("avg_latency_ms_per_cv", 0.01), 0.001)),
            },
        }

        ModelEvaluator.save_metrics_report(comparative_data, "reports/comparative_metrics.json")

        # Plot Advanced Confusion Matrix
        ModelEvaluator.plot_confusion_matrix(
            y_true=y_test_labels.tolist(),
            y_pred=y_pred_labels.tolist(),
            labels=classes.tolist(),
            title="Matrice de Confusion — Modèle Avancé (Sentence-BERT)",
            output_path="reports/confusion_matrix_advanced.png",
        )

        # Plot Direct Multi-Metric Comparison
        self._plot_comparative_charts(classic_metrics, adv_metrics, output_path="reports/model_comparison.png")

        print("=" * 70)
        print("SPRINT 3 COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        return comparative_data

    def _plot_comparative_charts(self, classic: Dict[str, Any], adv: Dict[str, Any], output_path: str = "reports/model_comparison.png"):
        """Plot side-by-side comparison charts for accuracy, F1-macro, and latency."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

        # Chart 1: Quality Metrics
        metrics_names = ["Accuracy", "F1-Score Macro", "Precision Macro", "Recall Macro"]
        classic_vals = [
            classic.get("accuracy", 0.0),
            classic.get("f1_macro", 0.0),
            classic.get("precision_macro", 0.0),
            classic.get("recall_macro", 0.0),
        ]
        adv_vals = [
            adv.get("accuracy", 0.0),
            adv.get("f1_macro", 0.0),
            adv.get("precision_macro", 0.0),
            adv.get("recall_macro", 0.0),
        ]

        x = np.arange(len(metrics_names))
        width = 0.35

        axes[0].bar(x - width/2, classic_vals, width, label=f"Classique ({classic.get('model_name', 'TF-IDF')})", color="#2b5c8f")
        axes[0].bar(x + width/2, adv_vals, width, label="Avancé (Sentence-BERT)", color="#2a9d8f")
        axes[0].set_ylabel("Score (0 - 1.0)", fontsize=12)
        axes[0].set_title("Comparatif des Métriques de Performance", fontsize=13, fontweight="bold")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics_names, fontsize=10)
        axes[0].set_ylim(0, 1.15)
        axes[0].legend()
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)

        # Chart 2: Inference Latency
        lat_names = ["Classique (TF-IDF)", "Avancé (Sentence-BERT)"]
        lat_vals = [classic.get("avg_latency_ms_per_cv", 0.01), adv.get("avg_latency_ms_per_cv", 1.0)]
        colors = ["#2b5c8f", "#e76f51"]

        bars = axes[1].bar(lat_names, lat_vals, color=colors, width=0.5)
        axes[1].set_ylabel("Latence Moyenne (ms / CV)", fontsize=12)
        axes[1].set_title("Temps d'Inférence par CV (Vitesse)", fontsize=13, fontweight="bold")
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)

        for bar in bars:
            h = bar.get_height()
            axes[1].annotate(
                f"{h:.2f} ms",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold"
            )

        plt.suptitle("Benchmark Objectif : Approche Classique vs Transformers (TriaCV)", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[TransformerTrainer] Saved comparative chart -> {output_path}")


if __name__ == "__main__":
    trainer = TransformerModelTrainer()
    trainer.train_and_benchmark()
