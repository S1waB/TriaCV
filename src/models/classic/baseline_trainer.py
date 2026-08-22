"""Classic Baseline Trainer: TF-IDF vectorization + 4 ML classifiers with CV and serialization."""
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.evaluation.evaluator import ModelEvaluator


class ClassicBaselineTrainer:
    """Trains, optimizes, and evaluates 4 classic ML baseline models with TF-IDF."""

    def __init__(self, data_path: str = "data/processed/clean_resumes.csv", artifacts_dir: str = "models_artifacts"):
        self.data_path = data_path
        self.artifacts_dir = artifacts_dir
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.models: Dict[str, Any] = {
            "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            "Linear SVM": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, max_iter=2000)),
            "Naive Bayes": MultinomialNB(alpha=0.1),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        }
        self.best_model_name: str = ""
        self.best_model: Any = None
        self.results: Dict[str, Any] = {}

    def load_and_split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Load preprocessed dataset and create stratified 80/20 train/test split."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Processed data file not found at {self.data_path}. Run Sprint 1 first.")

        df = pd.read_csv(self.data_path)
        df = df.dropna(subset=["Clean_Resume", "Category"]).reset_index(drop=True)

        X_text = df["Clean_Resume"].values
        y_labels = df["Category"].values

        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y_labels)

        # Stratified train/test split (80/20)
        X_train_raw, X_test_raw, y_train, y_test, train_idx, test_idx = train_test_split(
            X_text,
            y_encoded,
            df.index.values,
            test_size=0.20,
            random_state=42,
            stratify=y_encoded,
        )

        # Save test split info for exact replication in Sprint 3 (Transformers)
        test_df = df.iloc[test_idx].copy()
        os.makedirs(self.artifacts_dir, exist_ok=True)
        test_split_info = {
            "test_indices": test_idx.tolist(),
            "categories": self.label_encoder.classes_.tolist(),
            "samples": [
                {
                    "index": int(idx),
                    "category": str(row["Category"]),
                    "resume_raw": str(row["Resume"]),
                    "resume_clean": str(row["Clean_Resume"]),
                }
                for idx, row in test_df.iterrows()
            ],
        }
        test_split_path = os.path.join(self.artifacts_dir, "test_split.json")
        with open(test_split_path, "w", encoding="utf-8") as f:
            json.dump(test_split_info, f, indent=2, ensure_ascii=False)
        print(f"[BaselineTrainer] Saved reproducible test split ({len(test_df)} samples) -> {test_split_path}")

        # Vectorize text features using TF-IDF
        print("[BaselineTrainer] Vectorizing text with TF-IDF (1-2 ngrams, max_features=5000)...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train_raw)
        X_test_tfidf = self.vectorizer.transform(X_test_raw)

        return X_train_tfidf, X_test_tfidf, y_train, y_test, test_df

    def train_and_evaluate(self) -> Dict[str, Any]:
        """Train all 4 models, perform 5-fold CV, evaluate on test set, and serialize best model."""
        print("=" * 70)
        print("TriaCV — SPRINT 2: CLASSICAL MODELING & BASELINE BENCHMARK")
        print("=" * 70)

        X_train, X_test, y_train, y_test, test_df = self.load_and_split_data()
        classes = self.label_encoder.classes_
        y_test_labels = self.label_encoder.inverse_transform(y_test)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        all_metrics = {}
        best_f1 = -1.0
        predictions_map = {}

        print(f"\n[1/4] Training and Cross-Validating 4 ML Models (Train samples: {X_train.shape[0]}, Features: {X_train.shape[1]})...")
        
        for name, model in self.models.items():
            print(f"\n---> Training: {name}...")
            # 5-fold Cross-Validation on training set
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
            mean_cv_f1 = float(cv_scores.mean())
            std_cv_f1 = float(cv_scores.std())
            print(f"     5-Fold CV F1-Macro: {mean_cv_f1:.4f} (+/- {std_cv_f1:.4f})")

            # Fit on full training set
            t_start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t_start

            # Measure inference latency on test set
            t_infer_start = time.time()
            y_pred_encoded = model.predict(X_test)
            total_infer_time = time.time() - t_infer_start

            y_pred_labels = self.label_encoder.inverse_transform(y_pred_encoded)
            predictions_map[name] = y_pred_labels

            # Compute standard metrics
            metrics = ModelEvaluator.compute_metrics(
                y_true=y_test_labels.tolist(),
                y_pred=y_pred_labels.tolist(),
                model_name=name,
                inference_time_total_sec=total_infer_time,
            )
            metrics["cv_f1_macro_mean"] = mean_cv_f1
            metrics["cv_f1_macro_std"] = std_cv_f1
            metrics["train_time_sec"] = float(train_time)

            all_metrics[name] = metrics
            print(f"     Test Accuracy: {metrics['accuracy']:.4f} | F1-Macro: {metrics['f1_macro']:.4f} | Latency: {metrics['avg_latency_ms_per_cv']:.2f}ms/CV")

            # Track best model based on F1-Macro on test set
            if metrics["f1_macro"] > best_f1:
                best_f1 = metrics["f1_macro"]
                self.best_model_name = name
                self.best_model = model

        print(f"\n[2/4] Best Classical Baseline Selected: {self.best_model_name} (F1-Macro = {best_f1:.4f})")

        # 3. Serialization of artifacts
        print("\n[3/4] Serializing Best Model, TF-IDF Vectorizer and LabelEncoder...")
        best_model_path = os.path.join(self.artifacts_dir, "classic_best_model.pkl")
        vectorizer_path = os.path.join(self.artifacts_dir, "tfidf_vectorizer.pkl")
        label_encoder_path = os.path.join(self.artifacts_dir, "label_encoder.pkl")

        joblib.dump(self.best_model, best_model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.label_encoder, label_encoder_path)

        print(f"  -> Saved Model       : {best_model_path}")
        print(f"  -> Saved Vectorizer  : {vectorizer_path}")
        print(f"  -> Saved LabelEncoder: {label_encoder_path}")

        # 4. Generate Reports & Visualizations
        print("\n[4/4] Generating Sprint 2 Evaluation Reports...")
        os.makedirs("reports", exist_ok=True)
        report_data = {
            "sprint": "Sprint 2 - Classical Modeling Baseline",
            "best_model": self.best_model_name,
            "models_evaluated": list(self.models.keys()),
            "metrics": all_metrics,
        }
        report_json_path = "reports/classic_metrics.json"
        ModelEvaluator.save_metrics_report(report_data, report_json_path)

        # Plot Best Model Confusion Matrix
        cm_output_path = "reports/confusion_matrix_classic.png"
        best_preds = predictions_map[self.best_model_name]
        ModelEvaluator.plot_confusion_matrix(
            y_true=y_test_labels.tolist(),
            y_pred=best_preds.tolist(),
            labels=classes.tolist(),
            title=f"Matrice de Confusion — Meilleur Modèle Classique ({self.best_model_name})",
            output_path=cm_output_path,
        )

        # Plot Comparison Bar Chart
        self._plot_models_comparison(all_metrics, output_path="reports/classic_models_comparison.png")

        print("=" * 70)
        print("SPRINT 2 COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        return report_data

    def _plot_models_comparison(self, all_metrics: Dict[str, Any], output_path: str = "reports/classic_models_comparison.png"):
        """Plot comparative bar chart across 4 ML models."""
        models = list(all_metrics.keys())
        accuracies = [all_metrics[m]["accuracy"] for m in models]
        f1_macros = [all_metrics[m]["f1_macro"] for m in models]
        cv_f1s = [all_metrics[m]["cv_f1_macro_mean"] for m in models]

        x = np.arange(len(models))
        width = 0.25

        plt.figure(figsize=(12, 6), dpi=300)
        plt.bar(x - width, accuracies, width, label="Test Accuracy", color="#2b5c8f")
        plt.bar(x, f1_macros, width, label="Test F1-Macro", color="#2a9d8f")
        plt.bar(x + width, cv_f1s, width, label="5-Fold CV F1-Macro", color="#e76f51")

        plt.xlabel("Modèle de Machine Learning", fontsize=12, labelpad=10)
        plt.ylabel("Score (0 - 1.0)", fontsize=12)
        plt.title("Comparatif des Performances des Modèles Classiques (TF-IDF)", fontsize=14, fontweight="bold", pad=15)
        plt.xticks(x, models, fontsize=11)
        plt.ylim(0, 1.1)
        plt.legend(fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[BaselineTrainer] Saved models comparison chart -> {output_path}")


if __name__ == "__main__":
    trainer = ClassicBaselineTrainer()
    trainer.train_and_evaluate()
