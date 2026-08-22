"""Performance evaluation, confusion matrix visualization, and benchmark reporting."""
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class ModelEvaluator:
    """Computes standard classification metrics, confusion matrices, and inference speed."""

    @staticmethod
    def compute_metrics(
        y_true: List[str],
        y_pred: List[str],
        model_name: str = "Model",
        inference_time_total_sec: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute full classification metrics dictionary."""
        acc = accuracy_score(y_true, y_pred)
        p_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        p_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        r_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        r_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        num_samples = len(y_true)
        avg_latency_ms = (inference_time_total_sec / num_samples * 1000.0) if num_samples > 0 else 0.0

        return {
            "model_name": model_name,
            "accuracy": float(acc),
            "precision_macro": float(p_macro),
            "precision_weighted": float(p_weighted),
            "recall_macro": float(r_macro),
            "recall_weighted": float(r_weighted),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "num_test_samples": int(num_samples),
            "total_inference_time_sec": float(inference_time_total_sec),
            "avg_latency_ms_per_cv": float(avg_latency_ms),
        }

    @staticmethod
    def plot_confusion_matrix(
        y_true: List[str],
        y_pred: List[str],
        labels: List[str],
        title: str = "Confusion Matrix",
        output_path: str = "reports/confusion_matrix.png",
        figsize: Tuple[int, int] = (16, 13),
    ) -> None:
        """Generate and save high-resolution normalized confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        
        plt.figure(figsize=figsize, dpi=300)
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Taux de classification (Normalisé)"},
            linewidths=0.5,
            linecolor="#f0f0f0"
        )
        plt.title(title, fontsize=15, fontweight="bold", pad=15)
        plt.xlabel("Catégorie Prédite", fontsize=12, labelpad=10)
        plt.ylabel("Catégorie Réelle", fontsize=12, labelpad=10)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[ModelEvaluator] Saved confusion matrix -> {output_path}")

    @staticmethod
    def save_metrics_report(metrics_dict: Dict[str, Any], output_path: str) -> None:
        """Save metrics dictionary to a formatted JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
        print(f"[ModelEvaluator] Saved metrics JSON -> {output_path}")
