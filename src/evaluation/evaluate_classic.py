"""Sprint 2 – Classic model evaluation, confusion matrix export, and comparison table.

Called by the training runner after models are trained; can also be run standalone
if models_artifacts/ already contains the saved artefacts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
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

# Root is 2 levels up: src/evaluation -> src -> PROJECT_ROOT
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "models_artifacts"
REPORTS_DIR   = PROJECT_ROOT / "reports"
sys.path.insert(0, str(PROJECT_ROOT))


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
    figsize: tuple[int, int] = (18, 15),
) -> None:
    """Save a normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))), normalize="true")

    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Normalized rate"},
        linewidths=0.4,
        linecolor="#e8e8e8",
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Category", fontsize=11, labelpad=8)
    ax.set_ylabel("True Category",      fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [CM] -> {out_path.name}")


def evaluate_all(
    results: dict,
    y_test: np.ndarray,
    class_names: list[str],
    best_name: str,
) -> pd.DataFrame:
    """Compute full evaluation, export CSVs and confusion-matrix PNGs."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    print("\n" + "=" * 70)
    print("  Sprint 2 | Evaluation Report")
    print("=" * 70)

    for model_name, info in results.items():
        y_pred = info["y_pred"]

        acc   = accuracy_score(y_test, y_pred)
        f1_w  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_m  = f1_score(y_test, y_pred, average="macro",    zero_division=0)
        prec  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec   = recall_score(y_test,  y_pred, average="weighted", zero_division=0)

        rows.append({
            "Model"              : model_name,
            "Accuracy"           : round(acc,  4),
            "Precision (W)"      : round(prec, 4),
            "Recall (W)"         : round(rec,  4),
            "F1 Weighted"        : round(f1_w, 4),
            "F1 Macro"           : round(f1_m, 4),
            "CV F1 Mean"         : round(info["cv_f1_mean"], 4),
            "CV F1 Std"          : round(info["cv_f1_std"],  4),
            "Train Time (s)"     : info["train_time_s"],
            "Infer Latency (ms)" : round(info["infer_time_s"] * 1000, 2),
            "Best Params"        : str(info["best_params"]),
            "Selected Baseline"  : "YES" if model_name == best_name else "NO",
        })

        safe_name = model_name.lower().replace(" ", "_")
        cm_path   = REPORTS_DIR / f"confusion_matrix_{safe_name}.png"
        plot_confusion_matrix(
            y_test, y_pred, class_names,
            title=f"Confusion Matrix - {model_name}",
            out_path=cm_path,
        )

        report_path = REPORTS_DIR / f"classification_report_{safe_name}.txt"
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(classification_report(
                y_test, y_pred,
                target_names=class_names,
                zero_division=0,
            ))

    comparison_df = pd.DataFrame(rows)

    csv_path = REPORTS_DIR / "classic_models_comparison.csv"
    comparison_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  Comparison table -> {csv_path}")

    return comparison_df


def print_comparison_table(df: pd.DataFrame) -> None:
    """Print a clean ASCII comparison table to stdout."""
    display_cols = [
        "Model", "Accuracy", "Precision (W)", "Recall (W)",
        "F1 Weighted", "F1 Macro", "CV F1 Mean", "CV F1 Std", "Selected Baseline",
    ]
    print("\n" + "-" * 95)
    print(df[display_cols].to_string(index=False))
    print("-" * 95)


if __name__ == "__main__":
    from src.models.classic.train_baseline import load_data

    print("[evaluate_classic] Running standalone evaluation from saved artefacts...")
    model  = joblib.load(ARTIFACTS_DIR / "best_classic_model.joblib")
    tfidf  = joblib.load(ARTIFACTS_DIR / "tfidf_vectorizer.joblib")
    labels = joblib.load(ARTIFACTS_DIR / "label_classes.joblib")
    split  = np.load(ARTIFACTS_DIR / "split_indices.npz")

    X_text, y, class_names = load_data()
    idx_te = split["idx_test"]
    X_te_txt = tfidf.transform(X_text[idx_te])
    y_te     = y[idx_te]
    y_pred   = model.predict(X_te_txt)

    acc  = accuracy_score(y_te, y_pred)
    f1_w = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    print(f"Best model  Accuracy={acc:.4f}  F1-weighted={f1_w:.4f}")
