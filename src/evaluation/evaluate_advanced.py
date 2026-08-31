"""Sprint 3 – Advanced evaluation & Final comparison merger."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
REPORTS_DIR   = PROJECT_ROOT / "reports"
CLASSIC_CSV   = REPORTS_DIR / "classic_models_comparison.csv"
FINAL_CSV     = REPORTS_DIR / "final_comparison.csv"
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


def generate_final_comparison(advanced_results: list[dict]) -> pd.DataFrame:
    """Merge classic baseline models (Sprint 2) and advanced Transformer models (Sprint 3)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Plot confusion matrices for advanced models
    for res in advanced_results:
        safe_name = res["model_name"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").replace("+", "_")
        cm_path = REPORTS_DIR / f"confusion_matrix_{safe_name}.png"
        plot_confusion_matrix(
            res["y_true"],
            res["y_pred"],
            res["class_names"],
            title=f"Confusion Matrix - {res['model_name']}",
            out_path=cm_path,
        )
    
    # 2. Build rows for advanced models
    adv_rows = []
    for res in advanced_results:
        adv_rows.append({
            "Model": res["model_name"],
            "Type": res["type"],
            "Accuracy": res["accuracy"],
            "Precision (W)": res["precision_w"],
            "Recall (W)": res["recall_w"],
            "F1 Weighted": res["f1_weighted"],
            "F1 Macro": res["f1_macro"],
            "Train Time (s)": res["train_time_s"],
            "Inference Latency (ms)": res["infer_latency_ms"],
        })
    df_adv = pd.DataFrame(adv_rows)
    
    # 3. Read classic models comparison
    if CLASSIC_CSV.exists():
        df_classic = pd.read_csv(CLASSIC_CSV)
        # Rename/align columns
        classic_rows = []
        for _, row in df_classic.iterrows():
            classic_rows.append({
                "Model": row["Model"],
                "Type": "Classic (TF-IDF)",
                "Accuracy": row["Accuracy"],
                "Precision (W)": row["Precision (W)"],
                "Recall (W)": row["Recall (W)"],
                "F1 Weighted": row["F1 Weighted"],
                "F1 Macro": row["F1 Macro"],
                "Train Time (s)": row["Train Time (s)"],
                "Inference Latency (ms)": row["Infer Latency (ms)"],
            })
        df_merged = pd.concat([pd.DataFrame(classic_rows), df_adv], ignore_index=True)
    else:
        df_merged = df_adv
        
    # Save final comparison CSV
    df_merged.to_csv(FINAL_CSV, index=False, encoding="utf-8")
    print(f"\n  Final comparison table saved -> {FINAL_CSV}")
    
    return df_merged


def print_final_table(df: pd.DataFrame) -> None:
    cols = ["Model", "Type", "Accuracy", "Precision (W)", "Recall (W)", "F1 Weighted", "F1 Macro", "Inference Latency (ms)", "Train Time (s)"]
    print("\n" + "=" * 115)
    print("  FINAL COMPARISON: SPRINT 2 (CLASSIC ML) vs SPRINT 3 (TRANSFORMERS)")
    print("=" * 115)
    print(df[cols].to_string(index=False))
    print("=" * 115)
