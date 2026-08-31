"""Sprint 1 - EDA & Preprocessing Pipeline Runner for TriaCV.

Generates:
  1. data/processed/clean_resumes.csv     - cleaned dataset
  2. reports/eda_category_distribution.png
  3. reports/eda_length_distribution.png
  4. reports/eda_top_keywords.png
  5. reports/eda_summary.json             - serialised stats
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution: make imports work regardless of where the script is called
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent          # notebooks/
PROJECT_ROOT = SCRIPT_DIR.parent                        # TraiCV/
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocessing.cleaner import ResumeCleaner
from src.utils.dataset_loader import load_or_create_dataset


# ---------------------------------------------------------------------------
# Directory helpers (all paths relative to PROJECT_ROOT)
# ---------------------------------------------------------------------------
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR   = PROJECT_ROOT / "reports"

PROCESSED_FILE = PROCESSED_DIR / "clean_resumes.csv"


def separator(title: str = "") -> None:
    line = "=" * 70
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(line)


def run_eda_and_preprocessing() -> None:
    separator("TriaCV | SPRINT 1: DATA ACQUISITION, PREPROCESSING & EDA")

    # ------------------------------------------------------------------
    # STEP 1 - Load raw data
    # ------------------------------------------------------------------
    separator("STEP 1/5 | Loading raw dataset")
    raw_df = load_or_create_dataset(str(RAW_DIR))
    print(f"  Rows      : {len(raw_df)}")
    print(f"  Categories: {raw_df['Category'].nunique()}")
    print(f"  Columns   : {list(raw_df.columns)}")

    null_counts = raw_df.isnull().sum()
    print(f"  Null counts:\n{null_counts.to_string()}")
    raw_df = raw_df.dropna(subset=["Category", "Resume"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # STEP 2 - Clean text
    # ------------------------------------------------------------------
    separator("STEP 2/5 | Cleaning & lemmatizing resumes")
    cleaner = ResumeCleaner(remove_stopwords=True, lemmatize=True)

    raw_df["Raw_Char_Length"]   = raw_df["Resume"].apply(len)
    raw_df["Raw_Word_Count"]    = raw_df["Resume"].apply(lambda x: len(str(x).split()))
    raw_df["Clean_Resume"]      = raw_df["Resume"].apply(cleaner.clean_text)
    raw_df["Clean_Char_Length"] = raw_df["Clean_Resume"].apply(len)
    raw_df["Clean_Word_Count"]  = raw_df["Clean_Resume"].apply(lambda x: len(str(x).split()))

    clean_df = raw_df[raw_df["Clean_Word_Count"] > 5].copy().reset_index(drop=True)
    retention = len(clean_df) / len(raw_df) * 100
    print(f"  Clean samples : {len(clean_df)}  (retention: {retention:.1f}%)")

    # ------------------------------------------------------------------
    # STEP 3 - Save processed dataset
    # ------------------------------------------------------------------
    separator("STEP 3/5 | Saving processed dataset")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_FILE, index=False, encoding="utf-8")
    print(f"  Saved -> {PROCESSED_FILE}")

    # ------------------------------------------------------------------
    # STEP 4 - Compute stats & save JSON summary
    # ------------------------------------------------------------------
    separator("STEP 4/5 | Computing corpus statistics")
    category_counts = clean_df["Category"].value_counts().to_dict()

    vocab_counter: Counter = Counter()
    for text in clean_df["Clean_Resume"]:
        vocab_counter.update(text.split())

    top_50 = vocab_counter.most_common(50)

    stats_summary = {
        "total_samples"   : int(len(clean_df)),
        "num_categories"  : int(clean_df["Category"].nunique()),
        "categories"      : list(category_counts.keys()),
        "category_distribution": {k: int(v) for k, v in category_counts.items()},
        "raw_metrics": {
            "avg_char_length"   : float(clean_df["Raw_Char_Length"].mean()),
            "median_char_length": float(clean_df["Raw_Char_Length"].median()),
            "avg_word_count"    : float(clean_df["Raw_Word_Count"].mean()),
            "min_word_count"    : int(clean_df["Raw_Word_Count"].min()),
            "max_word_count"    : int(clean_df["Raw_Word_Count"].max()),
        },
        "clean_metrics": {
            "avg_char_length"   : float(clean_df["Clean_Char_Length"].mean()),
            "avg_word_count"    : float(clean_df["Clean_Word_Count"].mean()),
            "total_tokens"      : int(sum(vocab_counter.values())),
            "unique_vocab_size" : int(len(vocab_counter)),
            "top_20_words"      : [{"word": w, "count": c} for w, c in top_50[:20]],
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "eda_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(stats_summary, fh, indent=2, ensure_ascii=False)
    print(f"  Stats saved -> {summary_path}")

    # ------------------------------------------------------------------
    # STEP 5 - Generate visualisations
    # ------------------------------------------------------------------
    separator("STEP 5/5 | Generating charts")
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })

    # --- Chart 1: Category distribution ---
    fig1, ax1 = plt.subplots(figsize=(14, 8), dpi=150)
    cat_series = clean_df["Category"].value_counts()
    colors = sns.color_palette("viridis", len(cat_series))
    bars = ax1.barh(cat_series.index.tolist(), cat_series.values.tolist(), color=colors)
    ax1.set_title("CV Distribution by Professional Category", fontsize=15, fontweight="bold", pad=14)
    ax1.set_xlabel("Number of CVs", labelpad=8)
    ax1.set_ylabel("Category")
    for bar, val in zip(bars, cat_series.values):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f" {val}", va="center", fontsize=9, color="#333")
    fig1.tight_layout()
    p1 = REPORTS_DIR / "eda_category_distribution.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  -> {p1}")

    # --- Chart 2: Length distribution (raw vs clean) ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    sns.histplot(clean_df["Raw_Word_Count"], bins=30, kde=True, ax=ax2a, color="#2b5c8f")
    ax2a.axvline(clean_df["Raw_Word_Count"].mean(), color="red", linestyle="--",
                 label=f"Mean: {clean_df['Raw_Word_Count'].mean():.0f}")
    ax2a.set_title("Word Count Distribution (Raw)", fontweight="bold")
    ax2a.set_xlabel("Word Count")
    ax2a.set_ylabel("Frequency")
    ax2a.legend()

    sns.histplot(clean_df["Clean_Word_Count"], bins=30, kde=True, ax=ax2b, color="#2a9d8f")
    ax2b.axvline(clean_df["Clean_Word_Count"].mean(), color="red", linestyle="--",
                 label=f"Mean: {clean_df['Clean_Word_Count'].mean():.0f}")
    ax2b.set_title("Word Count Distribution (After Cleaning & Lemmatisation)", fontweight="bold")
    ax2b.set_xlabel("Clean Word Count")
    ax2b.set_ylabel("Frequency")
    ax2b.legend()

    fig2.suptitle("Impact of Preprocessing Pipeline on CV Length", fontsize=15, fontweight="bold")
    fig2.tight_layout()
    p2 = REPORTS_DIR / "eda_length_distribution.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  -> {p2}")

    # --- Chart 3: Top-25 keywords ---
    top25 = top_50[:25]
    words_list  = [w for w, _ in top25]
    counts_list = [c for _, c in top25]
    fig3, ax3 = plt.subplots(figsize=(12, 8), dpi=150)
    palette = sns.color_palette("rocket", len(words_list))
    ax3.barh(words_list[::-1], counts_list[::-1], color=palette[::-1])
    ax3.set_title("Top 25 Most Frequent Keywords in Cleaned Corpus", fontsize=14, fontweight="bold", pad=12)
    ax3.set_xlabel("Total Occurrences")
    ax3.set_ylabel("Keyword")
    fig3.tight_layout()
    p3 = REPORTS_DIR / "eda_top_keywords.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  -> {p3}")

    # ------------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------------
    separator("SPRINT 1 COMPLETE")
    print(f"\n  Dataset source   : see [DatasetLoader] log above")
    print(f"  Total rows       : {len(clean_df)}")
    print(f"  Categories       : {clean_df['Category'].nunique()}")
    print(f"  Null values      : {int(clean_df.isnull().sum().sum())}")
    print(f"  Processed file   : {PROCESSED_FILE}")
    print(f"\n  Sample (first 3 rows):")
    preview_cols = ["Category", "Clean_Word_Count", "Clean_Resume"]
    preview = clean_df[preview_cols].head(3).copy()
    preview["Clean_Resume"] = preview["Clean_Resume"].apply(lambda x: x[:80] + "..." if len(x) > 80 else x)
    print(preview.to_string(index=True))
    separator()


if __name__ == "__main__":
    run_eda_and_preprocessing()
