"""Exploratory Data Analysis (EDA) & Preprocessing Pipeline Runner.

Generates:
1. data/processed/clean_resumes.csv
2. reports/eda_category_distribution.png
3. reports/eda_length_distribution.png
4. reports/eda_top_keywords.png
5. reports/eda_summary.json
"""
import json
import os
import sys
from collections import Counter

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.preprocessing.cleaner import ResumeCleaner
from src.utils.dataset_loader import load_or_create_dataset


def run_eda_and_preprocessing():
    print("=" * 70)
    print("TriaCV — SPRINT 1: DATA ACQUISITION, PREPROCESSING & EDA")
    print("=" * 70)

    # 1. Load Raw Data
    raw_df = load_or_create_dataset("data/raw")
    print(f"\n[1/5] Raw Dataset Loaded: {len(raw_df)} samples across {raw_df['Category'].nunique()} categories.")
    print(f"Columns: {list(raw_df.columns)}")

    # Check for missing/null values
    null_counts = raw_df.isnull().sum().to_dict()
    print(f"Missing Values: {null_counts}")
    raw_df = raw_df.dropna(subset=["Category", "Resume"]).reset_index(drop=True)

    # 2. Apply Preprocessing Cleaner
    print("\n[2/5] Cleaning and normalizing resumes with ResumeCleaner...")
    cleaner = ResumeCleaner(remove_stopwords=True, lemmatize=True)
    
    raw_df["Raw_Char_Length"] = raw_df["Resume"].apply(len)
    raw_df["Raw_Word_Count"] = raw_df["Resume"].apply(lambda x: len(str(x).split()))
    
    raw_df["Clean_Resume"] = raw_df["Resume"].apply(cleaner.clean_text)
    raw_df["Clean_Char_Length"] = raw_df["Clean_Resume"].apply(len)
    raw_df["Clean_Word_Count"] = raw_df["Clean_Resume"].apply(lambda x: len(str(x).split()))

    # Filter any empty cleaned samples if any
    clean_df = raw_df[raw_df["Clean_Word_Count"] > 5].copy()
    print(f"Cleaned dataset: {len(clean_df)} valid samples (retention rate: {len(clean_df)/len(raw_df)*100:.1f}%)")

    # Save to data/processed
    os.makedirs("data/processed", exist_ok=True)
    processed_file = "data/processed/clean_resumes.csv"
    clean_df.to_csv(processed_file, index=False, encoding="utf-8")
    print(f"Saved processed dataset -> {processed_file}")

    # 3. Exploratory Data Analysis (EDA) Calculations
    print("\n[3/5] Computing statistical summaries...")
    category_counts = clean_df["Category"].value_counts().to_dict()
    
    vocab_counter = Counter()
    for text in clean_df["Clean_Resume"]:
        vocab_counter.update(text.split())
    
    total_tokens = sum(vocab_counter.values())
    unique_vocab_size = len(vocab_counter)
    top_50_words = vocab_counter.most_common(50)

    stats_summary = {
        "total_samples": int(len(clean_df)),
        "num_categories": int(clean_df["Category"].nunique()),
        "categories": list(category_counts.keys()),
        "category_distribution": category_counts,
        "raw_text_metrics": {
            "avg_char_length": float(clean_df["Raw_Char_Length"].mean()),
            "median_char_length": float(clean_df["Raw_Char_Length"].median()),
            "avg_word_count": float(clean_df["Raw_Word_Count"].mean()),
            "min_word_count": int(clean_df["Raw_Word_Count"].min()),
            "max_word_count": int(clean_df["Raw_Word_Count"].max()),
        },
        "clean_text_metrics": {
            "avg_char_length": float(clean_df["Clean_Char_Length"].mean()),
            "median_char_length": float(clean_df["Clean_Char_Length"].median()),
            "avg_word_count": float(clean_df["Clean_Word_Count"].mean()),
            "min_word_count": int(clean_df["Clean_Word_Count"].min()),
            "max_word_count": int(clean_df["Clean_Word_Count"].max()),
            "total_tokens": int(total_tokens),
            "unique_vocab_size": int(unique_vocab_size),
            "top_20_words": [{"word": w, "count": c} for w, c in top_50_words[:20]],
        },
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2, ensure_ascii=False)
    print("Saved EDA summary -> reports/eda_summary.json")

    # 4. Generate Visualizations
    print("\n[4/5] Generating publication-quality charts...")
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.sans-serif": "Arial", "font.family": "sans-serif"})

    # Figure 1: Category Distribution
    plt.figure(figsize=(14, 8), dpi=300)
    cat_series = clean_df["Category"].value_counts()
    ax = sns.barplot(
        x=cat_series.values,
        y=cat_series.index,
        palette="viridis",
        hue=cat_series.index,
        legend=False,
    )
    plt.title("Distribution des CVs par Catégorie Professionnelle (Dataset Kaggle)", fontsize=15, pad=15, fontweight="bold")
    plt.xlabel("Nombre de CVs", fontsize=12, labelpad=10)
    plt.ylabel("Catégorie Professionnelle", fontsize=12)
    for i, v in enumerate(cat_series.values):
        ax.text(v + 0.5, i, f" {v}", va="center", fontsize=10, color="#222")
    plt.tight_layout()
    plt.savefig("reports/eda_category_distribution.png", dpi=300)
    plt.close()
    print("  -> Exported reports/eda_category_distribution.png")

    # Figure 2: Length Distribution (Raw vs Clean)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    sns.histplot(clean_df["Raw_Word_Count"], bins=30, kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].set_title("Distribution du Nombre de Mots (Brut)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Nombre de Mots")
    axes[0].set_ylabel("Fréquence")
    axes[0].axvline(clean_df["Raw_Word_Count"].mean(), color="red", linestyle="--", label=f"Moyenne: {clean_df['Raw_Word_Count'].mean():.1f}")
    axes[0].legend()

    sns.histplot(clean_df["Clean_Word_Count"], bins=30, kde=True, ax=axes[1], color="#2a9d8f")
    axes[1].set_title("Distribution du Nombre de Mots (Après Nettoyage & Lemmatisation)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Nombre de Mots Nettoyés")
    axes[1].set_ylabel("Fréquence")
    axes[1].axvline(clean_df["Clean_Word_Count"].mean(), color="red", linestyle="--", label=f"Moyenne: {clean_df['Clean_Word_Count'].mean():.1f}")
    axes[1].legend()

    plt.suptitle("Impact du Pipeline de Prétraitement sur la Longueur des CVs", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("reports/eda_length_distribution.png", dpi=300)
    plt.close()
    print("  -> Exported reports/eda_length_distribution.png")

    # Figure 3: Top 25 Most Frequent Cleaned Words
    top25 = top_50_words[:25]
    words, counts = zip(*top25)
    plt.figure(figsize=(12, 7), dpi=300)
    sns.barplot(x=list(counts), y=list(words), palette="rocket", hue=list(words), legend=False)
    plt.title("Top 25 des Mots et Compétences les Plus Fréquents (Corpus Nettoyé)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Occurrences Totales", fontsize=12)
    plt.ylabel("Termes / Mots-clés", fontsize=12)
    plt.tight_layout()
    plt.savefig("reports/eda_top_keywords.png", dpi=300)
    plt.close()
    print("  -> Exported reports/eda_top_keywords.png")

    print("\n[5/5] SPRINT 1 PREPROCESSING & EDA COMPLETED SUCCESSFULLY!")
    print(f"Total processed samples: {len(clean_df)}")
    print(f"Processed file location: {os.path.abspath(processed_file)}")
    print("=" * 70)


if __name__ == "__main__":
    run_eda_and_preprocessing()
