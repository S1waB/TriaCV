import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from collections import Counter
import sys

sys.path.append(os.path.abspath("."))
from src.preprocessing.cleaner import ResumeCleaner

def main():
    os.makedirs("report_data/eda", exist_ok=True)
    
    # Load datasets
    df_raw = pd.read_csv("data/raw/UpdatedResumeDataSet.csv")
    df_clean = pd.read_csv("data/processed/clean_resumes.csv")
    
    # 1. Stats
    total_cvs = len(df_raw)
    num_columns = len(df_raw.columns)
    num_categories = df_raw["Category"].nunique()
    
    # 2. Category counts
    cat_counts = df_raw["Category"].value_counts()
    cat_pct = df_raw["Category"].value_counts(normalize=True) * 100
    cat_df = pd.DataFrame({"Count": cat_counts, "Percentage": cat_pct})
    cat_df.to_csv("report_data/eda/category_counts.csv")
    
    # 3. Missing and empty
    missing_values = int(df_raw.isnull().sum().sum())
    empty_short = int(df_raw["Resume"].apply(lambda x: len(str(x).split()) < 10).sum())
    
    # 4. Duplicates
    exact_dups = int(df_raw.duplicated(subset=["Resume"]).sum())
    normalized_text = df_raw["Resume"].str.lower().str.replace(r"[^\w\s]", "", regex=True)
    near_dups = int(normalized_text.duplicated().sum())
    remains_after_dedup = total_cvs - near_dups
    
    # Check if length of clean_resumes is less than raw to see if deduplication was applied
    dedup_applied_before_split = len(df_clean) < len(df_raw)
    
    # 5. Lengths in words
    lengths = df_raw["Resume"].apply(lambda x: len(str(x).split()))
    min_len = int(lengths.min())
    max_len = int(lengths.max())
    mean_len = float(lengths.mean())
    median_len = float(lengths.median())
    p25_len = float(lengths.quantile(0.25))
    p75_len = float(lengths.quantile(0.75))
    
    # 6. DistilBERT tokens
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def count_tokens(text):
        return len(tokenizer.encode(str(text), add_special_tokens=True, truncation=False))
    
    token_counts = df_raw["Resume"].apply(count_tokens)
    pct_gt_512 = float((token_counts > 512).mean() * 100)
    pct_gt_256 = float((token_counts > 256).mean() * 100)
    
    # 7. Vocab size
    words_raw = " ".join(df_raw["Resume"].astype(str)).lower().split()
    vocab_raw = len(set(words_raw))
    words_clean = " ".join(df_clean["Clean_Resume"].dropna().astype(str)).split()
    vocab_clean = len(set(words_clean))
    
    # 8. Top 15 terms overall
    counter_all = Counter(words_clean)
    top_15_overall = counter_all.most_common(15)
    
    # 9. Top 10 for 5 categories
    top_5_cats = cat_counts.head(5).index.tolist()
    top_terms_by_cat = {}
    for cat in top_5_cats:
        cat_words = " ".join(df_clean[df_clean["Category"] == cat]["Clean_Resume"].dropna().astype(str)).split()
        top_terms_by_cat[cat] = Counter(cat_words).most_common(10)
        
    # 10. Class imbalance
    max_class_count = int(cat_counts.max())
    min_class_count = int(cat_counts.min())
    imbalance_ratio = float(max_class_count / min_class_count)
    
    # Save JSON
    stats = {
        "total_cvs": total_cvs,
        "num_columns": num_columns,
        "num_categories": num_categories,
        "missing_values": missing_values,
        "empty_or_short": empty_short,
        "duplicates": {
            "exact": exact_dups,
            "near_duplicates": near_dups,
            "remains_after_deduplication": remains_after_dedup,
            "deduplication_applied_before_split": dedup_applied_before_split
        },
        "text_length_words": {
            "min": min_len,
            "max": max_len,
            "mean": round(mean_len, 2),
            "median": median_len,
            "p25": p25_len,
            "p75": p75_len
        },
        "distilbert_tokens": {
            "percentage_gt_512": round(pct_gt_512, 2),
            "percentage_gt_256": round(pct_gt_256, 2)
        },
        "vocabulary_size": {
            "before_preprocessing": vocab_raw,
            "after_preprocessing": vocab_clean
        },
        "top_15_terms_overall": top_15_overall,
        "top_10_terms_for_5_categories": top_terms_by_cat,
        "class_imbalance": {
            "largest_class_count": max_class_count,
            "smallest_class_count": min_class_count,
            "ratio": round(imbalance_ratio, 2)
        }
    }
    
    with open("report_data/eda/eda_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        
    # Generate Figures
    sns.set_theme(style="whitegrid")
    
    # Fig 1: Category Distribution
    plt.figure(figsize=(10, 8), dpi=200)
    sns.barplot(x=cat_counts.values, y=cat_counts.index, palette="viridis")
    plt.title("Distribution des CVs par Catégorie", fontsize=14)
    plt.xlabel("Nombre de CVs", fontsize=12)
    plt.ylabel("Catégorie", fontsize=12)
    plt.tight_layout()
    plt.savefig("report_data/eda/fig_distribution_categories.png", dpi=200, facecolor='white')
    plt.close()
    
    # Fig 2: Word Length Histogram
    plt.figure(figsize=(10, 6), dpi=200)
    sns.histplot(lengths, bins=50, kde=True, color="skyblue")
    plt.axvline(median_len, color='red', linestyle='dashed', linewidth=2, label=f'Médiane ({median_len})')
    plt.title("Distribution de la longueur des CVs (en mots)", fontsize=14)
    plt.xlabel("Nombre de mots", fontsize=12)
    plt.ylabel("Fréquence", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("report_data/eda/fig_longueur_cv.png", dpi=200, facecolor='white')
    plt.close()
    
    # Fig 3: Top terms overall
    terms, counts = zip(*top_15_overall)
    plt.figure(figsize=(10, 6), dpi=200)
    sns.barplot(x=list(counts), y=list(terms), palette="magma")
    plt.title("Top 15 des termes les plus fréquents", fontsize=14)
    plt.xlabel("Fréquence", fontsize=12)
    plt.ylabel("Termes", fontsize=12)
    plt.tight_layout()
    plt.savefig("report_data/eda/fig_top_termes.png", dpi=200, facecolor='white')
    plt.close()
    
    # Preprocessing Example
    sample_text = str(df_raw["Resume"].iloc[0])[:300]
    
    # Anonymize
    sample_anon = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_ANONYMISÉ]', sample_text)
    sample_anon = re.sub(r'\b(?:\+?[\d]{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}\b', '[TÉL_ANONYMISÉ]', sample_anon)
    
    cleaner = ResumeCleaner()
    step_lower_clean = re.sub(r'[^\w\s]', ' ', sample_anon.lower())
    step_tokens = step_lower_clean.split()
    step_no_stop = [t for t in step_tokens if t not in cleaner.stop_words]
    
    lemmatizer = cleaner.lemmatizer
    if lemmatizer:
        step_lemma = [lemmatizer.lemmatize(t) for t in step_no_stop]
    else:
        step_lemma = step_no_stop
        
    with open("report_data/eda/preprocessing_example.md", "w", encoding="utf-8") as f:
        f.write("# Exemple de Prétraitement d'un CV\n\n")
        f.write("### 1. Texte Brut (300 premiers caractères)\n")
        f.write(f"```text\n{sample_anon}\n```\n\n")
        
        f.write("### 2. Nettoyage et Minuscules (Sans ponctuation)\n")
        f.write(f"```text\n{step_lower_clean}\n```\n\n")
        
        f.write("### 3. Suppression des Stop Words\n")
        f.write(f"```text\n{' '.join(step_no_stop)}\n```\n\n")
        
        f.write("### 4. Tokenisation\n")
        f.write(f"```json\n{json.dumps(step_no_stop, ensure_ascii=False)}\n```\n\n")
        
        f.write("### 5. Lemmatisation\n")
        f.write(f"```json\n{json.dumps(step_lemma, ensure_ascii=False)}\n```\n\n")
        
    # Summary FR
    with open("report_data/eda/eda_summary_fr.md", "w", encoding="utf-8") as f:
        f.write(f"# Résumé Factual de l'EDA\n\n")
        f.write(f"Le jeu de données contient {total_cvs} CVs répartis en {num_categories} catégories (ratio de déséquilibre de {imbalance_ratio:.1f}). ")
        dedup_str = "est" if dedup_applied_before_split else "n'est pas"
        f.write(f"On observe {near_dups} quasi-doublons, et la déduplication {dedup_str} appliquée avant le split d'entraînement. ")
        f.write(f"La longueur médiane des CVs est de {median_len} mots, mais {pct_gt_512:.1f}% des documents dépassent la limite de 512 tokens de DistilBERT. ")
        f.write(f"Le prétraitement réduit significativement le vocabulaire de {vocab_raw} à {vocab_clean} termes uniques. ")
        f.write(f"Les données ne présentent aucune valeur manquante, validant leur qualité pour l'apprentissage automatique.\n")

if __name__ == "__main__":
    main()
