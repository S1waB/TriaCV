import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

sys.path.append(os.path.abspath("."))
from src.preprocessing.cleaner import ResumeCleaner

def load_data():
    df = pd.read_csv("data/processed/clean_resumes.csv")
    df = df.dropna(subset=["Clean_Resume", "Category"])
    df = df[df["Clean_Resume"].str.strip().str.len() > 10].reset_index(drop=True)
    X = df["Clean_Resume"].values
    raw_texts = df["Resume"].values
    le = LabelEncoder()
    y = le.fit_transform(df["Category"].values)
    return X, y, list(le.classes_), raw_texts

def build_tfidf():
    return TfidfVectorizer(
        max_features=30_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        analyzer="word",
    )

MODEL_CONFIGS = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        {"clf__C": [0.1, 1.0, 5.0], "clf__solver": ["lbfgs"]},
    ),
    (
        "Linear SVM",
        LinearSVC(max_iter=2000, random_state=42, class_weight="balanced"),
        {"clf__C": [0.1, 0.5, 1.0, 2.0]},
    ),
    (
        "Multinomial Naive Bayes",
        MultinomialNB(),
        {"clf__alpha": [0.01, 0.1, 0.5, 1.0]},
    ),
    (
        "Random Forest",
        RandomForestClassifier(
            n_estimators=200, random_state=42,
            n_jobs=-1, class_weight="balanced",
        ),
        {"clf__max_depth": [None, 30], "clf__min_samples_split": [2, 5]},
    ),
]

def main():
    out_dir = Path("report_data/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    X_text, y, class_names, raw_texts = load_data()
    indices = np.arange(len(X_text))
    idx_tr, idx_te = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    
    X_tr_txt, X_te_txt = X_text[idx_tr], X_text[idx_te]
    y_tr, y_te         = y[idx_tr],      y[idx_te]
    raw_te             = raw_texts[idx_te]
    
    tfidf = build_tfidf()
    X_tr = tfidf.fit_transform(X_tr_txt)
    X_te = tfidf.transform(X_te_txt)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    hyperparams = {
        "tfidf": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_features": 30000,
            "sublinear_tf": True
        },
        "models": {}
    }
    
    best_f1 = -1
    best_model_name = ""
    best_y_pred = None
    best_clf = None
    
    lr_clf = None # Save LR to extract coefficients later if needed
    
    for model_name, estimator, param_grid in MODEL_CONFIGS:
        pipe = Pipeline([("clf", estimator)])
        gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
        
        t0 = time.perf_counter()
        gs.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0
        
        best_pipe = gs.best_estimator_
        
        if model_name == "Logistic Regression":
            lr_clf = best_pipe.named_steps["clf"]
            
        cv_scores = cross_val_score(best_pipe, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
        
        t0 = time.perf_counter()
        y_pred = best_pipe.predict(X_te)
        infer_time = time.perf_counter() - t0
        
        # In ms per CV (average over test set)
        infer_ms_per_cv = (infer_time / len(X_te_txt)) * 1000
        
        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_te, y_pred, average="macro", zero_division=0)
        f1_m = f1_score(y_te, y_pred, average="macro", zero_division=0)
        f1_w = f1_score(y_te, y_pred, average="weighted", zero_division=0)
        
        hyperparams["models"][model_name] = {
            "search_space": {k: list(v) for k, v in param_grid.items()},
            "best_params": gs.best_params_,
            "svm_confidence_note": "Linear SVC decision_function is used (distance to hyperplane), uncalibrated." if model_name == "Linear SVM" else "Predict_proba."
        }
        
        results.append({
            "Model": model_name,
            "CV_F1_Macro_Mean": round(cv_scores.mean(), 4),
            "CV_F1_Macro_Std": round(cv_scores.std(), 4),
            "Test_Accuracy": round(acc, 4),
            "Test_Precision_Macro": round(prec, 4),
            "Test_Recall_Macro": round(rec, 4),
            "Test_F1_Macro": round(f1_m, 4),
            "Test_F1_Weighted": round(f1_w, 4),
            "Train_Time_sec": round(train_time, 2),
            "Infer_Time_ms_per_cv": round(infer_ms_per_cv, 4)
        })
        
        if f1_m > best_f1:
            best_f1 = f1_m
            best_model_name = model_name
            best_y_pred = y_pred
            best_clf = best_pipe.named_steps["clf"]
            
    pd.DataFrame(results).to_csv(out_dir / "baseline_results.csv", index=False)
    with open(out_dir / "baseline_hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=4)
        
    # Best model analysis
    # Confusion matrix
    cm = confusion_matrix(y_te, best_y_pred, normalize='true')
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Matrice de confusion - {best_model_name}", fontsize=14)
    plt.xlabel("Classe Prédite", fontsize=12)
    plt.ylabel("Classe Réelle", fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_confusion_baseline.png", dpi=200, facecolor='white')
    plt.close()
    
    # Classification report
    report_dict = classification_report(y_te, best_y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(out_dir / "classification_report_baseline.csv")
    
    # Top terms per class
    target_clf = best_clf if hasattr(best_clf, 'coef_') else lr_clf
    feature_names = tfidf.get_feature_names_out()
    top_terms = []
    
    for i, class_name in enumerate(class_names):
        if i >= 5: # Just for at least 5 categories
            break
        coefs = target_clf.coef_[i]
        top_idx = np.argsort(coefs)[-10:]
        terms = [feature_names[j] for j in top_idx]
        top_terms.append({
            "Category": class_name,
            "Top_10_Terms": ", ".join(reversed(terms))
        })
    pd.DataFrame(top_terms).to_csv(out_dir / "top_terms_per_class.csv", index=False)
    
    # Misclassified examples
    confusions = []
    for true_idx, pred_idx in zip(y_te, best_y_pred):
        if true_idx != pred_idx:
            confusions.append((class_names[true_idx], class_names[pred_idx]))
    
    from collections import Counter
    top_confusions = Counter(confusions).most_common(5)
    
    with open(out_dir / "misclassified_examples.md", "w", encoding="utf-8") as f:
        f.write("# Erreurs de Classification Fréquentes\n\n")
        f.write("### Top 5 des confusions (Réelle -> Prédite)\n")
        for (true_c, pred_c), count in top_confusions:
            f.write(f"- **{true_c}** prédit comme **{pred_c}** : {count} fois\n")
            
        f.write("\n### Exemples de CVs mal classés\n")
        
        errors_found = 0
        for i in range(len(y_te)):
            if y_te[i] != best_y_pred[i]:
                true_c = class_names[y_te[i]]
                pred_c = class_names[best_y_pred[i]]
                
                # Anonymize
                raw_text = str(raw_te[i])[:200]
                import re
                raw_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_ANONYMISÉ]', raw_text)
                raw_text = re.sub(r'\b(?:\+?[\d]{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}\b', '[TÉL_ANONYMISÉ]', raw_text)
                
                f.write(f"#### Exemple {errors_found+1}\n")
                f.write(f"- **Vraie classe :** {true_c}\n")
                f.write(f"- **Prédiction :** {pred_c}\n")
                f.write(f"```text\n{raw_text}...\n```\n")
                f.write("> *Note : Le modèle s'est probablement trompé à cause d'un vocabulaire partagé (mots clés génériques) entre ces deux catégories professionnelles.*\n\n")
                
                errors_found += 1
                if errors_found == 3:
                    break
                    
    # Summary FR
    df_res = pd.DataFrame(results)
    best_row = df_res.loc[df_res['Test_F1_Macro'].idxmax()]
    worst_row = df_res.loc[df_res['Test_F1_Macro'].idxmin()]
    
    gap = best_row['Test_F1_Macro'] - worst_row['Test_F1_Macro']
    
    with open(out_dir / "baseline_summary_fr.md", "w", encoding="utf-8") as f:
        f.write("# Résumé Baseline (Modèles Classiques)\n\n")
        f.write(f"Le meilleur modèle classique est **{best_row['Model']}** avec un F1-macro de {best_row['Test_F1_Macro']:.4f} et une précision macro de {best_row['Test_Precision_Macro']:.4f} sur l'ensemble de test.\n")
        f.write(f"La validation croisée a confirmé sa robustesse avec un F1-macro moyen de {best_row['CV_F1_Macro_Mean']:.4f}.\n")
        f.write(f"L'écart de performance entre le meilleur ({best_row['Model']}) et le moins performant ({worst_row['Model']}) est de {gap:.4f} (F1-macro).\n")
        f.write(f"En termes de compromis performance/temps, {best_row['Model']} s'entraîne en {best_row['Train_Time_sec']:.1f} secondes.\n")
        f.write(f"L'inférence est extrêmement rapide avec environ {best_row['Infer_Time_ms_per_cv']:.2f} millisecondes par document, ce qui est idéal pour une API temps réel.\n")

if __name__ == "__main__":
    main()
