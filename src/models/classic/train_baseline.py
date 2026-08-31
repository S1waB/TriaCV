"""Sprint 2 – Classic ML Baseline Trainer for TriaCV.

Pipeline
--------
1. Load data/processed/clean_resumes.csv
2. TF-IDF vectorisation
3. Stratified train/test split (saved as split_indices.npz for Sprint 3 reuse)
4. Train 4 classifiers with lightweight GridSearch + 5-fold CV
5. Save best model + vectorizer to models_artifacts/
6. Delegate evaluation to src/evaluation/evaluate_classic.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Root is 3 levels up: src/models/classic -> src/models -> src -> PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_CSV   = PROJECT_ROOT / "data" / "processed" / "clean_resumes.csv"
ARTIFACTS_DIR   = PROJECT_ROOT / "models_artifacts"
SPLIT_FILE      = ARTIFACTS_DIR / "split_indices.npz"
REPORTS_DIR     = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_FOLDS      = 5


def sep(title: str = "") -> None:
    line = "=" * 70
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X_text, y_labels, class_names)."""
    df = pd.read_csv(PROCESSED_CSV)
    df = df.dropna(subset=["Clean_Resume", "Category"])
    df = df[df["Clean_Resume"].str.strip().str.len() > 10].reset_index(drop=True)
    X = df["Clean_Resume"].values
    le = LabelEncoder()
    y = le.fit_transform(df["Category"].values)
    return X, y, list(le.classes_)


def build_tfidf() -> TfidfVectorizer:
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
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
        {"clf__C": [0.1, 1.0, 5.0], "clf__solver": ["lbfgs"]},
    ),
    (
        "Linear SVM",
        LinearSVC(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
        {"clf__C": [0.1, 0.5, 1.0, 2.0]},
    ),
    (
        "Naive Bayes",
        MultinomialNB(),
        {"clf__alpha": [0.01, 0.1, 0.5, 1.0]},
    ),
    (
        "Random Forest",
        RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight="balanced",
        ),
        {"clf__max_depth": [None, 30], "clf__min_samples_split": [2, 5]},
    ),
]


def run_training() -> dict:
    """Train all models, run CV + GridSearch, save best model. Return results dict."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    sep("Sprint 2 | TF-IDF Vectorisation & Classic ML Training")

    # 1. Data
    X_text, y, class_names = load_data()
    print(f"  Loaded  : {len(X_text)} samples, {len(class_names)} categories")

    # 2. Stratified split
    indices = np.arange(len(X_text))
    idx_tr, idx_te = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_tr_txt, X_te_txt = X_text[idx_tr], X_text[idx_te]
    y_tr, y_te         = y[idx_tr],      y[idx_te]

    np.savez(SPLIT_FILE, idx_train=idx_tr, idx_test=idx_te)
    print(f"  Train   : {len(X_tr_txt)}  |  Test: {len(X_te_txt)}")
    print(f"  Split saved -> {SPLIT_FILE}")

    # 3. TF-IDF
    sep("Step 2/4 | TF-IDF fit on training set")
    tfidf = build_tfidf()
    X_tr = tfidf.fit_transform(X_tr_txt)
    X_te = tfidf.transform(X_te_txt)
    print(f"  Vocabulary size : {len(tfidf.vocabulary_):,}")
    print(f"  Train matrix    : {X_tr.shape}")
    print(f"  Test  matrix    : {X_te.shape}")

    # 4. Train & evaluate each model
    sep("Step 3/4 | GridSearch + 5-fold CV per model")
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {}

    for model_name, estimator, param_grid in MODEL_CONFIGS:
        print(f"\n  ------------------------------------------------------------")
        print(f"  Training : {model_name}")

        pipe = Pipeline([("clf", estimator)])

        gs = GridSearchCV(
            pipe, param_grid, cv=cv, scoring="f1_macro",
            n_jobs=-1, refit=True, verbose=0,
        )
        t0 = time.perf_counter()
        gs.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0

        best_pipe = gs.best_estimator_

        cv_scores = cross_val_score(best_pipe, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)

        t0 = time.perf_counter()
        y_pred = best_pipe.predict(X_te)
        infer_time = time.perf_counter() - t0

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        test_acc  = accuracy_score(y_te, y_pred)
        test_f1w  = f1_score(y_te, y_pred, average="weighted", zero_division=0)
        test_f1m  = f1_score(y_te, y_pred, average="macro",    zero_division=0)
        test_prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
        test_rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)

        results[model_name] = {
            "cv_f1_mean"  : float(cv_scores.mean()),
            "cv_f1_std"   : float(cv_scores.std()),
            "best_params" : gs.best_params_,
            "train_time_s": round(train_time, 2),
            "infer_time_s": round(infer_time, 4),
            "test_accuracy"          : round(test_acc,  4),
            "test_f1_weighted"       : round(test_f1w,  4),
            "test_f1_macro"          : round(test_f1m,  4),
            "test_precision_weighted": round(test_prec, 4),
            "test_recall_weighted"   : round(test_rec,  4),
            "model"                  : best_pipe,
            "y_pred"                 : y_pred,
        }

        print(f"    CV F1-macro  : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"    Test Accuracy: {test_acc:.4f}  |  F1-weighted: {test_f1w:.4f}  |  F1-macro: {test_f1m:.4f}")
        print(f"    Best params  : {gs.best_params_}")
        print(f"    Train time   : {train_time:.1f}s  |  Infer: {infer_time*1000:.1f}ms")

    # 5. Select & save best model
    sep("Step 4/4 | Best model selection & artefact export")
    # If tie, prioritize lowest inference latency / simplest model
    best_name = max(results, key=lambda k: (results[k]["test_f1_weighted"], results[k]["cv_f1_mean"], -results[k]["infer_time_s"]))
    best_info = results[best_name]

    print(f"\n  Best model : {best_name}")
    print(f"  F1-weighted: {best_info['test_f1_weighted']}  |  Accuracy: {best_info['test_accuracy']}")

    best_model_path   = ARTIFACTS_DIR / "best_classic_model.joblib"
    best_tfidf_path   = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
    best_classes_path = ARTIFACTS_DIR / "label_classes.joblib"

    joblib.dump(best_info["model"],  best_model_path,   compress=3)
    joblib.dump(tfidf,               best_tfidf_path,   compress=3)
    joblib.dump(class_names,         best_classes_path, compress=3)

    print(f"  Saved model    -> {best_model_path}")
    print(f"  Saved tfidf    -> {best_tfidf_path}")
    print(f"  Saved classes  -> {best_classes_path}")

    return {
        "results"      : results,
        "y_test"       : y_te,
        "class_names"  : class_names,
        "best_name"    : best_name,
        "tfidf"        : tfidf,
        "X_test_matrix": X_te,
    }


if __name__ == "__main__":
    run_training()
