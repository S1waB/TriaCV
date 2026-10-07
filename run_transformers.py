import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback

# Make sure root is in sys path
sys.path.append(os.path.abspath("."))

def main():
    out_dir = Path("report_data/transformers")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    df = pd.read_csv("data/processed/clean_resumes.csv")
    df = df.dropna(subset=["Clean_Resume", "Category"])
    df = df[df["Clean_Resume"].str.strip().str.len() > 10].reset_index(drop=True)
    
    X_clean = df["Clean_Resume"].values
    X_raw = df["Resume"].values
    
    le = LabelEncoder()
    y = le.fit_transform(df["Category"].values)
    class_names = list(le.classes_)
    
    indices = np.arange(len(X_clean))
    idx_tr, idx_te = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    
    # Split
    X_tr_clean, X_te_clean = X_clean[idx_tr], X_clean[idx_te]
    X_tr_raw, X_te_raw     = X_raw[idx_tr], X_raw[idx_te]
    y_tr, y_te             = y[idx_tr], y[idx_te]
    
    results = []
    hyperparams = {}
    
    # =========================================================================
    # Strategy A: Sentence-BERT + Logistic Regression
    # =========================================================================
    sbert_model_name = "all-MiniLM-L6-v2"
    sbert = SentenceTransformer(sbert_model_name)
    max_seq_len = sbert.max_seq_length
    embed_dim = sbert.get_sentence_embedding_dimension()
    
    hyperparams["Strategy_A_SBERT"] = {
        "model_name": sbert_model_name,
        "embedding_dimension": embed_dim,
        "max_sequence_length": max_seq_len,
        "handling_long_cvs": "Truncation to max_seq_length (256 tokens)",
        "classifier": "Logistic Regression (C=1.0, class_weight='balanced')"
    }
    
    # Encode
    t0 = time.perf_counter()
    X_tr_emb = sbert.encode(X_tr_clean.tolist(), show_progress_bar=False, batch_size=32)
    t_embed_tr = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    X_te_emb = sbert.encode(X_te_clean.tolist(), show_progress_bar=False, batch_size=32)
    t_embed_te = time.perf_counter() - t0
    
    # Train
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    t0 = time.perf_counter()
    clf.fit(X_tr_emb, y_tr)
    t_train_clf = time.perf_counter() - t0
    
    # Infer
    t0 = time.perf_counter()
    y_pred_sbert = clf.predict(X_te_emb)
    t_infer_clf = time.perf_counter() - t0
    
    infer_ms_per_cv_sbert = ((t_embed_te + t_infer_clf) / len(y_te)) * 1000
    
    def get_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision_Macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall_Macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1_Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "F1_Weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0)
        }
    
    sbert_m = get_metrics(y_te, y_pred_sbert)
    results.append({
        "Model": "Sentence-BERT + LR",
        "Test_Accuracy": round(sbert_m["Accuracy"], 4),
        "Test_Precision_Macro": round(sbert_m["Precision_Macro"], 4),
        "Test_Recall_Macro": round(sbert_m["Recall_Macro"], 4),
        "Test_F1_Macro": round(sbert_m["F1_Macro"], 4),
        "Test_F1_Weighted": round(sbert_m["F1_Weighted"], 4),
        "Train_Time_sec": round(t_embed_tr + t_train_clf, 2),
        "Infer_Time_ms_per_cv": round(infer_ms_per_cv_sbert, 4),
        "y_pred": y_pred_sbert
    })
    
    # =========================================================================
    # Strategy A Ablation: Lightly cleaned (raw) vs Lemmatized
    # =========================================================================
    X_tr_emb_raw = sbert.encode(X_tr_raw.tolist(), show_progress_bar=False, batch_size=32)
    X_te_emb_raw = sbert.encode(X_te_raw.tolist(), show_progress_bar=False, batch_size=32)
    clf_raw = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    clf_raw.fit(X_tr_emb_raw, y_tr)
    y_pred_sbert_raw = clf_raw.predict(X_te_emb_raw)
    ablation_f1 = f1_score(y_te, y_pred_sbert_raw, average="macro", zero_division=0)
    
    hyperparams["Ablation"] = {
        "SBERT_Lemmatized_F1_Macro": round(sbert_m["F1_Macro"], 4),
        "SBERT_RawText_F1_Macro": round(ablation_f1, 4)
    }
    
    # =========================================================================
    # Strategy B: DistilBERT Fine-tuning
    # =========================================================================
    distilbert_model = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(distilbert_model)
    
    # Reduced settings for CPU
    is_cpu = not torch.cuda.is_available()
    epochs = 2 if is_cpu else 4
    batch_size = 4 if is_cpu else 16
    
    hyperparams["Strategy_B_DistilBERT"] = {
        "checkpoint": distilbert_model,
        "max_length": 512,
        "learning_rate": 2e-5,
        "batch_size": batch_size,
        "epochs": epochs,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "optimizer": "AdamW",
        "early_stopping": "Not applied (reduced epochs for CPU)",
        "device_used": "CPU" if is_cpu else "GPU",
        "reduced_settings_note": "Trained on CPU, so epochs reduced to 2 and batch_size to 4." if is_cpu else "None"
    }
    
    class ResumeDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512)
            self.labels = labels
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)

    train_dataset = ResumeDataset(X_tr_clean, y_tr)
    val_dataset = ResumeDataset(X_te_clean, y_te)
    
    model = AutoModelForSequenceClassification.from_pretrained(distilbert_model, num_labels=len(class_names))
    
    # Custom Callback to log metrics per epoch
    log_history = []
    class LogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                # Trainer logs loss, eval_loss, eval_f1 etc depending on step
                log_history.append(logs.copy())
                
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return {"f1_macro": f1_score(labels, preds, average="macro", zero_division=0)}
        
    training_args = TrainingArguments(
        output_dir="models_artifacts/distilbert_tmp",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=False,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[LogCallback()]
    )
    
    print("Starting DistilBERT training...")
    t0 = time.perf_counter()
    trainer.train()
    t_train_db = time.perf_counter() - t0
    
    print("Evaluating DistilBERT...")
    t0 = time.perf_counter()
    preds_output = trainer.predict(val_dataset)
    t_infer_db = time.perf_counter() - t0
    
    y_pred_db = np.argmax(preds_output.predictions, axis=1)
    db_m = get_metrics(y_te, y_pred_db)
    
    infer_ms_per_cv_db = (t_infer_db / len(y_te)) * 1000
    
    results.append({
        "Model": "DistilBERT Fine-tuned",
        "Test_Accuracy": round(db_m["Accuracy"], 4),
        "Test_Precision_Macro": round(db_m["Precision_Macro"], 4),
        "Test_Recall_Macro": round(db_m["Recall_Macro"], 4),
        "Test_F1_Macro": round(db_m["F1_Macro"], 4),
        "Test_F1_Weighted": round(db_m["F1_Weighted"], 4),
        "Train_Time_sec": round(t_train_db, 2),
        "Infer_Time_ms_per_cv": round(infer_ms_per_cv_db, 4),
        "y_pred": y_pred_db
    })
    
    # Extract logs for distilbert_training_log.csv
    epochs_data = []
    # parse log_history
    for e in range(1, epochs + 1):
        tr_loss = next((l.get("loss") for l in log_history if l.get("epoch") == e and "loss" in l), None)
        val_loss = next((l.get("eval_loss") for l in log_history if l.get("epoch") == e and "eval_loss" in l), None)
        val_f1 = next((l.get("eval_f1_macro") for l in log_history if l.get("epoch") == e and "eval_f1_macro" in l), None)
        epochs_data.append({"Epoch": e, "Train_Loss": tr_loss, "Validation_Loss": val_loss, "Validation_F1_Macro": val_f1})
        
    df_logs = pd.DataFrame(epochs_data)
    df_logs.to_csv(out_dir / "distilbert_training_log.csv", index=False)
    
    # Plot curves
    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot(df_logs["Epoch"], df_logs["Train_Loss"], label="Perte (Train)", marker='o')
    plt.plot(df_logs["Epoch"], df_logs["Validation_Loss"], label="Perte (Validation)", marker='s')
    plt.title("Courbes d'apprentissage - DistilBERT", fontsize=14)
    plt.xlabel("Époque", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_courbes_distilbert.png", facecolor='white')
    plt.close()
    
    overfitting_gap = 0
    if not df_logs.empty and df_logs["Train_Loss"].notnull().all() and df_logs["Validation_Loss"].notnull().all():
        last_train_loss = df_logs["Train_Loss"].iloc[-1]
        last_val_loss = df_logs["Validation_Loss"].iloc[-1]
        overfitting_gap = last_val_loss - last_train_loss
        
    hyperparams["DistilBERT_Overfitting_Check"] = {
        "Train_Loss_Final": last_train_loss if 'last_train_loss' in locals() else None,
        "Val_Loss_Final": last_val_loss if 'last_val_loss' in locals() else None,
        "Loss_Gap": overfitting_gap
    }
    
    # Find BEST model
    best_res = max(results, key=lambda x: x["Test_F1_Macro"])
    best_name = best_res["Model"]
    best_y_pred = best_res["y_pred"]
    
    # Clean up y_pred from results before saving
    for r in results:
        del r["y_pred"]
        
    pd.DataFrame(results).to_csv(out_dir / "transformers_results.csv", index=False)
    with open(out_dir / "transformers_hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=4)
        
    # BEST Model Outputs
    # Confusion matrix
    cm = confusion_matrix(y_te, best_y_pred, normalize='true')
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Oranges", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Matrice de confusion - {best_name}", fontsize=14)
    plt.xlabel("Classe Prédite", fontsize=12)
    plt.ylabel("Classe Réelle", fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_confusion_transformer.png", dpi=200, facecolor='white')
    plt.close()
    
    # Classification report
    report_dict = classification_report(y_te, best_y_pred, target_names=class_names, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(out_dir / "classification_report_transformer.csv")
    
    # Top 5 confusions
    confusions = []
    for t_idx, p_idx in zip(y_te, best_y_pred):
        if t_idx != p_idx:
            confusions.append((class_names[t_idx], class_names[p_idx]))
    
    top_confusions = Counter(confusions).most_common(5)
    
    # Summary FR
    with open(out_dir / "transformers_summary_fr.md", "w", encoding="utf-8") as f:
        f.write("# Résumé Transformers\n\n")
        
        if is_cpu:
            f.write("> **Note matérielle :** Exécuté sur CPU uniquement. Les paramètres de DistilBERT ont été réduits (2 époques, batch=4) pour terminer l'entraînement dans un temps raisonnable.\n\n")
            
        f.write(f"Le meilleur modèle basé sur les Transformers est **{best_name}** avec un F1-macro de {best_res['Test_F1_Macro']:.4f}.\n")
        f.write(f"Sentence-BERT obtient un F1-macro de {sbert_m['F1_Macro']:.4f}, avec une ablation révélant que le texte brut donne {ablation_f1:.4f}.\n")
        f.write(f"DistilBERT (fine-tuné) obtient un F1-macro de {db_m['F1_Macro']:.4f} avec un écart Train/Val Loss de {overfitting_gap:.4f}.\n")
        f.write(f"L'inférence de Sentence-BERT prend environ {infer_ms_per_cv_sbert:.1f} ms, contre {infer_ms_per_cv_db:.1f} ms pour DistilBERT.\n")
        
        if top_confusions:
            f.write("Les confusions les plus fréquentes sont :\n")
            for (tc, pc), count in top_confusions:
                f.write(f"- {tc} confondu avec {pc} ({count} fois)\n")
        else:
            f.write("Le modèle a atteint une précision parfaite sur cet ensemble de test, ne produisant aucune erreur de classification.\n")
            
if __name__ == "__main__":
    main()
