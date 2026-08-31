# TriaCV — Système Intelligent de Classification et d'Analyse de CVs

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![Flask](https://img.shields.io/badge/API-Flask-black.svg)
![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61dafb.svg)
![Machine Learning](https://img.shields.io/badge/ML-Linear%20SVM%20%7C%20Sentence--BERT-orange.svg)

---

## 📌 Contexte du Projet

**TriaCV** est une solution complète de tri et de classification automatique de candidatures (CVs) structurée en 4 sprints méthodologiques :
1. **Sprint 1 — Données & Prétraitement NLP** : Acquisition, nettoyage avec préservation des termes techniques composés (`C++`, `.NET`, `Node.js`, etc.), lemmatisation, et analyse exploratoire (EDA).
2. **Sprint 2 — Baseline Classique (TF-IDF)** : Comparaison de 4 algorithmes classiques (Linear SVM, Logistic Regression, Naive Bayes, Random Forest) avec validation croisée 5-fold et GridSearch.
3. **Sprint 3 — Approches Avancées (Transformers)** : Embeddings denses Sentence-BERT (`all-MiniLM-L6-v2`) et fine-tuning DistilBERT avec benchmark complet précision/latence.
4. **Sprint 4 — Application Web & Déploiement** : API REST Flask modulaire, interface React + Vite responsive, et suite de tests pytest.

---

## 🏗️ Architecture du Projet

```
TraiCV/
├── api/                        # Backend REST Flask
│   ├── routes/                 # Endpoints (/predict, /categories, /history)
│   ├── services/               # Services de prédiction duale & historique
│   └── app.py                  # Application factory Flask
├── frontend/                   # Frontend moderne React + Vite
│   ├── src/                    # Composants d'upload, résultats, comparaison
│   └── vite.config.js          # Configuration avec proxy /api
├── src/
│   ├── preprocessing/          # Extracteur multi-format & cleaner NLP
│   ├── models/
│   │   ├── classic/            # Baseline TF-IDF (train_baseline.py)
│   │   └── advanced/           # Sentence-BERT & DistilBERT fine-tuning
│   ├── evaluation/             # Évaluation et métriques comparatives
│   └── utils/                  # Dataset loader & miroir Kaggle
├── models_artifacts/           # Modèles entraînés, vectoriseurs & splits
├── reports/                    # Matrices de confusion & rapports CSV
└── tests/                      # Suite de tests unitaires et d'intégration pytest
```

---

## 📊 Benchmark & Comparaison Finale des Modèles

Toutes les performances sont évaluées sur le **même split de test indépendant (250 CVs)** issu du split stratifié 80/20.

| Modèle | Type d'Approche | Accuracy | Précision (W) | Rappel (W) | F1 Weighted | F1 Macro | Latence Inférence | Temps Entraînement |
|---|---|---|---|---|---|---|---|---|
| **Linear SVM** ⭐ | **Classic (TF-IDF)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.90 ms** | **0.31s** |
| Logistic Regression | Classic (TF-IDF) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.10 ms | 5.21s |
| Naive Bayes | Classic (TF-IDF) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.60 ms | 0.09s |
| Random Forest | Classic (TF-IDF) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 51.00 ms | 1.75s |
| **Sentence-BERT (MiniLM) + LR** | **Dense Embeddings** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **18.17 ms** | **18.50s** |
| **DistilBERT (Fine-tuned)** | **End-to-End Transformer** | **0.9520** | **0.9663** | **0.9520** | **0.9450** | **0.9450** | **71.60 ms** | **318.90s** |

### 💡 Arbitrage Technique
- **Pour la classification de catégorie (Production)** : Le **Linear SVM (TF-IDF)** est la solution optimale (0.9 ms de latence, F1 parfait de 1.0).
- **Pour le matching sémantique offre ↔ CV** : Les **embeddings Sentence-BERT** offrent une compréhension contextuelle et des synonymes inégalée.

---

## 🚀 Installation & Lancement

### 1. Prérequis
- Python 3.11+
- Node.js 18+ et npm

### 2. Démarrage de l'API Backend Flask
```powershell
# Activer le venv
.\venv\Scripts\Activate.ps1

# Lancer l'API sur http://127.0.0.1:5000
python api/app.py
```

### 3. Démarrage du Frontend React + Vite
```powershell
cd frontend
npm install
npm run dev
# L'application est accessible sur http://localhost:5173
```

### 4. Lancement des Tests
```powershell
pytest tests/ -v
```

---

## 🔌 Documentation des Endpoints API

| Méthode | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Classification duale d'un CV (fichier PDF/DOCX/TXT ou texte brut) |
| `GET` | `/api/categories` | Liste des 25 catégories professionnelles supportées |
| `GET` | `/api/history` | Historique temporaire des prédictions de la session |
| `DELETE` | `/api/history` | Réinitialisation de l'historique de session |
| `GET` | `/api/health` | Vérification de l'état de l'API |
