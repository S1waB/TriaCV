# Project Facts

## 1. Environment & Versions
- **OS:** Windows-10-10.0.26200-SP0
- **Python:** 3.11.9
- **Libraries:**
  - pandas: 3.0.5
  - numpy: 2.4.6
  - scikit-learn: 1.9.0
  - transformers: 5.16.1
  - sentence-transformers: 6.0.0
  - torch: 2.13.0+cpu
  - flask: 3.1.3
  - nltk: 3.10.3 (used for stopwords/lemmatization in `cleaner.py`)
  - spacy: 3.8.16 (installed but not utilized in current cleaning pipeline)
  - pdfplumber: 0.11.10
  - python-docx: 1.2.0

## 2. Hardware
- **CPU:** 13th Gen Intel(R) Core(TM) i7-13650HX
- **RAM:** 32 GB
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6141 MiB)
  - *Note:* PyTorch is installed as `cpu` only, thus the GPU is not actively used by Torch.

## 3. Preprocessing Pipeline
- **Library used:** NLTK (for stopwords and lemmatization).
- **Stop-word language:** English (fallback to English).

## 4. Modeling Setup
- **Train/Test Split:** 80/20 (test_size=0.20)
- **Stratified:** Yes (stratified on category)
- **Random Seed:** 42
- **Cross-Validation:** 5 Folds (StratifiedKFold)
- **Scoring Metric:** F1-Macro

## 5. Advanced Models
- **Sentence-BERT:** `all-MiniLM-L6-v2`
- **DistilBERT:** `distilbert-base-uncased`

## 6. Repository Tree (Depth 3)
```text
TriaCV/
├── api/                  - Flask API server routes and services
├── data/                 - Raw and processed datasets
├── frontend/             - React frontend using Vite
├── models_artifacts/     - Serialized models, vectorizers, and splits
├── notebooks/            - Exploratory data analysis notebooks
├── reports/              - Evaluation metrics, JSON, CSV and PNG visualizations
└── src/                  - Source code for preprocessing, models, and evaluation
```

## 7. Flask Routes
- `GET /api/health`: Health check endpoint. Returns `{"status": "healthy"}`
- `POST /api/predict`: Runs inference on a file or text. Expects multipart/form-data or JSON. Returns predictions from both models.
- `GET /api/history`: Returns in-memory session history.
- `DELETE /api/history`: Clears the session history.
- `GET /api/categories`: Returns the 25 supported job categories.

## 8. React Frontend
- **Build tool:** Vite 5.4.2
- **Components:**
  - `App.jsx`: Main application wrapper and state manager.
  - `UploadForm.jsx`: File and text upload form component.
  - `PredictionResult.jsx`: Displays classification results for both models.
  - `HistoryDrawer.jsx`: Sidebar drawer showing session history.
  - `CategoriesModal.jsx`: Modal displaying all supported categories.

## 9. Database
- **Engine:** In-memory Python List (`HistoryStore`)
- **Table names:** N/A (represented by `_history` list).
- **Columns stored:** `id`, `timestamp`, `filename`, `classic_category`, `classic_confidence`, `classic_confidence_percentage`, `advanced_category`, `advanced_confidence`, `advanced_confidence_percentage`, `agreement`, `keywords`, `word_count`.

## 10. Deliverables Checklist
- [x] **DONE:** Cleaned and documented dataset (`data/processed/clean_resumes.csv`)
- [x] **DONE:** Source code of baseline and advanced models + evaluation scripts (`src/models/`, `src/evaluation/`)
- [x] **DONE:** Comparison table of performances (`reports/final_comparison.csv`)
- [x] **DONE:** Working web application (`api/`, `frontend/`)
- [ ] **PARTIAL:** Internship report (`report_data/`)
- [ ] **MISSING:** Presentation material

## 11. Git Activity
- **Commits:** 7
- **Branches:** 1 (`main`)
- **First Commit Date:** Tue Sep 1 08:54:14 2026
- **Last Commit Date:** Tue Sep 1 08:54:14 2026

## 🚩 Discrepancy Flags (Differences from the plan)
- **SQLite:** The plan expected an SQLite database, but the app uses an in-memory Python list instead (`HistoryStore`).
- **GPU Usage:** Torch is installed as CPU-only despite having an RTX 4050 GPU available.
