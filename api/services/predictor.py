"""Prediction Service for TriaCV.

Loads both Classic (Linear SVM / TF-IDF) and Advanced (Sentence-BERT MiniLM)
models, providing dual predictions with normalized confidence scores,
inference latency metrics, and extracted keywords.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.special import softmax
from sentence_transformers import SentenceTransformer

# Root resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.cleaner import ResumeCleaner
from src.preprocessing.extractor import ResumeExtractor

ARTIFACTS_DIR = PROJECT_ROOT / "models_artifacts"


class PredictionService:
    """Thread-safe prediction service for dual-model resume classification."""

    def __init__(self):
        self.cleaner = ResumeCleaner(remove_stopwords=True, lemmatize=True)
        self.extractor = ResumeExtractor()
        
        # Load artifacts
        self.classic_model = None
        self.tfidf_vectorizer = None
        self.class_names: List[str] = []
        self.sbert_embedder = None
        self.sbert_classifier = None
        
        self._load_models()

    def _load_models(self):
        """Load trained models from models_artifacts/."""
        # 1. Classic baseline model
        classic_path = ARTIFACTS_DIR / "best_classic_model.joblib"
        tfidf_path = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
        classes_path = ARTIFACTS_DIR / "label_classes.joblib"
        
        if classic_path.exists() and tfidf_path.exists() and classes_path.exists():
            self.classic_model = joblib.load(classic_path)
            self.tfidf_vectorizer = joblib.load(tfidf_path)
            self.class_names = joblib.load(classes_path)
            print(f"[PredictionService] Loaded classic model: {type(self.classic_model).__name__}")
        else:
            print("[PredictionService] Warning: Classic model artifacts not found.")

        # 2. Advanced Sentence-BERT model
        sbert_path = ARTIFACTS_DIR / "sentence_bert_classifier.joblib"
        if sbert_path.exists():
            payload = joblib.load(sbert_path)
            model_name = payload.get("model_name", "all-MiniLM-L6-v2")
            self.sbert_embedder = SentenceTransformer(model_name)
            self.sbert_classifier = payload.get("classifier")
            if not self.class_names and "class_names" in payload:
                self.class_names = payload["class_names"]
            print(f"[PredictionService] Loaded Sentence-BERT model: '{model_name}' + Classifier")
        else:
            print("[PredictionService] Warning: Sentence-BERT artifacts not found.")

    def get_categories(self) -> List[str]:
        """Return list of supported job categories."""
        return sorted(self.class_names)

    def extract_and_clean_text(self, file_source: Any, filename: str = "", raw_text: Optional[str] = None) -> Tuple[str, str]:
        """Extract text from file or string, then apply cleaning pipeline."""
        if raw_text and raw_text.strip():
            raw = raw_text.strip()
        elif file_source is not None:
            raw = self.extractor.extract(file_source, filename=filename)
        else:
            raw = ""

        if not raw or len(raw.strip()) < 5:
            raise ValueError("Le contenu du CV est vide ou illisible.")

        clean = self.cleaner.clean_text(raw)
        if not clean or len(clean.split()) < 3:
            raise ValueError("Le CV contient trop peu de texte exploitable après nettoyage.")

        return raw, clean

    def _predict_classic(self, clean_text: str) -> Dict[str, Any]:
        """Compute prediction & calibrated confidence from TF-IDF + Classic model."""
        t0 = time.perf_counter()
        X_vec = self.tfidf_vectorizer.transform([clean_text])
        
        clf = self.classic_model
        if hasattr(clf, "decision_function"):
            decision = clf.decision_function(X_vec)
            if decision.ndim == 1:
                # Binary case
                probs = softmax(np.vstack([-decision * 4.0, decision * 4.0]).T, axis=1)[0]
            else:
                # Multiclass margin calibration via temperature scaling
                scaled_decision = (decision[0] - np.mean(decision[0])) / (np.std(decision[0]) + 1e-6)
                probs = softmax(scaled_decision * 2.0)
        elif hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_vec)[0]
        else:
            pred_idx = clf.predict(X_vec)[0]
            probs = np.zeros(len(self.class_names))
            probs[pred_idx] = 1.0

        latency_ms = (time.perf_counter() - t0) * 1000.0
        best_idx = int(np.argmax(probs))
        best_category = self.class_names[best_idx]
        best_score = float(probs[best_idx])
        
        # Top 3 predictions
        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [
            {"category": self.class_names[i], "confidence": round(float(probs[i]), 4)}
            for i in top3_indices
        ]

        return {
            "model_name": "Linear SVM (TF-IDF)",
            "model_type": "Classic Machine Learning",
            "category": best_category,
            "confidence": round(best_score, 4),
            "confidence_percentage": f"{best_score * 100:.1f}%",
            "top_candidates": top3,
            "latency_ms": round(latency_ms, 2),
        }

    def _predict_advanced(self, clean_text: str) -> Dict[str, Any]:
        """Compute prediction & confidence using Sentence-BERT embeddings + classifier."""
        t0 = time.perf_counter()
        
        # Generate dense embedding
        emb = self.sbert_embedder.encode([clean_text], normalize_embeddings=True, show_progress_bar=False)
        
        # Predict with downstream classifier
        probs = self.sbert_classifier.predict_proba(emb)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        best_idx = int(np.argmax(probs))
        best_category = self.class_names[best_idx]
        best_score = float(probs[best_idx])
        
        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [
            {"category": self.class_names[i], "confidence": round(float(probs[i]), 4)}
            for i in top3_indices
        ]

        return {
            "model_name": "Sentence-BERT (MiniLM-L6-v2) + LR",
            "model_type": "Transformer & Dense Embeddings",
            "category": best_category,
            "confidence": round(best_score, 4),
            "confidence_percentage": f"{best_score * 100:.1f}%",
            "top_candidates": top3,
            "latency_ms": round(latency_ms, 2),
        }

    def predict(
        self,
        file_source: Any = None,
        filename: str = "",
        raw_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform full dual prediction pipeline on resume input."""
        raw_extracted, clean_text = self.extract_and_clean_text(
            file_source=file_source,
            filename=filename,
            raw_text=raw_text,
        )

        keywords = self.cleaner.extract_keywords(clean_text, top_k=12)
        classic_result = self._predict_classic(clean_text)
        advanced_result = self._predict_advanced(clean_text)
        is_agreement = (classic_result["category"] == advanced_result["category"])

        return {
            "status": "success",
            "filename": filename or "Saisie directe",
            "text_length_chars": len(raw_extracted),
            "word_count_raw": len(raw_extracted.split()),
            "word_count_clean": len(clean_text.split()),
            "keywords": keywords,
            "models_agreement": is_agreement,
            "classic_model": classic_result,
            "advanced_model": advanced_result,
            "text_preview": raw_extracted[:350] + ("..." if len(raw_extracted) > 350 else ""),
        }


predictor_service = PredictionService()
