"""Unit tests for trained models and vectorizer inference."""
import os
import joblib
import pytest
from src.preprocessing.cleaner import ResumeCleaner


@pytest.fixture
def artifacts():
    assert os.path.exists("models_artifacts/classic_best_model.pkl"), "Model artifact missing"
    assert os.path.exists("models_artifacts/tfidf_vectorizer.pkl"), "Vectorizer artifact missing"
    assert os.path.exists("models_artifacts/label_encoder.pkl"), "LabelEncoder artifact missing"

    model = joblib.load("models_artifacts/classic_best_model.pkl")
    vectorizer = joblib.load("models_artifacts/tfidf_vectorizer.pkl")
    encoder = joblib.load("models_artifacts/label_encoder.pkl")
    return model, vectorizer, encoder


def test_classic_model_prediction_data_science(artifacts):
    model, vectorizer, encoder = artifacts
    cleaner = ResumeCleaner()
    cv_text = """
    Experienced Data Scientist with 5 years in Python, machine learning, deep learning,
    PyTorch, TensorFlow, scikit-learn, and NLP predictive modeling. Built neural networks and XGBoost models.
    """
    cleaned = cleaner.clean_text(cv_text)
    vec = vectorizer.transform([cleaned])
    pred_idx = model.predict(vec)[0]
    category = encoder.inverse_transform([pred_idx])[0]
    assert category == "Data Science"


def test_classic_model_prediction_java_developer(artifacts):
    model, vectorizer, encoder = artifacts
    cleaner = ResumeCleaner()
    cv_text = """
    Senior Java Developer specialized in Spring Boot, Microservices, Hibernate, REST APIs,
    Maven, Docker, and Kafka architecture with relational database SQL expertise.
    """
    cleaned = cleaner.clean_text(cv_text)
    vec = vectorizer.transform([cleaned])
    pred_idx = model.predict(vec)[0]
    category = encoder.inverse_transform([pred_idx])[0]
    assert category == "Java Developer"


def test_classic_model_probabilities(artifacts):
    model, vectorizer, encoder = artifacts
    cleaner = ResumeCleaner()
    cv_text = "Recruitment specialist and HR manager handling onboarding, talent acquisition and employee relations."
    cleaned = cleaner.clean_text(cv_text)
    vec = vectorizer.transform([cleaned])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        assert len(probs) == len(encoder.classes_)
        assert probs.sum() == pytest.approx(1.0, rel=1e-3)
