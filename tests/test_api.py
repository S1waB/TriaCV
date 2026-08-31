"""Comprehensive PyTest Suite for TriaCV Flask API."""
import io
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from api.app import create_app


@pytest.fixture
def client():
    """Flask test client fixture."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """Test /api/health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "TriaCV" in data["service"]


def test_get_categories(client):
    """Test /api/categories endpoint returns all 25 categories."""
    response = client.get("/api/categories")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["total"] == 25
    assert "Data Science" in data["categories"]
    assert "Python Developer" in data["categories"]
    assert "Java Developer" in data["categories"]


def test_predict_valid_text(client):
    """Test /api/predict with direct text input."""
    cv_text = """
    SUMMARY:
    Senior Python Developer with 6 years experience in Django, Flask, FastAPI, Docker, and PostgreSQL.
    Extensive background in building REST APIs, asynchronous workers with Celery, and machine learning pipelines.
    Core Skills: Python, Django, Flask, Docker, Kubernetes, CI/CD, Git, Linux.
    """
    payload = {"text": cv_text, "filename": "python_cv.txt"}
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "classic_model" in data
    assert "advanced_model" in data
    assert data["classic_model"]["category"] == "Python Developer"
    assert data["advanced_model"]["category"] == "Python Developer"
    assert data["classic_model"]["confidence"] > 0.5
    assert len(data["keywords"]) > 0


def test_predict_valid_file_upload(client):
    """Test /api/predict with a simulated TXT file upload."""
    sample_cv = (
        "Experienced Data Scientist with 5 years experience in machine learning, "
        "scikit-learn, TensorFlow, pandas, numpy, and statistical modeling. "
        "Led data analytics projects and deployed predictive models into production."
    )
    data = {
        "file": (io.BytesIO(sample_cv.encode("utf-8")), "resume_data_science.txt"),
    }
    response = client.post(
        "/api/predict",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    res_json = response.get_json()
    assert res_json["status"] == "success"
    assert res_json["classic_model"]["category"] == "Data Science"
    assert res_json["advanced_model"]["category"] == "Data Science"


def test_predict_invalid_file_format(client):
    """Test /api/predict rejecting unsupported extensions like .exe or .png."""
    data = {
        "file": (io.BytesIO(b"dummy binary data"), "malware.exe"),
    }
    response = client.post(
        "/api/predict",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    res_json = response.get_json()
    assert res_json["status"] == "error"
    assert "non supporté" in res_json["error"]


def test_predict_empty_file(client):
    """Test /api/predict with empty file upload."""
    data = {
        "file": (io.BytesIO(b""), "empty.txt"),
    }
    response = client.post(
        "/api/predict",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    res_json = response.get_json()
    assert res_json["status"] == "error"


def test_predict_empty_text(client):
    """Test /api/predict with blank string."""
    payload = {"text": "   ", "filename": "blank.txt"}
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    res_json = response.get_json()
    assert res_json["status"] == "error"


def test_history_lifecycle(client):
    """Test session history addition, retrieval, and clearing."""
    # 1. Clear history first
    client.delete("/api/history")
    
    # 2. Make a prediction
    cv_text = "Senior DevOps Engineer mastering Kubernetes, Terraform, AWS, Docker, CI/CD pipelines, and Prometheus."
    client.post(
        "/api/predict",
        data=json.dumps({"text": cv_text, "filename": "devops.txt"}),
        content_type="application/json",
    )
    
    # 3. Get history
    hist_resp = client.get("/api/history")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.get_json()
    assert hist_data["count"] >= 1
    assert hist_data["history"][0]["classic_category"] == "DevOps Engineer"
    
    # 4. Clear history
    del_resp = client.delete("/api/history")
    assert del_resp.status_code == 200
    
    # 5. Verify cleared
    hist_after = client.get("/api/history").get_json()
    assert hist_after["count"] == 0
