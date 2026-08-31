"""Prediction API route handler."""
from __future__ import annotations

import os
from flask import Blueprint, jsonify, request
from api.services.predictor import predictor_service
from api.services.history_store import history_store

predict_bp = Blueprint("predict", __name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".rtf", ".md"}


def is_allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


@predict_bp.route("/api/predict", methods=["POST"])
def predict():
    """Predict category for uploaded CV or direct text input.
    
    Accepts:
    - multipart/form-data: file=<file>
    - multipart/form-data or application/json: text="<raw_text>"
    """
    try:
        filename = ""
        file_bytes = None
        raw_text = None

        # Check if file uploaded
        if "file" in request.files:
            file_obj = request.files["file"]
            if file_obj.filename == "":
                return jsonify({
                    "status": "error",
                    "error": "Aucun fichier sélectionné.",
                }), 400
                
            filename = file_obj.filename
            if not is_allowed_file(filename):
                return jsonify({
                    "status": "error",
                    "error": f"Format de fichier '{filename}' non supporté. Formats acceptés : PDF, DOCX, TXT.",
                }), 400
                
            file_bytes = file_obj.read()
            if not file_bytes or len(file_bytes) == 0:
                return jsonify({
                    "status": "error",
                    "error": "Le fichier téléchargé est vide.",
                }), 400

        # Check for direct text in form or JSON body
        elif request.is_json and request.json and "text" in request.json:
            raw_text = request.json.get("text", "")
            filename = request.json.get("filename", "Saisie directe")
        elif "text" in request.form:
            raw_text = request.form.get("text", "")
            filename = request.form.get("filename", "Saisie directe")
        else:
            return jsonify({
                "status": "error",
                "error": "Veuillez fournir un fichier (PDF, DOCX, TXT) ou un texte sous le champ 'text'.",
            }), 400

        # Execute prediction
        result = predictor_service.predict(
            file_source=file_bytes,
            filename=filename,
            raw_text=raw_text,
        )

        # Record in session history
        history_store.add(result)

        return jsonify(result), 200

    except ValueError as val_err:
        return jsonify({
            "status": "error",
            "error": str(val_err),
        }), 400
    except Exception as exc:
        return jsonify({
            "status": "error",
            "error": f"Erreur interne lors du traitement : {str(exc)}",
        }), 500
