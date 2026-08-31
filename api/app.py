"""TriaCV Flask Application Factory & Server."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS

# Project root resolution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.routes.predict import predict_bp
from api.routes.categories import categories_bp
from api.routes.history import history_bp


def create_app() -> Flask:
    """Initialize and configure Flask application."""
    app = Flask(__name__)
    
    # Configure CORS for local development and frontend integrations
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register blueprints
    app.register_blueprint(predict_bp)
    app.register_blueprint(categories_bp)
    app.register_blueprint(history_bp)
    
    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({
            "status": "healthy",
            "service": "TriaCV Classification API",
            "version": "1.0.0",
        }), 200

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "status": "error",
            "error": "Endpoint introuvable.",
        }), 404

    @app.errorhandler(500)
    def server_error(error):
        return jsonify({
            "status": "error",
            "error": "Erreur serveur interne.",
        }), 500

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting TriaCV API server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
