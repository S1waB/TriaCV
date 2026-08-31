"""Categories list route."""
from flask import Blueprint, jsonify
from api.services.predictor import predictor_service

categories_bp = Blueprint("categories", __name__)


@categories_bp.route("/api/categories", methods=["GET"])
def get_categories():
    """Return the list of all supported 25 job categories."""
    categories = predictor_service.get_categories()
    return jsonify({
        "status": "success",
        "total": len(categories),
        "categories": categories,
    }), 200
