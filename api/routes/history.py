"""Session history route handler."""
from flask import Blueprint, jsonify
from api.services.history_store import history_store

history_bp = Blueprint("history", __name__)


@history_bp.route("/api/history", methods=["GET"])
def get_history():
    """Return in-memory session history."""
    entries = history_store.get_all()
    return jsonify({
        "status": "success",
        "count": len(entries),
        "history": entries,
    }), 200


@history_bp.route("/api/history", methods=["DELETE"])
def clear_history():
    """Clear session history."""
    history_store.clear()
    return jsonify({
        "status": "success",
        "message": "Historique de session effacé avec succès.",
    }), 200
