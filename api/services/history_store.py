"""Session history in-memory store for TriaCV predictions."""
from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List


class HistoryStore:
    """Thread-safe in-memory session history store."""

    def __init__(self, max_items: int = 100):
        self.max_items = max_items
        self._history: List[Dict[str, Any]] = []

    def add(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Add a prediction result to history and return the stored entry."""
        entry_id = str(uuid.uuid4())[:8]
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        entry = {
            "id": entry_id,
            "timestamp": timestamp,
            "filename": prediction_result.get("filename", "CV sans titre"),
            "classic_category": prediction_result["classic_model"]["category"],
            "classic_confidence": prediction_result["classic_model"]["confidence"],
            "classic_confidence_percentage": prediction_result["classic_model"]["confidence_percentage"],
            "advanced_category": prediction_result["advanced_model"]["category"],
            "advanced_confidence": prediction_result["advanced_model"]["confidence"],
            "advanced_confidence_percentage": prediction_result["advanced_model"]["confidence_percentage"],
            "agreement": prediction_result.get("models_agreement", True),
            "keywords": prediction_result.get("keywords", [])[:5],
            "word_count": prediction_result.get("word_count_raw", 0),
        }
        
        self._history.insert(0, entry)
        if len(self._history) > self.max_items:
            self._history.pop()
            
        return entry

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all history entries."""
        return list(self._history)

    def clear(self) -> None:
        """Clear the history."""
        self._history.clear()


# Global singleton
history_store = HistoryStore()
