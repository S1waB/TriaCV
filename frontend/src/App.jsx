import React, { useState, useEffect } from "react";
import { FileText, History, Grid, AlertTriangle, Sparkles, RefreshCw, BarChart2 } from "lucide-react";
import UploadForm from "./components/UploadForm";
import PredictionResult from "./components/PredictionResult";
import HistoryDrawer from "./components/HistoryDrawer";
import CategoriesModal from "./components/CategoriesModal";

export default function App() {
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [categories, setCategories] = useState([]);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [isCategoriesOpen, setIsCategoriesOpen] = useState(false);

  // Fetch initial categories and history
  useEffect(() => {
    fetchCategories();
    fetchHistory();
  }, []);

  const fetchCategories = async () => {
    try {
      const res = await fetch("/api/categories");
      if (res.ok) {
        const data = await res.json();
        setCategories(data.categories || []);
      }
    } catch (err) {
      console.warn("Could not fetch categories:", err);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch("/api/history");
      if (res.ok) {
        const data = await res.json();
        setHistory(data.history || []);
      }
    } catch (err) {
      console.warn("Could not fetch history:", err);
    }
  };

  const handlePredict = async ({ type, data }) => {
    setIsLoading(true);
    setError(null);

    try {
      let res;
      if (type === "file") {
        res = await fetch("/api/predict", {
          method: "POST",
          body: data,
        });
      } else {
        res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        });
      }

      const json = await res.json();

      if (!res.ok) {
        throw new Error(json.error || "Erreur lors de la classification du CV.");
      }

      setPrediction(json);
      // Refresh history
      fetchHistory();
    } catch (err) {
      setError(err.message || "Impossible de contacter l'API de prédiction.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearHistory = async () => {
    try {
      await fetch("/api/history", { method: "DELETE" });
      setHistory([]);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="logo-wrapper">
          <div className="logo-icon">
            <FileText size={24} color="#ffffff" />
          </div>
          <div>
            <h1 className="logo-title">TriaCV</h1>
            <p className="logo-tagline">Classification & Analyse Duale de CVs (Classic SVM + Sentence-BERT)</p>
          </div>
        </div>

        <div className="header-actions">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => setIsCategoriesOpen(true)}
            title="Consulter les 25 catégories"
          >
            <Grid size={16} />
            <span style={{ display: "inline" }}>25 Catégories</span>
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => setIsHistoryOpen(true)}
            title="Historique temporaire de session"
          >
            <History size={16} />
            <span style={{ display: "inline" }}>Session ({history.length})</span>
          </button>
        </div>
      </header>

      {/* Global Alert */}
      {error && (
        <div className="alert alert-danger">
          <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: "2px" }} />
          <div style={{ flex: 1 }}>{error}</div>
          <button
            type="button"
            onClick={() => setError(null)}
            style={{ background: "none", border: "none", color: "#fca5a5", cursor: "pointer" }}
          >
            &times;
          </button>
        </div>
      )}

      {/* Main Two-Column Layout */}
      <main className="main-grid">
        <section>
          <UploadForm onPredict={handlePredict} isLoading={isLoading} />
        </section>

        <section>
          <PredictionResult result={prediction} />
        </section>
      </main>

      {/* Modals & Drawers */}
      <HistoryDrawer
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        history={history}
        onClear={handleClearHistory}
      />

      <CategoriesModal
        isOpen={isCategoriesOpen}
        onClose={() => setIsCategoriesOpen(false)}
        categories={categories}
      />

      {/* Footer */}
      <footer className="app-footer">
        <p>
          <strong>TriaCV</strong> — Pipeline NLP de Traitement & Classification de CVs · Baseline Linear SVM (0.9 ms) & Sentence-BERT
        </p>
      </footer>
    </div>
  );
}
