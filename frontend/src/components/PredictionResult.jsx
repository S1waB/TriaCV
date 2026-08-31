import React from "react";
import { CheckCircle2, AlertCircle, Zap, Cpu, Award, Tag, Clock, FileText } from "lucide-react";

export default function PredictionResult({ result }) {
  if (!result) {
    return (
      <div className="card empty-state">
        <FileText className="empty-icon" />
        <h3 style={{ fontSize: "1.1rem", marginBottom: "8px", color: "#94a3b8" }}>
          Aucun résultat pour le moment
        </h3>
        <p style={{ fontSize: "0.85rem", maxWidth: "340px", margin: "0 auto" }}>
          Importez un CV au format PDF, DOCX ou collez du texte brut pour obtenir la classification duale.
        </p>
      </div>
    );
  }

  const {
    filename,
    models_agreement,
    classic_model,
    advanced_model,
    keywords,
    word_count_clean,
    text_preview,
  } = result;

  return (
    <div className="card result-box">
      <div className="card-header">
        <div>
          <div style={{ fontSize: "0.78rem", color: "#64748b", fontWeight: 600 }}>RÉSULTAT D'ANALYSE</div>
          <h2 className="card-title" style={{ marginTop: "2px" }}>
            {filename}
          </h2>
        </div>

        <div>
          {models_agreement ? (
            <span className="agreement-badge perfect">
              <CheckCircle2 size={14} /> Accord Parfait
            </span>
          ) : (
            <span className="agreement-badge divergence">
              <AlertCircle size={14} /> Divergence Modèles
            </span>
          )}
        </div>
      </div>

      {/* Main Dual Comparison Cards */}
      <div className="models-comparison-grid">
        {/* Classic Model */}
        <div className="model-card highlight">
          <div className="model-badge">Baseline Retenue</div>
          <h3 className="model-name" style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <Zap size={16} color="#f59e0b" />
            {classic_model.model_name}
          </h3>
          <div className="predicted-category">{classic_model.category}</div>

          <div className="confidence-bar-wrapper">
            <div className="confidence-meta">
              <span>Confiance</span>
              <span>{classic_model.confidence_percentage}</span>
            </div>
            <div className="confidence-bar-bg">
              <div
                className="confidence-bar-fill"
                style={{
                  width: classic_model.confidence_percentage,
                  background: "linear-gradient(90deg, #f59e0b, #eab308)",
                }}
              />
            </div>
          </div>

          <div className="latency-tag">
            <Clock size={12} />
            Latence : <strong>{classic_model.latency_ms} ms</strong>
          </div>

          {/* Top candidates */}
          <div style={{ marginTop: "14px", paddingTop: "10px", borderTop: "1px solid rgba(255,255,255,0.06)", fontSize: "0.76rem" }}>
            <div style={{ color: "#64748b", marginBottom: "4px" }}>Alternatives :</div>
            {classic_model.top_candidates?.slice(1, 3).map((alt, idx) => (
              <div key={idx} style={{ display: "flex", justifyContent: "space-between", color: "#94a3b8" }}>
                <span>{alt.category}</span>
                <span>{(alt.confidence * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Advanced Model */}
        <div className="model-card">
          <div className="model-badge" style={{ color: "#c084fc" }}>Approche Dense / NLP</div>
          <h3 className="model-name" style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <Cpu size={16} color="#c084fc" />
            {advanced_model.model_name}
          </h3>
          <div className="predicted-category" style={{ background: "linear-gradient(90deg, #c084fc, #e879f9)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            {advanced_model.category}
          </div>

          <div className="confidence-bar-wrapper">
            <div className="confidence-meta">
              <span>Confiance</span>
              <span>{advanced_model.confidence_percentage}</span>
            </div>
            <div className="confidence-bar-bg">
              <div
                className="confidence-bar-fill"
                style={{
                  width: advanced_model.confidence_percentage,
                  background: "linear-gradient(90deg, #a855f7, #ec4899)",
                }}
              />
            </div>
          </div>

          <div className="latency-tag">
            <Clock size={12} />
            Latence : <strong>{advanced_model.latency_ms} ms</strong>
          </div>

          {/* Top candidates */}
          <div style={{ marginTop: "14px", paddingTop: "10px", borderTop: "1px solid rgba(255,255,255,0.06)", fontSize: "0.76rem" }}>
            <div style={{ color: "#64748b", marginBottom: "4px" }}>Alternatives :</div>
            {advanced_model.top_candidates?.slice(1, 3).map((alt, idx) => (
              <div key={idx} style={{ display: "flex", justifyContent: "space-between", color: "#94a3b8" }}>
                <span>{alt.category}</span>
                <span>{(alt.confidence * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Extracted Skills / Keywords */}
      {keywords && keywords.length > 0 && (
        <div className="keywords-wrapper">
          <div className="keywords-title">
            <Tag size={13} style={{ display: "inline", verticalAlign: "middle", marginRight: "4px" }} />
            Compétences & Mots-clés Clés Détectés ({keywords.length})
          </div>
          <div className="keywords-list">
            {keywords.map((kw, i) => (
              <span key={i} className="keyword-tag">
                {kw}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Preview Snippet */}
      {text_preview && (
        <div style={{ marginTop: "18px", fontSize: "0.8rem", color: "#64748b" }}>
          <strong>Extrait nettoyé ({word_count_clean} mots) :</strong>
          <div style={{ marginTop: "4px", fontStyle: "italic", lineHeight: "1.4" }}>
            "{text_preview}"
          </div>
        </div>
      )}
    </div>
  );
}
