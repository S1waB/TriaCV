import React from "react";
import { History, X, Trash2, CheckCircle, AlertCircle, FileText } from "lucide-react";

export default function HistoryDrawer({ isOpen, onClose, history, onClear }) {
  if (!isOpen) return null;

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <div className="drawer-content" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
          <h2 style={{ fontSize: "1.2rem", fontWeight: 700, display: "flex", alignItems: "center", gap: "8px" }}>
            <History size={20} color="#6366f1" />
            Historique de Session
          </h2>
          <div style={{ display: "flex", gap: "8px" }}>
            {history.length > 0 && (
              <button
                type="button"
                className="btn btn-outline"
                style={{ padding: "6px 10px", fontSize: "0.75rem", color: "#ef4444" }}
                onClick={onClear}
                title="Effacer la session"
              >
                <Trash2 size={14} />
                Vider
              </button>
            )}
            <button
              type="button"
              className="btn btn-outline"
              style={{ padding: "6px" }}
              onClick={onClose}
            >
              <X size={18} />
            </button>
          </div>
        </div>

        <p style={{ fontSize: "0.8rem", color: "#64748b", marginBottom: "16px" }}>
          Cet historique est temporaire (en mémoire de session) et s'efface automatiquement à la fermeture.
        </p>

        {history.length === 0 ? (
          <div className="empty-state" style={{ padding: "40px 10px" }}>
            <FileText className="empty-icon" />
            <p style={{ fontSize: "0.9rem" }}>Aucun CV analysé au cours de cette session.</p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {history.map((item) => (
              <div key={item.id} className="history-item">
                <div className="history-header">
                  <span style={{ fontWeight: 600, color: "#cbd5e1" }}>{item.filename}</span>
                  <span>{item.timestamp}</span>
                </div>
                
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "6px" }}>
                  <div>
                    <div style={{ fontSize: "0.75rem", color: "#64748b" }}>Catégorie Prédite :</div>
                    <div className="history-category">{item.classic_category}</div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <span style={{ fontSize: "0.82rem", fontWeight: 700, color: "#10b981" }}>
                      {item.classic_confidence_percentage}
                    </span>
                  </div>
                </div>

                {item.keywords && item.keywords.length > 0 && (
                  <div style={{ display: "flex", gap: "4px", flexWrap: "wrap", marginTop: "8px" }}>
                    {item.keywords.map((kw, i) => (
                      <span key={i} style={{ fontSize: "0.7rem", background: "rgba(255,255,255,0.05)", padding: "2px 6px", borderRadius: "4px", color: "#94a3b8" }}>
                        {kw}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
