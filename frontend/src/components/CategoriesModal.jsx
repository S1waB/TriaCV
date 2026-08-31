import React from "react";
import { Grid, X, Check } from "lucide-react";

export default function CategoriesModal({ isOpen, onClose, categories }) {
  if (!isOpen) return null;

  return (
    <div className="drawer-overlay" onClick={onClose} style={{ alignItems: "center", justifyContent: "center" }}>
      <div
        className="card"
        onClick={(e) => e.stopPropagation()}
        style={{ width: "680px", maxWidth: "92vw", maxHeight: "85vh", display: "flex", flexDirection: "column" }}
      >
        <div className="card-header">
          <h2 className="card-title">
            <Grid size={20} color="#6366f1" />
            25 Catégories Professionnelles Supportées
          </h2>
          <button type="button" className="btn btn-outline" style={{ padding: "6px" }} onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <p style={{ fontSize: "0.85rem", color: "#94a3b8", marginBottom: "16px" }}>
          TriaCV est entraîné sur 25 métiers spécialisés pour classifier et trier instantanément les candidatures.
        </p>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
            gap: "10px",
            overflowY: "auto",
            paddingRight: "6px",
          }}
        >
          {categories.map((cat, idx) => (
            <div
              key={idx}
              style={{
                background: "rgba(255, 255, 255, 0.03)",
                border: "1px solid var(--border-color)",
                padding: "10px 14px",
                borderRadius: "var(--radius-sm)",
                fontSize: "0.84rem",
                fontWeight: 600,
                color: "#e2e8f0",
                display: "flex",
                alignItems: "center",
                gap: "8px",
              }}
            >
              <div
                style={{
                  width: "6px",
                  height: "6px",
                  borderRadius: "50%",
                  background: "#6366f1",
                }}
              />
              {cat}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
