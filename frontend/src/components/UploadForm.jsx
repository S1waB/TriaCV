import React, { useState, useRef } from "react";
import { UploadCloud, FileText, Type, X, ArrowRight, Sparkles, Loader2 } from "lucide-react";

export default function UploadForm({ onPredict, isLoading }) {
  const [activeTab, setActiveTab] = useState("file"); // "file" | "text"
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [rawText, setRawText] = useState("");
  const fileInputRef = useRef(null);

  const sampleCV = `SUMMARY:
Senior Data Scientist with 6 years of experience in NLP, Computer Vision, and Predictive Analytics.
Proven track record in deploying PyTorch, TensorFlow, and scikit-learn models to production microservices.

TECHNICAL SKILLS:
- Languages & Frameworks: Python, SQL, R, PyTorch, Transformers, scikit-learn, pandas, numpy
- Big Data & Cloud: AWS SageMaker, Docker, Kubernetes, Spark, MLflow, CI/CD
- Methods: LLM Fine-tuning, BERT, Embeddings, Time-Series Forecasting, Statistical Modeling`;

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (activeTab === "file") {
      if (!file) return;
      const formData = new FormData();
      formData.append("file", file);
      onPredict({ type: "file", data: formData });
    } else {
      if (!rawText.trim()) return;
      onPredict({ type: "text", data: { text: rawText, filename: "Saisie directe" } });
    }
  };

  const loadSample = () => {
    setActiveTab("text");
    setRawText(sampleCV);
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">
          <UploadCloud size={22} color="#6366f1" />
          Analyser un CV
        </h2>
        <button type="button" className="btn btn-outline" style={{ fontSize: "0.78rem", padding: "6px 12px" }} onClick={loadSample}>
          <Sparkles size={14} />
          Exemple Data Science
        </button>
      </div>

      <div className="tabs-nav">
        <button
          type="button"
          className={`tab-btn ${activeTab === "file" ? "active" : ""}`}
          onClick={() => setActiveTab("file")}
        >
          <FileText size={16} />
          Fichier (PDF, DOCX, TXT)
        </button>
        <button
          type="button"
          className={`tab-btn ${activeTab === "text" ? "active" : ""}`}
          onClick={() => setActiveTab("text")}
        >
          <Type size={16} />
          Texte brut
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        {activeTab === "file" ? (
          <div>
            <div
              className={`dropzone ${dragOver ? "dragover" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.doc,.txt,.rtf,.md"
                style={{ display: "none" }}
                onChange={handleFileChange}
              />
              <div className="dropzone-icon-box">
                <UploadCloud size={28} />
              </div>
              <p className="dropzone-text">Glissez-déposez votre CV ici ou parcourez vos fichiers</p>
              <p className="dropzone-hint">Formats acceptés : PDF, DOCX, TXT (Max 15 Mo)</p>
            </div>

            {file && (
              <div className="file-pill">
                <div className="file-info">
                  <FileText size={18} color="#6366f1" />
                  <div>
                    <div style={{ color: "#f8fafc" }}>{file.name}</div>
                    <div style={{ fontSize: "0.75rem", color: "#64748b" }}>
                      {(file.size / 1024).toFixed(1)} Ko
                    </div>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); setFile(null); }}
                  style={{ background: "none", border: "none", color: "#94a3b8", cursor: "pointer" }}
                >
                  <X size={18} />
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="textarea-wrapper">
            <textarea
              className="cv-textarea"
              placeholder="Collez ici le contenu texte brut du CV à classifier..."
              value={rawText}
              onChange={(e) => setRawText(e.target.value)}
            />
          </div>
        )}

        <div style={{ marginTop: "20px" }}>
          <button
            type="submit"
            className="btn btn-primary"
            style={{ width: "100%", justifyContent: "center", padding: "14px" }}
            disabled={isLoading || (activeTab === "file" ? !file : !rawText.trim())}
          >
            {isLoading ? (
              <>
                <Loader2 size={18} className="animate-spin" />
                Traitement et classification en cours...
              </>
            ) : (
              <>
                Lancer l'analyse du CV
                <ArrowRight size={18} />
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
