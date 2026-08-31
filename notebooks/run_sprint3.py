"""Sprint 3 – Orchestrator: Sentence-BERT, DistilBERT Fine-tuning, & Final Comparison."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.advanced.sentence_bert import train_sentence_bert_pipeline
from src.models.advanced.distilbert_finetune import train_distilbert_pipeline
from src.evaluation.evaluate_advanced import generate_final_comparison, print_final_table


def main():
    print("=" * 80)
    print("  TRIA-CV | SPRINT 3: ADVANCED APPROACHES & TRANSFORMER BENCHMARK")
    print("=" * 80)
    
    # 1. Train Sentence-BERT + Logistic Regression
    res_sbert = train_sentence_bert_pipeline()
    
    # 2. Fine-tune DistilBERT
    res_distilbert = train_distilbert_pipeline()
    
    # 3. Generate Final Comparison Report
    final_df = generate_final_comparison([res_sbert, res_distilbert])
    
    # 4. Print Table
    print_final_table(final_df)
    
    print("\n  Sprint 3 completed successfully.")


if __name__ == "__main__":
    main()
