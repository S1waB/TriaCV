"""Sprint 2 – Orchestrator: training + evaluation in one script."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # TraiCV/
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.classic.train_baseline import run_training
from src.evaluation.evaluate_classic   import evaluate_all, print_comparison_table


def main():
    # Phase 1 – training
    payload = run_training()

    # Phase 2 – evaluation
    comparison_df = evaluate_all(
        results     = payload["results"],
        y_test      = payload["y_test"],
        class_names = payload["class_names"],
        best_name   = payload["best_name"],
    )

    # Final display
    print_comparison_table(comparison_df)

    best = comparison_df[comparison_df["Selected Baseline"] == "YES"].iloc[0]
    print(f"\n  [SELECTED BASELINE] : {best['Model']}")
    print(f"  - Accuracy          : {best['Accuracy']}")
    print(f"  - F1-weighted       : {best['F1 Weighted']}")
    print(f"  - F1-macro          : {best['F1 Macro']}")
    print(f"  - CV F1 Mean        : {best['CV F1 Mean']} +/- {best['CV F1 Std']}")
    print(f"  - Inference Latency : {best['Infer Latency (ms)']} ms")
    print(f"  - Best Params       : {best['Best Params']}")
    print("\n  Sprint 2 completed successfully.")


if __name__ == "__main__":
    main()
