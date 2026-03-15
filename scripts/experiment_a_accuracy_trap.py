"""Experiment A — The Accuracy Trap.

Theoretical claim being tested:
    Accuracy is misleading for imbalanced classification. A naive majority-class
    classifier can achieve >99% accuracy on a 0.1% fraud dataset while having
    zero fraud detection ability. AUC-PR correctly scores it near the base rate.

Expected outcome:
    - DummyClassifier accuracy > Logistic Regression accuracy (or very close).
    - DummyClassifier AUC-PR ≈ base rate; Logistic Regression AUC-PR >> base rate.
    - The bar chart makes the misleading nature of accuracy visually obvious.

Output:
    figures/exp_a_accuracy_trap.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_config, load_dataset, get_features_and_target
from src.data import describe_dataset, make_train_test_split
from src.models import train_dummy, train_logistic_regression, get_scores
from src.metrics import compute_all_metrics
from src.plotting import plot_metric_bars, save_figure


def run(config: dict) -> None:
    print("\n=== Experiment A: The Accuracy Trap ===")

    # --- Data ---
    df = load_dataset(config, source="synthetic")
    X, y = get_features_and_target(df)
    stats = describe_dataset(y, label="synthetic")
    base_rate = stats["fraud_rate"]

    X_train, X_test, y_train, y_test = make_train_test_split(X, y, config)

    # --- Models ---
    print("\nTraining models...")
    dummy = train_dummy(X_train, y_train)
    lr = train_logistic_regression(X_train, y_train, config)

    scores_dummy = get_scores(dummy, X_test)
    scores_lr = get_scores(lr, X_test)

    # --- Metrics ---
    m_dummy = compute_all_metrics(y_test, scores_dummy, threshold=0.5)
    m_lr = compute_all_metrics(y_test, scores_lr, threshold=0.5)

    metric_keys = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"]

    print("\nResults at threshold = 0.5:")
    print(f"{'Metric':<12} {'Dummy':>10} {'Logistic Reg':>14}")
    print("-" * 40)
    for k in metric_keys:
        print(f"{k:<12} {m_dummy.get(k, float('nan')):>10.4f} {m_lr.get(k, float('nan')):>14.4f}")

    print(f"\nRandom AUC-PR baseline (= fraud rate): {base_rate:.4f}")
    print(f"  Dummy AUC-PR: {m_dummy['auc_pr']:.4f}  (expected ≈ {base_rate:.4f})")
    print(f"  Logistic Regression AUC-PR: {m_lr['auc_pr']:.4f}")

    # --- Figure ---
    models = {
        "Dummy Classifier": m_dummy,
        "Logistic Regression": m_lr,
    }
    fig = plot_metric_bars(
        models,
        metric_keys,
        config,
        title="The Accuracy Trap",
        subtitle="Accuracy is misleading; AUC-PR reveals the dummy classifier's true performance",
    )
    save_figure(fig, "exp_a_accuracy_trap", config)


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
