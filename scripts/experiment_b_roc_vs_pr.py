"""Experiment B — The PR Curve and the ROC Curve Side by Side.

Theoretical claim being tested:
    PR curves reveal discrimination differences that ROC curves hide under
    class imbalance. A strong, a moderate, and a weak model look more similar
    in ROC space than they do in PR space.

Expected outcome:
    - AUC-ROC values are close across models; AUC-PR differences are larger.
    - The weak classifier's PR curve drops sharply near the random baseline.
    - The random baseline is nearly invisible in the ROC plot but prominent
      in the PR plot (because the baseline = base rate ≈ 0).

Output:
    figures/exp_b_roc_curves.png
    figures/exp_b_pr_curves.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_config, load_dataset, get_features_and_target
from src.data import describe_dataset, make_train_test_split
from src.models import (
    train_logistic_regression,
    train_random_forest,
    train_weak_classifier,
    get_scores,
)
from src.plotting import plot_roc_curves, plot_pr_curves, save_figure


def run(config: dict) -> dict:
    """Return the best-model key for downstream use by Experiment D."""
    print("\n=== Experiment B: ROC vs PR Curve Side by Side ===")

    # --- Data ---
    df = load_dataset(config, source="synthetic")
    X, y = get_features_and_target(df)
    stats = describe_dataset(y, label="synthetic")
    base_rate = stats["fraud_rate"]

    X_train, X_test, y_train, y_test = make_train_test_split(X, y, config)

    # --- Models ---
    print("\nTraining models...")
    models_fitted = {
        "logistic_regression": train_logistic_regression(X_train, y_train, config),
        "random_forest": train_random_forest(X_train, y_train, config),
        "weak_classifier": train_weak_classifier(X_train, y_train, config),
    }
    model_labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "weak_classifier": "Weak Classifier (depth=2)",
    }

    # Build (y_true, y_scores) pairs keyed by display name
    model_data = {
        model_labels[k]: (y_test, get_scores(v, X_test))
        for k, v in models_fitted.items()
    }

    # --- Figures ---
    print("\nGenerating figures...")
    n_samples = len(y_test) + len(y_train)
    fig_roc = plot_roc_curves(
        model_data,
        config,
        title="ROC Curves",
        subtitle=f"Three classifiers on synthetic fraud data (n={n_samples:,}, fraud rate = {base_rate:.1%})",
    )
    save_figure(fig_roc, "exp_b_roc_curves", config)

    fig_pr = plot_pr_curves(
        model_data,
        base_rate,
        config,
        recall_targets=config["recall_targets"],
        title="Precision\u2013Recall Curves",
        subtitle="Same classifiers evaluated in Precision\u2013Recall space",
    )
    save_figure(fig_pr, "exp_b_pr_curves", config)

    # Identify best model (highest AUC-PR) for Experiment D
    from sklearn.metrics import average_precision_score
    auc_pr_scores = {
        k: average_precision_score(y_test, get_scores(v, X_test))
        for k, v in models_fitted.items()
    }
    best_key = max(auc_pr_scores, key=auc_pr_scores.__getitem__)
    print(f"\nAUC-PR by model:")
    for k, v in auc_pr_scores.items():
        marker = " <-- best" if k == best_key else ""
        print(f"  {model_labels[k]:<30} {v:.4f}{marker}")

    return {"best_model_key": best_key, "models_fitted": models_fitted}


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
