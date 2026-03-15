"""Experiment D — Precision@Recall as an Operating Point.

Theoretical claim being tested:
    Precision@75recall and Precision@85recall represent meaningfully different
    business decisions with different confusion matrix profiles. Higher recall
    requirements force precision down and FP counts up. The article reader must
    decide what is acceptable for their business.

Expected outcome:
    - As the recall target increases, precision decreases and FP count increases.
    - The FN count decreases as recall increases (fewer missed frauds).
    - The bar chart makes the trade-off explicit in business terms.

Output:
    figures/exp_d_precision_at_recall.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_config, load_dataset, get_features_and_target
from src.data import describe_dataset, make_train_test_split
from src.models import train_random_forest, get_scores
from src.metrics import precision_at_recall, compute_metrics_at_threshold
from src.plotting import plot_precision_at_recall_bar, save_figure


def run(config: dict, model=None) -> None:
    """Run Experiment D.

    Parameters
    ----------
    config :
        Project configuration dict.
    model :
        Optional pre-fitted model from Experiment B. If None, trains a new
        Random Forest on the synthetic dataset.
    """
    print("\n=== Experiment D: Precision@Recall as an Operating Point ===")

    # --- Data ---
    df = load_dataset(config, source="synthetic")
    X, y = get_features_and_target(df)
    describe_dataset(y, label="synthetic")

    X_train, X_test, y_train, y_test = make_train_test_split(X, y, config)

    # --- Model ---
    if model is None:
        print("\nTraining Random Forest (best model from Experiment B)...")
        model = train_random_forest(X_train, y_train, config)

    y_scores = get_scores(model, X_test)

    # --- Precision@Recall at each target ---
    recall_targets = config["recall_targets"]
    print(f"\nRecall targets: {recall_targets}")
    print(f"\n{'Recall target':>14} {'Threshold':>10} {'Precision':>10} {'FP':>8} {'FN':>8}")
    print("-" * 55)

    results = []
    for r in recall_targets:
        prec, thr = precision_at_recall(y_test, y_scores, recall_target=r)
        m = compute_metrics_at_threshold(y_test, y_scores, threshold=thr)
        results.append(m)
        print(f"{r:>14.2f} {thr:>10.4f} {prec:>10.4f} {m['fp']:>8,} {m['fn']:>8,}")

    # --- Business cost at each operating point ---
    c_fp = config["costs"]["false_positive"]
    c_fn = config["costs"]["false_negative"]
    print(f"\nExpected cost at each operating point (C_FP=${c_fp}, C_FN=${c_fn}):")
    print(f"{'Recall target':>14} {'FP cost':>12} {'FN cost':>12} {'Total cost':>12}")
    print("-" * 52)
    for r, m in zip(recall_targets, results):
        cost_fp_total = m["fp"] * c_fp
        cost_fn_total = m["fn"] * c_fn
        total = cost_fp_total + cost_fn_total
        print(f"{r:>14.2f} {cost_fp_total:>12,.0f} {cost_fn_total:>12,.0f} {total:>12,.0f}")

    # --- Figure ---
    fig = plot_precision_at_recall_bar(
        recall_targets,
        results,
        config,
        title="Precision@Recall — Business Trade-off",
        subtitle="Higher recall requirement → lower precision → more false alarms",
    )
    save_figure(fig, "exp_d_precision_at_recall", config)


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
