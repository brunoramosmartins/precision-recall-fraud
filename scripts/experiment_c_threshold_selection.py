"""Experiment C — Threshold Selection and Its Effect on the Confusion Matrix.

Theoretical claim being tested:
    The default threshold of 0.5 is almost never optimal for imbalanced fraud
    detection. The optimal threshold is determined by the cost structure, not by
    the probability midpoint. Using the wrong threshold has measurable business
    consequences visible in the confusion matrix.

Expected outcome:
    - At threshold=0.5: low recall, few FP, many FN — the model misses most fraud.
    - At F1-optimal threshold: better balance between Precision and Recall.
    - At cost-optimal threshold (≈0.024): high recall, more FP, very few FN —
      correct behavior given C_FN >> C_FP.
    - The three-line chart shows F1 peaking at a threshold far below 0.5.

Output:
    figures/exp_c_threshold_sweep.png
    figures/exp_c_confusion_matrices.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data import load_config, load_dataset, get_features_and_target
from src.data import describe_dataset, make_train_test_split
from src.models import train_logistic_regression, get_scores
from src.metrics import (
    cost_optimal_threshold,
    sweep_thresholds,
    compute_metrics_at_threshold,
)
from src.plotting import plot_threshold_sweep, plot_confusion_matrices, save_figure


def run(config: dict) -> None:
    """Executes the threshold selection experiment and generates related figures.

    Args:
        config: Project configuration dictionary mapping hyperparameters and costs.
    """
    print("\n=== Experiment C: Threshold Selection ===")

    df = load_dataset(config, source="synthetic")
    X, y = get_features_and_target(df)
    describe_dataset(y, label="synthetic")

    X_train, X_test, y_train, y_test = make_train_test_split(X, y, config)

    print("\nTraining Logistic Regression...")
    model = train_logistic_regression(X_train, y_train, config)
    y_scores = get_scores(model, X_test)

    tau_default = 0.5
    tau_cost = cost_optimal_threshold(
        config["costs"]["false_positive"],
        config["costs"]["false_negative"],
    )
    
    thr_cfg = config["thresholds"]
    thresholds = np.linspace(thr_cfg["range_start"], thr_cfg["range_end"], thr_cfg["n_steps"])
    results = sweep_thresholds(y_test, y_scores, thresholds)
    f1s = np.array([r["f1"] for r in results])
    tau_f1 = float(thresholds[np.argmax(f1s)])

    print("\nThreshold values:")
    print(f"  Default (τ = 0.5)  : {tau_default:.3f}")
    print(f"  F1-optimal         : {tau_f1:.3f}")
    print(f"  Cost-optimal (τ*)  : {tau_cost:.3f}  "
          f"[C_FP=${config['costs']['false_positive']:.0f}, "
          f"C_FN=${config['costs']['false_negative']:.0f}]")

    m_default = compute_metrics_at_threshold(y_test, y_scores, tau_default)
    m_f1 = compute_metrics_at_threshold(y_test, y_scores, tau_f1)
    m_cost = compute_metrics_at_threshold(y_test, y_scores, tau_cost)

    for label, m in [
        (f"Default (τ={tau_default})", m_default),
        (f"F1-optimal (τ={tau_f1:.3f})", m_f1),
        (f"Cost-optimal (τ*={tau_cost:.3f})", m_cost),
    ]:
        print(f"\n{label}:")
        print(f"  TP={m['tp']}  FN={m['fn']}  FP={m['fp']}  TN={m['tn']}")
        print(f"  Precision={m['precision']:.3f}  Recall={m['recall']:.3f}  F1={m['f1']:.3f}")

    c_fp_str = r"$C_{FP}$"
    c_fn_str = r"$C_{FN}$"
    tau_star_str = r"$\tau^*$"
    tau_str = r"$\tau$"
    
    sweep_subtitle = (
        f"Cost-optimal {tau_star_str} = {tau_cost:.3f}  "
        f"({c_fp_str} = \\${config['costs']['false_positive']:.0f}, "
        f"{c_fn_str} = \\${config['costs']['false_negative']:.0f})"
    )

    fig_sweep = plot_threshold_sweep(
        y_test, y_scores, config,
        threshold_cost_optimal=tau_cost,
        title="Precision, Recall and F1 vs Decision Threshold",
        subtitle=sweep_subtitle,
    )
    save_figure(fig_sweep, "exp_c_threshold_sweep", config)

    cm_suptitle = (
        f"Confusion Matrices at Three Decision Thresholds — "
        f"{c_fp_str} = \\${config['costs']['false_positive']:.0f}, "
        f"{c_fn_str} = \\${config['costs']['false_negative']:.0f}"
    )

    threshold_labels = [
        f"Default  ({tau_str} = {tau_default})",
        f"F1-optimal  ({tau_str} = {tau_f1:.3f})",
        f"Cost-optimal  ({tau_star_str} = {tau_cost:.3f})",
    ]
    
    fig_cm = plot_confusion_matrices(
        threshold_results=[m_default, m_f1, m_cost],
        threshold_labels=threshold_labels,
        config=config,
        suptitle=cm_suptitle,
    )
    save_figure(fig_cm, "exp_c_confusion_matrices", config)


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)