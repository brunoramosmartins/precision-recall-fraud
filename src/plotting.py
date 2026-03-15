"""Figure generation utilities for publication-quality output.

Provides standardized plotting functions with consistent styling, color palettes,
and typography for model evaluation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.metrics import sweep_thresholds


MODEL_COLORS = {
    "logistic_regression": "#1f4e79",
    "random_forest": "#b85450",
    "weak_classifier": "#6d9eeb",
    "dummy": "#808080",
    "lr (balanced)": "#1f4e79",
    "lr (c=0.01)": "#76a5af",
    "lr (c=10)": "#0b5394",
    "rf (depth=5)": "#d6b656",
    "rf (depth=15)": "#b85450",
}

MODEL_LINESTYLES = {
    "logistic_regression": "-",
    "random_forest": "--",
    "weak_classifier": "-.",
    "dummy": ":",
}

_FALLBACK_PALETTE = [
    "#1f4e79", "#b85450", "#6ca870", "#d6b656",
    "#8e7cc3", "#e06666", "#76a5af", "#999999"
]
_FALLBACK_LINESTYLES = ["-", "--", "-.", ":"]


def apply_style() -> None:
    """Configures global Matplotlib parameters for consistency and readability."""
    plt.style.use("seaborn-v0_8-white")

    plt.rcParams.update({
        "figure.figsize": (9, 5.5),
        "figure.dpi": 600,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "normal",
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "axes.axisbelow": True,
        "text.usetex": False,
    })


def _clean_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Removes upper and right spines and applies subtle grid formatting.

    Args:
        ax: The Matplotlib axes object to modify.
        grid_axis: The axis to apply the grid ('x', 'y', or 'both').
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.grid(True, axis=grid_axis)


def _model_style(name: str, index: int = 0) -> tuple[str, str]:
    """Retrieves standard color and linestyle for a given model.

    Args:
        name: The model identifier string.
        index: Fallback index if the model name is not explicitly defined.

    Returns:
        A tuple containing the hex color string and matplotlib linestyle string.
    """
    name_lower = name.lower()
    color = MODEL_COLORS.get(name_lower, _FALLBACK_PALETTE[index % len(_FALLBACK_PALETTE)])
    linestyle = MODEL_LINESTYLES.get(name_lower, _FALLBACK_LINESTYLES[index % len(_FALLBACK_LINESTYLES)])
    return color, linestyle


def save_figure(fig: plt.Figure, name: str, config: dict) -> Path:
    """Exports the figure to the designated output directory.

    Args:
        fig: The Matplotlib figure to save.
        name: The target filename without extension.
        config: Project configuration dictionary containing figure parameters.

    Returns:
        The absolute Path object where the figure was saved.
    """
    from src.data import find_project_root
    
    out_dir = find_project_root() / config["figures"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = config["figures"]["format"]
    path = out_dir / f"{name}.{fmt}"
    
    fig.savefig(path, bbox_inches="tight", transparent=False, facecolor="white")
    print(f"  Saved: {path}")
    return path


def plot_metric_bars(
    models: dict[str, dict],
    metrics: list[str],
    config: dict,
    title: str = "Metric Comparison Across Models",
    subtitle: str = "",
) -> plt.Figure:
    """Generates a grouped bar chart comparing multiple evaluation metrics."""
    apply_style()
    n_models = len(models)
    n_metrics = len(metrics)
    
    x = np.arange(n_metrics)
    width = 0.6 / n_models

    fig, ax = plt.subplots(figsize=(max(9, n_metrics * 1.5), 5.5))

    for i, (name, vals) in enumerate(models.items()):
        color, _ = _model_style(name, i)
        offset = (i - n_models / 2 + 0.5) * width
        bar_vals = [vals.get(m, 0.0) for m in metrics]
        
        bars = ax.bar(
            x + offset, bar_vals, width, 
            label=name, color=color, 
            edgecolor="#333333", linewidth=0.8
        )

        for bar, v in zip(bars, bar_vals):
            if v > 0.001:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8, color="#333333"
                )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper().replace("_", "-") for m in metrics], rotation=0)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title, pad=15)
    _clean_axes(ax, grid_axis="y")
    
    return fig


def plot_roc_curves(
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    config: dict,
    title: str = "ROC Curves",
    subtitle: str = "Threshold-free ranking quality",
) -> plt.Figure:
    """Generates ROC curves for comparative model performance."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for i, (name, (y_true, y_scores)) in enumerate(models.items()):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = auc(fpr, tpr)
        color, linestyle = _model_style(name, i)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=color, linestyle=linestyle)

    ax.plot([0, 1], [0, 1], color="#999999", lw=1.2, linestyle=":", label="Random baseline")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title, pad=15)
    _clean_axes(ax, grid_axis="both")

    ax.legend(loc="lower right")
    
    return fig


def plot_pr_curves(
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    base_rate: float,
    config: dict,
    recall_targets: list[float] | None = None,
    title: str = "Precision–Recall Curves",
    subtitle: str = "Model separation in PR space",
) -> plt.Figure:
    """Generates Precision-Recall curves indicating imbalanced classification capability."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for i, (name, (y_true, y_scores)) in enumerate(models.items()):
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        auc_val = auc(rec, prec)
        color, linestyle = _model_style(name, i)
        ax.plot(rec, prec, label=f"{name} (AUC-PR = {auc_val:.3f})", color=color, linestyle=linestyle)

    ax.axhline(base_rate, color="#999999", linestyle=":", lw=1.2, label=f"Random baseline (≈ {base_rate:.4f})")

    if recall_targets:
        for r in recall_targets:
            ax.axvline(r, color="#cccccc", linestyle="-", lw=1.0)
            ax.text(r + 0.015, base_rate + 0.02, f"r={r}", fontsize=8, color="#666666")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title, pad=15)
    _clean_axes(ax, grid_axis="both")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    
    return fig


def plot_threshold_sweep(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    config: dict,
    threshold_cost_optimal: float | None = None,
    title: str = "Precision, Recall and F1 vs Decision Threshold",
    subtitle: str = "",
) -> plt.Figure:
    """Generates a sweep chart tracking Precision, Recall, and F1 across thresholds."""
    apply_style()
    thr_cfg = config["thresholds"]
    thresholds = np.linspace(thr_cfg["range_start"], thr_cfg["range_end"], thr_cfg["n_steps"])
    results = sweep_thresholds(y_true, y_scores, thresholds)

    precs = np.array([r["precision"] for r in results])
    recs  = np.array([r["recall"]    for r in results])
    f1s   = np.array([r["f1"]        for r in results])
    f1_opt_thr = float(thresholds[np.argmax(f1s)])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(thresholds, precs, label="Precision", color="#1f4e79", linestyle="-")
    ax.plot(thresholds, recs,  label="Recall",    color="#b85450", linestyle="--")
    ax.plot(thresholds, f1s,   label="F1",        color="#6ca870", linestyle="-.")

    _vline_and_label(ax, 0.5, "#999999", "Default\nτ=0.50", y_pos=0.9)
    _vline_and_label(ax, f1_opt_thr, "#6ca870", f"F1-opt\nτ={f1_opt_thr:.3f}", y_pos=0.75)
    
    if threshold_cost_optimal is not None:
        _vline_and_label(ax, threshold_cost_optimal, "#b85450", f"Cost-opt\nτ*={threshold_cost_optimal:.3f}", y_pos=0.6)

    ax.set_xlabel(r"Decision Threshold ($\tau$)")
    ax.set_ylabel("Score")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title, pad=15)
    _clean_axes(ax, grid_axis="both")

    ax.legend(loc="center right")
    
    return fig


def _vline_and_label(ax: plt.Axes, x: float, color: str, label: str, y_pos: float) -> None:
    """Draws a vertical line with an associated bounding box label."""
    ax.axvline(x, color=color, linestyle=":", lw=1.2)
    ax.text(
        x + 0.015, y_pos, label, color=color, fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=color, alpha=0.9)
    )


def plot_confusion_matrices(
    threshold_results: list[dict],
    threshold_labels: list[str],
    config: dict,
    suptitle: str = "Confusion Matrices at Three Decision Thresholds",
) -> plt.Figure:
    """Renders a comparative grid of confusion matrices across thresholds."""
    apply_style()
    n = len(threshold_results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    
    if n == 1: 
        axes = [axes]

    for ax, m, label in zip(axes, threshold_results, threshold_labels):
        _draw_confusion_matrix(ax, m, label)

    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.05)
    
    return fig


def _draw_confusion_matrix(ax: plt.Axes, m: dict, title: str) -> None:
    """Renders a single normalized confusion matrix subplot."""
    matrix = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]])
    row_totals = matrix.sum(axis=1, keepdims=True)
    normed = matrix / np.where(row_totals > 0, row_totals, 1)

    cell_labels = [["True Positive", "False Negative"], ["False Positive", "True Negative"]]
    row_labels = ["Actual: Fraud", "Actual: Legit"]
    col_labels = ["Predicted: Fraud", "Predicted: Legit"]

    cmap = plt.get_cmap("Blues")

    for i in range(2):
        for j in range(2):
            intensity = normed[i, j]
            bg_color = cmap(0.1 + 0.8 * intensity)
            text_color = "white" if intensity > 0.5 else "#1a1a1a"

            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, transform=ax.transData, color=bg_color, ec="#333333", lw=1.0))
            pct = normed[i, j] * 100
            ax.text(
                j + 0.5, 1.5 - i, f"{cell_labels[i][j]}\n{matrix[i, j]:,}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=9, color=text_color
            )

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9, rotation=90, va="center")
    ax.xaxis.tick_top()
    
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"{title}\nPrecision = {m['precision']:.3f} | Recall = {m['recall']:.3f}", fontsize=9, pad=20)


def plot_precision_at_recall_bar(
    recall_targets: list[float],
    results: list[dict],
    config: dict,
    title: str = "Precision@Recall — Business Trade-off",
    subtitle: str = "Higher recall requirement → lower precision → more false alarms",
) -> plt.Figure:
    """Generates dual bar charts contrasting Precision vs. Error counts per recall target."""
    apply_style()
    labels = [f"r={r}" for r in recall_targets]
    precs = [r["precision"] for r in results]
    fp_counts = [r["fp"] for r in results]
    fn_counts = [r["fn"] for r in results]
    
    x = np.arange(len(recall_targets))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    bars = ax1.bar(x, precs, color="#1f4e79", edgecolor="#333333", linewidth=0.8, width=0.5)
    for bar, v in zip(bars, precs):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision at Each Recall Target", pad=10)
    ax1.set_ylim(0, min(1.1, max(precs) * 1.25 if precs else 1.0))
    _clean_axes(ax1, grid_axis="y")

    ax2.bar(x - width / 2, fp_counts, width, label="False Positives", color="#b85450", edgecolor="#333333", linewidth=0.8)
    ax2.bar(x + width / 2, fn_counts, width, label="False Negatives", color="#d6b656", edgecolor="#333333", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Count")
    ax2.set_title("FP and FN Counts at Each Recall Target", pad=10)
    
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    _clean_axes(ax2, grid_axis="y")

    full_title = title if not subtitle else f"{title}\n{subtitle}"
    fig.suptitle(full_title, fontsize=12, fontweight="bold", y=1.05)
    
    return fig


def plot_ranking_comparison(
    model_names: list[str],
    metrics_dict: dict[str, list[float]],
    config: dict,
    title: str = "Model Ranking by Metric",
) -> plt.Figure:
    """Generates a heatmap comparing model ranks across distinct evaluation metrics."""
    apply_style()
    metric_names = list(metrics_dict.keys())
    scores = np.array([metrics_dict[m] for m in metric_names]).T

    vmin, vmax = min(float(scores.min()), 0.5), 1.0

    fig, ax = plt.subplots(figsize=(max(9, len(metric_names) * 1.5), max(4, len(model_names) * 0.8)))
    im = ax.imshow(scores, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in metric_names], fontsize=9)
    ax.set_yticklabels(model_names, fontsize=9)

    for i in range(len(model_names)):
        for j in range(len(metric_names)):
            v = scores[i, j]
            norm_v = (v - vmin) / max(vmax - vmin, 1e-6)
            text_color = "white" if norm_v > 0.60 else "#1a1a1a"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5, color=text_color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
    cbar.outline.set_visible(False)
    
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, pad=15)
    
    return fig