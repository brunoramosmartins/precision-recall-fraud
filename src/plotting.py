"""Figure generation utilities for publication-quality output.

Provides standardized plotting functions with consistent styling, color palettes,
and typography for model evaluation metrics. All visual parameters are read from
config.yaml to ensure a single source of truth.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.metrics import sweep_thresholds


# ── Linestyle cycling (not color-dependent) ──────────────────────────────────
_LINESTYLES = ["-", "--", "-.", ":"]

# ── Marker cycling for print/B&W distinction ────────────────────────────────
_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def _get_model_colors(config: dict) -> dict[str, str]:
    """Returns model→color mapping from config, with fallback palette."""
    cfg_colors = config.get("figures", {}).get("model_colors", {})
    return {k.lower(): v for k, v in cfg_colors.items()}


def _get_palette(config: dict) -> list[str]:
    """Returns the ordered fallback palette from config (Wong 2011)."""
    cfg = config.get("figures", {}).get("colors", {})
    if cfg:
        return list(cfg.values())
    return ["#0072B2", "#E69F00", "#009E73", "#D55E00",
            "#CC79A7", "#F0E442", "#56B4E9", "#000000"]


def apply_style(config: dict) -> None:
    """Configures global Matplotlib parameters from config.yaml."""
    fig_cfg = config.get("figures", {})
    dpi = fig_cfg.get("dpi", 300)
    figsize = tuple(fig_cfg.get("figsize", [10, 6]))
    style = fig_cfg.get("style", "seaborn-v0_8-whitegrid")

    plt.style.use(style)

    plt.rcParams.update({
        "figure.figsize": figsize,
        "figure.dpi": dpi,
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
    """Removes upper and right spines and applies subtle grid formatting."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.grid(True, axis=grid_axis)


def _model_style(name: str, index: int, config: dict) -> tuple[str, str]:
    """Retrieves color and linestyle for a model, using config palette."""
    name_lower = name.lower()
    model_colors = _get_model_colors(config)
    palette = _get_palette(config)

    color = model_colors.get(name_lower, palette[index % len(palette)])
    linestyle = _LINESTYLES[index % len(_LINESTYLES)]
    return color, linestyle


def save_figure(fig: plt.Figure, name: str, config: dict) -> Path:
    """Exports the figure to the designated output directory."""
    from src.data import find_project_root

    out_dir = find_project_root() / config["figures"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = config["figures"]["format"]
    dpi = config["figures"].get("dpi", 300)
    path = out_dir / f"{name}.{fmt}"

    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                transparent=False, facecolor="white")
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Experiment A — Metric Bars
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_bars(
    models: dict[str, dict],
    metrics: list[str],
    config: dict,
    title: str = "Metric Comparison Across Models",
    subtitle: str = "",
) -> plt.Figure:
    """Generates a grouped bar chart comparing multiple evaluation metrics."""
    apply_style(config)
    n_models = len(models)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.6 / n_models

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.5), 6))

    for i, (name, vals) in enumerate(models.items()):
        color, _ = _model_style(name, i, config)
        offset = (i - n_models / 2 + 0.5) * width
        bar_vals = [vals.get(m, 0.0) for m in metrics]

        bars = ax.bar(
            x + offset, bar_vals, width,
            label=name, color=color,
            edgecolor="#333333", linewidth=0.8
        )

        for bar, v in zip(bars, bar_vals):
            label_y = bar.get_height() + 0.015
            label_text = f"{v:.3f}"
            if v < 0.001:
                # Explicitly show zero/near-zero values with emphasis
                label_text = "0.000"
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 0.03,
                    label_text, ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=color, alpha=0.9, lw=0.8),
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, label_y,
                    label_text, ha="center", va="bottom",
                    fontsize=8, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper().replace("_", "-") for m in metrics],
                       rotation=0)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title, pad=15)
    _clean_axes(ax, grid_axis="y")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Experiment B — ROC and PR Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    config: dict,
    title: str = "ROC Curves",
    subtitle: str = "",
) -> plt.Figure:
    """Generates ROC curves for comparative model performance."""
    apply_style(config)
    fig, ax = plt.subplots(figsize=(7, 7))

    for i, (name, (y_true, y_scores)) in enumerate(models.items()):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = auc(fpr, tpr)
        color, linestyle = _model_style(name, i, config)
        marker = _MARKERS[i % len(_MARKERS)]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})",
                color=color, linestyle=linestyle,
                marker=marker, markevery=0.1, markersize=5)

    ax.plot([0, 1], [0, 1], color="#999999", lw=1.2, linestyle=":",
            label="Random baseline")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect("equal")

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
    title: str = "Precision\u2013Recall Curves",
    subtitle: str = "",
) -> plt.Figure:
    """Generates Precision-Recall curves with recall target annotations."""
    apply_style(config)
    fig, ax = plt.subplots(figsize=(7, 7))

    for i, (name, (y_true, y_scores)) in enumerate(models.items()):
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        auc_val = auc(rec, prec)
        color, linestyle = _model_style(name, i, config)
        marker = _MARKERS[i % len(_MARKERS)]
        ax.plot(rec, prec, label=f"{name} (AUC-PR = {auc_val:.3f})",
                color=color, linestyle=linestyle,
                marker=marker, markevery=0.1, markersize=5)

    # Random baseline with annotated arrow
    ax.axhline(base_rate, color="#999999", linestyle=":", lw=1.2,
               label=f"Random baseline (\u2248 {base_rate:.4f})")
    ax.annotate(
        f"baseline = {base_rate:.4f}",
        xy=(0.5, base_rate), xytext=(0.35, 0.12),
        fontsize=8, color="#666666",
        arrowprops=dict(arrowstyle="->", color="#999999", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                  alpha=0.9),
    )

    # Recall targets — staggered labels to avoid overlap
    if recall_targets:
        y_positions = np.linspace(0.20, 0.08, len(recall_targets))
        for r, y_pos in zip(recall_targets, y_positions):
            ax.axvline(r, color="#cccccc", linestyle="-", lw=0.8, alpha=0.7)
            ax.text(r, y_pos, f"r={r}", fontsize=7.5, color="#666666",
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="#cccccc", alpha=0.85, lw=0.5))

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title, pad=15)
    _clean_axes(ax, grid_axis="both")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Experiment C — Threshold Sweep and Confusion Matrices
# ─────────────────────────────────────────────────────────────────────────────

def plot_threshold_sweep(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    config: dict,
    threshold_cost_optimal: float | None = None,
    title: str = "Precision, Recall and F1 vs Decision Threshold",
    subtitle: str = "",
) -> plt.Figure:
    """Generates a sweep chart tracking Precision, Recall, and F1 across thresholds."""
    apply_style(config)
    palette = _get_palette(config)
    thr_cfg = config["thresholds"]
    thresholds = np.linspace(thr_cfg["range_start"], thr_cfg["range_end"],
                             thr_cfg["n_steps"])
    results = sweep_thresholds(y_true, y_scores, thresholds)

    precs = np.array([r["precision"] for r in results])
    recs = np.array([r["recall"] for r in results])
    f1s = np.array([r["f1"] for r in results])
    f1_opt_thr = float(thresholds[np.argmax(f1s)])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precs, label="Precision",
            color=palette[0], linestyle="-")
    ax.plot(thresholds, recs, label="Recall",
            color=palette[3], linestyle="--")
    ax.plot(thresholds, f1s, label="F1",
            color=palette[2], linestyle="-.")

    _vline_and_label(ax, 0.5, "#999999",
                     "Default\n\u03c4=0.50", y_pos=0.5)
    _vline_and_label(ax, f1_opt_thr, palette[2],
                     f"F1-opt\n\u03c4={f1_opt_thr:.3f}", y_pos=0.75)

    if threshold_cost_optimal is not None:
        _vline_and_label(ax, threshold_cost_optimal, palette[3],
                         f"Cost-opt\n\u03c4*={threshold_cost_optimal:.3f}",
                         y_pos=0.6)

    ax.set_xlabel(r"Decision Threshold ($\tau$)")
    ax.set_ylabel("Score")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title, pad=15)
    _clean_axes(ax, grid_axis="both")

    ax.legend(loc="center right")

    return fig


def _vline_and_label(ax: plt.Axes, x: float, color: str,
                     label: str, y_pos: float) -> None:
    """Draws a vertical line with an associated bounding box label."""
    ax.axvline(x, color=color, linestyle=":", lw=1.2)
    ax.text(
        x + 0.015, y_pos, label, color=color, fontsize=8,
        va="top", ha="left",
        bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=color, alpha=0.9),
    )


def plot_confusion_matrices(
    threshold_results: list[dict],
    threshold_labels: list[str],
    config: dict,
    suptitle: str = "Confusion Matrices at Three Decision Thresholds",
) -> plt.Figure:
    """Renders a comparative grid of confusion matrices across thresholds."""
    apply_style(config)
    n = len(threshold_results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))

    if n == 1:
        axes = [axes]

    for ax, m, label in zip(axes, threshold_results, threshold_labels):
        _draw_confusion_matrix(ax, m, label)

    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.08)

    return fig


def _draw_confusion_matrix(ax: plt.Axes, m: dict, title: str) -> None:
    """Renders a single normalized confusion matrix subplot.

    Layout follows sklearn convention: Positive (Fraud) on top row.
    """
    # Row 0 = Fraud (Positive), Row 1 = Legit (Negative)
    matrix = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]])
    row_totals = matrix.sum(axis=1, keepdims=True)
    normed = matrix / np.where(row_totals > 0, row_totals, 1)

    cell_labels = [["True Positive", "False Negative"],
                   ["False Positive", "True Negative"]]
    row_labels = ["Actual: Fraud", "Actual: Legit"]
    col_labels = ["Predicted: Fraud", "Predicted: Legit"]

    cmap = plt.get_cmap("Blues")

    for i in range(2):
        for j in range(2):
            intensity = normed[i, j]
            bg_color = cmap(0.1 + 0.8 * intensity)
            text_color = "white" if intensity > 0.5 else "#1a1a1a"

            ax.add_patch(plt.Rectangle(
                (j, 1 - i), 1, 1, transform=ax.transData,
                color=bg_color, ec="#333333", lw=1.0))
            pct = normed[i, j] * 100
            ax.text(
                j + 0.5, 1.5 - i,
                f"{cell_labels[i][j]}\n{matrix[i, j]:,}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=9, color=text_color,
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

    ax.set_title(
        f"{title}\nPrecision = {m['precision']:.3f} | "
        f"Recall = {m['recall']:.3f}",
        fontsize=9, pad=20,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Experiment D — Precision@Recall Bar
# ─────────────────────────────────────────────────────────────────────────────

def plot_precision_at_recall_bar(
    recall_targets: list[float],
    results: list[dict],
    config: dict,
    title: str = "Precision@Recall",
    subtitle: str = "",
) -> plt.Figure:
    """Generates dual bar charts: Precision + FP/FN counts with cost annotations."""
    apply_style(config)
    palette = _get_palette(config)
    labels = [f"r={r}" for r in recall_targets]
    precs = [r["precision"] for r in results]
    fp_counts = [r["fp"] for r in results]
    fn_counts = [r["fn"] for r in results]

    c_fp = config["costs"]["false_positive"]
    c_fn = config["costs"]["false_negative"]

    x = np.arange(len(recall_targets))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Left panel: Precision bars
    bars = ax1.bar(x, precs, color=palette[0], edgecolor="#333333",
                   linewidth=0.8, width=0.5)
    for bar, v in zip(bars, precs):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision at Each Recall Target", pad=10)
    ax1.set_ylim(0, min(1.15, max(precs) * 1.25 if precs else 1.0))
    _clean_axes(ax1, grid_axis="y")

    # Right panel: FP/FN counts with cost annotations
    bars_fp = ax2.bar(x - width / 2, fp_counts, width,
                      label="False Positives", color=palette[3],
                      edgecolor="#333333", linewidth=0.8)
    bars_fn = ax2.bar(x + width / 2, fn_counts, width,
                      label="False Negatives", color=palette[1],
                      edgecolor="#333333", linewidth=0.8)

    # Add cost annotation above each group
    global_max = max(max(fp_counts), max(fn_counts))
    for xi, fp, fn in zip(x, fp_counts, fn_counts):
        total_cost = fp * c_fp + fn * c_fn
        max_bar = max(fp, fn)
        ax2.text(xi, max_bar + global_max * 0.05,
                 f"${total_cost:,.0f}",
                 ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                 color="#333333",
                 bbox=dict(boxstyle="round,pad=0.15", fc="#f5f5f5",
                           ec="#cccccc", alpha=0.9))

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Count")
    ax2.set_title("FP and FN Counts at Each Recall Target\n"
                  f"(labels = expected cost at $C_{{FP}}$=${c_fp:.0f}, "
                  f"$C_{{FN}}$=${c_fn:.0f})", fontsize=10, pad=10)
    # Extra headroom for cost labels
    ax2.set_ylim(0, global_max * 1.25)

    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    _clean_axes(ax2, grid_axis="y")

    full_title = title if not subtitle else f"{title}\n{subtitle}"
    fig.suptitle(full_title, fontsize=12, fontweight="bold", y=1.05)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Experiment E — Ranking Comparison Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_ranking_comparison(
    model_names: list[str],
    metrics_dict: dict[str, list[float]],
    config: dict,
    title: str = "Model Ranking by Metric",
) -> plt.Figure:
    """Generates a heatmap comparing model scores with ranking divergence highlights."""
    apply_style(config)
    metric_names = list(metrics_dict.keys())
    scores = np.array([metrics_dict[m] for m in metric_names]).T

    vmin = float(scores.min()) - 0.02
    vmax = 1.0

    fig, ax = plt.subplots(
        figsize=(max(10, len(metric_names) * 1.6),
                 max(4, len(model_names) * 0.9)))
    im = ax.imshow(scores, aspect="auto", cmap="Blues",
                   vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in metric_names],
                       fontsize=9)
    ax.set_yticklabels(model_names, fontsize=9)

    # Compute per-column rankings (1-based, lower=better)
    rankings = np.zeros_like(scores, dtype=int)
    for j in range(scores.shape[1]):
        order = np.argsort(-scores[:, j])  # descending
        for rank, idx in enumerate(order):
            rankings[idx, j] = rank + 1

    # Identify top-1 per column
    top1_per_metric = scores.argmax(axis=0)

    # Check for ranking divergence: does F1 top-1 differ from any P@R top-1?
    f1_col = 0  # "F1 (best)" is the first column
    f1_top = top1_per_metric[f1_col]
    divergent_cells = set()
    for j in range(len(metric_names)):
        if top1_per_metric[j] != f1_top:
            divergent_cells.add((top1_per_metric[j], j))
            divergent_cells.add((f1_top, j))

    for i in range(len(model_names)):
        for j in range(len(metric_names)):
            v = scores[i, j]
            rank = rankings[i, j]
            norm_v = (v - vmin) / max(vmax - vmin, 1e-6)
            text_color = "white" if norm_v > 0.60 else "#1a1a1a"

            # Main value
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8.5, color=text_color)

            # Rank subscript (top-right of cell)
            rank_color = "#FFD700" if rank == 1 else text_color
            ax.text(j + 0.38, i + 0.32, f"#{rank}",
                    ha="right", va="bottom", fontsize=6,
                    color=rank_color, fontweight="bold" if rank == 1 else "normal")

            # Highlight divergent cells with border
            if (i, j) in divergent_cells:
                rect = mpatches.FancyBboxPatch(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    boxstyle="round,pad=0.02",
                    linewidth=2.5, edgecolor="#D55E00",
                    facecolor="none", zorder=5)
                ax.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
    cbar.outline.set_visible(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, pad=15)

    return fig
