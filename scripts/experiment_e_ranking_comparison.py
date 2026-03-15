"""Experiment E — F1 vs Precision@75Recall: Model Ranking Can Differ.

Theoretical claim being tested:
    Ranking models by F1 can give a different answer than ranking by
    Precision@75recall. This is the core empirical result of the article:
    metric choice changes which model you deploy.

Expected outcome:
    At least one model pair where F1 and Precision@75recall give opposite rankings.
    The heatmap makes the divergence points explicit.

Output:
    figures/exp_e_ranking_comparison.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import load_config, load_dataset, get_features_and_target
from src.data import describe_dataset, make_train_test_split
from src.models import get_scores
from src.metrics import precision_at_recall, compute_metrics_at_threshold, sweep_thresholds
from src.plotting import plot_ranking_comparison, save_figure


def _train_variant(clf, X_train, y_train) -> Pipeline:
    """Fit a StandardScaler + classifier pipeline."""
    p = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    p.fit(X_train, y_train)
    return p


def _best_f1(y_true, y_scores, n_steps=99) -> float:
    """Return the best F1 achievable across a threshold grid."""
    thresholds = np.linspace(0.01, 0.99, n_steps)
    results = sweep_thresholds(y_true, y_scores, thresholds)
    return max(r["f1"] for r in results)


def run(config: dict) -> None:
    print("\n=== Experiment E: F1 vs Precision@75Recall — Model Ranking ===")

    # --- Data ---
    df = load_dataset(config, source="synthetic")
    X, y = get_features_and_target(df)
    describe_dataset(y, label="synthetic")

    X_train, X_test, y_train, y_test = make_train_test_split(X, y, config)
    seed = config["random_seed"]

    # --- Model variants ---
    # Five variants with intentionally different precision/recall trade-offs
    variants = {
        "LR (balanced)": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000,
            solver="lbfgs", random_state=seed,
        ),
        "LR (C=0.01)": LogisticRegression(
            C=0.01, class_weight="balanced", max_iter=1000,
            solver="lbfgs", random_state=seed,
        ),
        "LR (C=10)": LogisticRegression(
            C=10.0, class_weight="balanced", max_iter=1000,
            solver="lbfgs", random_state=seed,
        ),
        "RF (depth=5)": RandomForestClassifier(
            n_estimators=100, max_depth=5, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
        "RF (depth=15)": RandomForestClassifier(
            n_estimators=100, max_depth=15, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
    }

    print("\nTraining model variants...")
    fitted = {
        name: _train_variant(clf, X_train, y_train)
        for name, clf in variants.items()
    }

    # --- Compute metrics for each variant ---
    recall_targets = config["recall_targets"]
    metric_keys = ["F1 (best)", "AUC-PR"] + [f"P@{int(r*100)}R" for r in recall_targets]

    scores_per_model: dict[str, list[float]] = {k: [] for k in metric_keys}
    model_names = list(fitted.keys())

    print(f"\n{'Model':<20}", end="")
    for k in metric_keys:
        print(f"{k:>12}", end="")
    print()
    print("-" * (20 + 12 * len(metric_keys)))

    for name, pipeline in fitted.items():
        y_scores = get_scores(pipeline, X_test)

        f1_best = _best_f1(y_test, y_scores)
        auc_pr = average_precision_score(y_test, y_scores)

        p_at_r = []
        for r in recall_targets:
            prec, _ = precision_at_recall(y_test, y_scores, recall_target=r)
            p_at_r.append(prec)

        row = [f1_best, auc_pr] + p_at_r
        for k, v in zip(metric_keys, row):
            scores_per_model[k].append(v)

        print(f"{name:<20}", end="")
        for v in row:
            print(f"{v:>12.4f}", end="")
        print()

    # --- Identify ranking divergences ---
    print("\nRanking by F1 (best):")
    f1_ranks = np.argsort(scores_per_model["F1 (best)"])[::-1]
    for rank, idx in enumerate(f1_ranks):
        print(f"  {rank + 1}. {model_names[idx]}")

    print("\nRanking by P@75R:")
    p75_key = f"P@{int(recall_targets[0]*100)}R"
    p75_ranks = np.argsort(scores_per_model[p75_key])[::-1]
    for rank, idx in enumerate(p75_ranks):
        print(f"  {rank + 1}. {model_names[idx]}")

    top_f1 = model_names[f1_ranks[0]]
    top_p75 = model_names[p75_ranks[0]]
    if top_f1 != top_p75:
        print(f"\nRanking divergence confirmed:")
        print(f"  Best by F1       : {top_f1}")
        print(f"  Best by P@75R    : {top_p75}")
        print("  These are DIFFERENT models — metric choice changes the deployment decision.")
    else:
        print(f"\nTop-1 model agrees ({top_f1}), check mid-table for divergences.")

    # --- Figure ---
    fig = plot_ranking_comparison(
        model_names,
        scores_per_model,
        config,
        title="Model Ranking by Metric — Where F1 and Precision@Recall Diverge",
    )
    save_figure(fig, "exp_e_ranking_comparison", config)


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
