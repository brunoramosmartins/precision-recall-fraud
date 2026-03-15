"""Model training utilities.

All hyperparameters are read from config.yaml. Each function returns a fitted
sklearn Pipeline (StandardScaler + classifier) with a predict_proba method,
ready to produce fraud probability scores.
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _make_pipeline(clf) -> Pipeline:
    """Wrap a classifier in a StandardScaler pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------

def train_dummy(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Pipeline:
    """Train a majority-class dummy classifier.

    Always predicts the most frequent class (legitimate).
    Used in Experiment A to demonstrate the accuracy trap:
    high accuracy despite zero fraud detection ability.
    """
    pipeline = _make_pipeline(DummyClassifier(strategy="most_frequent"))
    pipeline.fit(X_train, y_train)
    return pipeline


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
) -> Pipeline:
    """Train a logistic regression with parameters from config["models"]["logistic_regression"]."""
    cfg = config["models"]["logistic_regression"]
    clf = LogisticRegression(
        C=cfg["C"],
        max_iter=cfg["max_iter"],
        solver=cfg["solver"],
        class_weight=cfg["class_weight"],
        random_state=config["random_seed"],
    )
    pipeline = _make_pipeline(clf)
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
) -> Pipeline:
    """Train a random forest with parameters from config["models"]["random_forest"]."""
    cfg = config["models"]["random_forest"]
    clf = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        class_weight=cfg["class_weight"],
        random_state=config["random_seed"],
        n_jobs=-1,
    )
    pipeline = _make_pipeline(clf)
    pipeline.fit(X_train, y_train)
    return pipeline


def train_weak_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
) -> Pipeline:
    """Train a deliberately weak decision tree (max_depth=2).

    Used in Experiment B to show that model quality differences are visible
    in PR space but compressed in ROC space under class imbalance.
    """
    cfg = config["models"]["weak_classifier"]
    clf = DecisionTreeClassifier(
        max_depth=cfg["max_depth"],
        class_weight=cfg["class_weight"],
        random_state=config["random_seed"],
    )
    pipeline = _make_pipeline(clf)
    pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

def get_scores(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Extract P(fraud | x) scores from a fitted pipeline.

    Returns the positive-class (fraud) probability from predict_proba,
    or falls back to predict() for classifiers without probability output.
    """
    if hasattr(pipeline["clf"], "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.predict(X).astype(float)
