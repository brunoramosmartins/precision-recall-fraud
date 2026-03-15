"""Data loading, preprocessing, and dataset utilities.

All parameters are read from config.yaml. No hardcoded paths or seeds.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """Find the project root by walking up from this file until config.yaml is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "config.yaml").exists():
            return parent
    return Path.cwd()


def load_config(config_path: Path | str | None = None) -> dict:
    """Load project configuration from config.yaml."""
    if config_path is None:
        config_path = find_project_root() / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset generation and loading
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(config: dict) -> pd.DataFrame:
    """Generate a synthetic fraud dataset using sklearn's make_classification.

    Parameters are read from config["dataset"]["synthetic"]. The class
    imbalance ratio mirrors real-world fraud detection conditions (default
    0.1% fraud rate).
    """
    syn = config["dataset"]["synthetic"]
    seed = config["random_seed"]
    n_redundant = max(0, syn["n_features"] - syn["n_informative"] - 2)

    X, y = make_classification(
        n_samples=syn["n_samples"],
        n_features=syn["n_features"],
        n_informative=syn["n_informative"],
        n_redundant=n_redundant,
        n_clusters_per_class=1,
        weights=[1.0 - syn["fraud_rate"], syn["fraud_rate"]],
        flip_y=0,
        random_state=seed,
    )

    cols = [f"V{i + 1}" for i in range(syn["n_features"])]
    df = pd.DataFrame(X, columns=cols)
    df["Class"] = y
    return df


def load_kaggle_dataset(config: dict) -> pd.DataFrame:
    """Load the Kaggle Credit Card Fraud Detection dataset.

    Expects the CSV at config["dataset"]["kaggle"]["raw_path"].
    See data/README.md for download instructions.

    Raises
    ------
    FileNotFoundError
        If the dataset file is not found at the configured path.
    """
    path = find_project_root() / config["dataset"]["kaggle"]["raw_path"]
    if not path.exists():
        raise FileNotFoundError(
            f"Kaggle dataset not found at {path}.\n"
            "See data/README.md for download instructions."
        )
    return pd.read_csv(path)


def load_dataset(config: dict, source: str | None = None) -> pd.DataFrame:
    """Load dataset based on the source parameter.

    Parameters
    ----------
    config :
        Project configuration dict (from load_config).
    source :
        "kaggle" | "synthetic" | None. If None, uses config["dataset"]["source"].

    Returns
    -------
    DataFrame with feature columns and a "Class" column (1 = fraud).
    """
    source = source or config["dataset"]["source"]
    if source == "kaggle":
        return load_kaggle_dataset(config)
    if source == "synthetic":
        return generate_synthetic_dataset(config)
    raise ValueError(
        f"Unknown dataset source: {source!r}. Expected 'kaggle' or 'synthetic'."
    )


def get_features_and_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target vector y from a loaded DataFrame.

    Assumes the target column is "Class" (1 = fraud, 0 = legitimate).
    """
    target_col = "Class"
    feature_cols = [c for c in df.columns if c != target_col]
    return df[feature_cols].values, df[target_col].values


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline() -> Pipeline:
    """Return a preprocessing pipeline with StandardScaler.

    All transformations are inside the Pipeline to prevent data leakage
    during cross-validation: fit only on training folds, transform test folds.
    """
    return Pipeline([("scaler", StandardScaler())])


# ---------------------------------------------------------------------------
# Train/test split and cross-validation
# ---------------------------------------------------------------------------

def make_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split preserving the fraud class distribution."""
    return train_test_split(
        X,
        y,
        test_size=config["dataset"]["test_size"],
        random_state=config["random_seed"],
        stratify=y,
    )


def make_cv_splitter(config: dict) -> StratifiedKFold:
    """Return a StratifiedKFold splitter configured from config."""
    return StratifiedKFold(
        n_splits=config["dataset"]["cv_folds"],
        shuffle=True,
        random_state=config["random_seed"],
    )


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def describe_dataset(y: np.ndarray, label: str = "") -> dict:
    """Print and return class balance statistics.

    Returns
    -------
    dict with keys: label, n_samples, n_fraud, n_legitimate, fraud_rate.
    """
    n_total = len(y)
    n_fraud = int(y.sum())
    n_legit = n_total - n_fraud
    fraud_rate = n_fraud / n_total

    stats = {
        "label": label,
        "n_samples": n_total,
        "n_fraud": n_fraud,
        "n_legitimate": n_legit,
        "fraud_rate": fraud_rate,
    }

    tag = f" ({label})" if label else ""
    print(f"\nDataset statistics{tag}")
    print(f"  Total transactions : {n_total:>10,}")
    print(f"  Fraud              : {n_fraud:>10,}  ({fraud_rate:.4%})")
    print(f"  Legitimate         : {n_legit:>10,}  ({1 - fraud_rate:.4%})")
    return stats


# ---------------------------------------------------------------------------
# CLI entry point: generate and save synthetic dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = load_config()
    print("Generating synthetic dataset...")
    df = generate_synthetic_dataset(cfg)
    X, y = get_features_and_target(df)
    describe_dataset(y, label="synthetic")

    out_path = find_project_root() / cfg["dataset"]["synthetic"]["path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
