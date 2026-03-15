"""Full pipeline runner — regenerates all experiment figures end-to-end.

Usage (from project root):
    python scripts/run_all.py

Requirements:
    pip install -r requirements.txt

For experiments that use the Kaggle dataset, download it first:
    See data/README.md for instructions.
    By default, all experiments use the synthetic dataset for full reproducibility.

Reproducibility contract:
    Fresh environment → pip install -r requirements.txt → python scripts/run_all.py
    → all figures in figures/ regenerated deterministically.
    All seeds are controlled by config.yaml.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_config

import scripts.experiment_a_accuracy_trap as exp_a
import scripts.experiment_b_roc_vs_pr as exp_b
import scripts.experiment_c_threshold_selection as exp_c
import scripts.experiment_d_precision_at_recall as exp_d
import scripts.experiment_e_ranking_comparison as exp_e


def main() -> None:
    print("=" * 60)
    print("  Not All Errors Cost the Same — Full Experiment Pipeline")
    print("=" * 60)

    cfg = load_config()
    t_start = time.time()

    results = {}

    # Experiment A
    t0 = time.time()
    exp_a.run(cfg)
    print(f"  [Exp A done in {time.time() - t0:.1f}s]")

    # Experiment B — returns best model for Exp D
    t0 = time.time()
    b_output = exp_b.run(cfg)
    print(f"  [Exp B done in {time.time() - t0:.1f}s]")
    results["best_model_key"] = b_output["best_model_key"]
    results["models_b"] = b_output["models_fitted"]

    # Experiment C
    t0 = time.time()
    exp_c.run(cfg)
    print(f"  [Exp C done in {time.time() - t0:.1f}s]")

    # Experiment D — uses best model from B
    t0 = time.time()
    best_key = results["best_model_key"]
    best_model = results["models_b"][best_key]
    exp_d.run(cfg, model=best_model)
    print(f"  [Exp D done in {time.time() - t0:.1f}s]")

    # Experiment E
    t0 = time.time()
    exp_e.run(cfg)
    print(f"  [Exp E done in {time.time() - t0:.1f}s]")

    # Summary
    elapsed = time.time() - t_start
    from src.data import find_project_root
    figures_dir = find_project_root() / cfg["figures"]["output_dir"]
    figures = sorted(figures_dir.glob(f"*.{cfg['figures']['format']}"))

    print("\n" + "=" * 60)
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  {len(figures)} figure(s) saved to {figures_dir}/")
    for f in figures:
        print(f"    {f.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
