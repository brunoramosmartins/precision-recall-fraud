# Experiments Summary

> This document connects each experiment to the theoretical claim it validates
> and records the key result. Numerical values are filled in after running
> `python scripts/run_all.py`.

---

## Experiment A — The Accuracy Trap

**Script:** `scripts/experiment_a_accuracy_trap.py`
**Figure:** `figures/exp_a_accuracy_trap.png`

**Theoretical claim (Phase 1, §1.2):**
Accuracy is meaningless for imbalanced classification. A majority-class dummy
classifier can achieve >99% accuracy on a 0.1% fraud dataset while catching
zero frauds. AUC-PR correctly scores it at the base rate.

**What the experiment does:**
Trains a DummyClassifier (always predicts "legitimate") and a Logistic Regression
on the synthetic fraud dataset. Computes Accuracy, Precision, Recall, F1,
AUC-ROC, and AUC-PR for each. Visualizes the comparison as a bar chart.

**Expected results:**

| Metric | Dummy Classifier | Logistic Regression |
|--------|-----------------|---------------------|
| Accuracy | ~0.999 | < 0.999 |
| Precision | undefined / 0 | > 0 |
| Recall | 0.0 | > 0 |
| F1 | 0.0 | > 0 |
| AUC-ROC | 0.5 | >> 0.5 |
| AUC-PR | ≈ base rate | >> base rate |

**Actual results:** *(fill in after running)*

| Metric | Dummy Classifier | Logistic Regression |
|--------|-----------------|---------------------|
| Accuracy | | |
| Precision | | |
| Recall | | |
| F1 | | |
| AUC-ROC | | |
| AUC-PR | | |

**Connection to article:** Section 3 (Bayes and the Base Rate Problem) and
Section 10 (Metric Selection Framework). The figure will appear alongside the
discussion of why accuracy misleads and why AUC-PR is the honest baseline.

---

## Experiment B — ROC vs PR Curve Side by Side

**Script:** `scripts/experiment_b_roc_vs_pr.py`
**Figures:** `figures/exp_b_roc_curves.png`, `figures/exp_b_pr_curves.png`

**Theoretical claim (Phase 2, §2.1–2.2):**
PR curves reveal discrimination differences that ROC curves hide under class
imbalance. The weak classifier looks competitive in ROC space but poor in PR space.

**What the experiment does:**
Trains three classifiers (Logistic Regression, Random Forest, shallow decision
tree). Plots both the ROC and PR curves for all three on the same axes. Annotates
AUC values and the random baseline.

**Expected results:**

| Model | AUC-ROC | AUC-PR |
|-------|---------|--------|
| Logistic Regression | | |
| Random Forest | | |
| Weak Classifier (depth=2) | | |
| Random baseline | 0.500 | ≈ base rate |

**Expected pattern:** AUC-ROC values clustered together; AUC-PR values spread apart.

**Actual results:** *(fill in after running)*

| Model | AUC-ROC | AUC-PR |
|-------|---------|--------|
| Logistic Regression | | |
| Random Forest | | |
| Weak Classifier (depth=2) | | |

**Connection to article:** Section 6 (ROC Curve) and Section 7 (PR Curve). The
side-by-side figures are the central visual evidence for the ROC vs PR argument.

---

## Experiment C — Threshold Selection and Its Effect on the Confusion Matrix

**Script:** `scripts/experiment_c_threshold_selection.py`
**Figures:** `figures/exp_c_threshold_sweep.png`, `figures/exp_c_confusion_matrices.png`

**Theoretical claim (Phase 1, §1.3):**
The default threshold of 0.5 is almost never optimal for fraud detection. The
optimal threshold is τ* = C_FP / (C_FP + C_FN) = 5 / 205 ≈ 0.024. Operating
at the wrong threshold has measurable business consequences in the confusion matrix.

**What the experiment does:**
Trains a Logistic Regression. Sweeps thresholds from 0.01 to 0.99 and plots
Precision, Recall, and F1 vs threshold. Annotates three thresholds: default (0.5),
F1-optimal, and cost-optimal (τ* ≈ 0.024). Shows the confusion matrix at each.

**Expected results:**

| Threshold | Precision | Recall | F1 | FP count | FN count |
|-----------|-----------|--------|----|----------|----------|
| Default (0.5) | high | low | medium | few | many |
| F1-optimal | medium | medium | high | medium | medium |
| Cost-optimal (≈0.024) | low | high | medium | many | very few |

**Actual results:** *(fill in after running)*

| Threshold | Precision | Recall | F1 | FP | FN |
|-----------|-----------|--------|----|----|-----|
| Default (0.5) | | | | | |
| F1-optimal | | | | | |
| Cost-optimal (τ*) | | | | | |

**Connection to article:** Section 4 (Asymmetry of Errors) and Section 8
(Precision@Recall). The confusion matrix figure at the cost-optimal threshold
is the empirical confirmation of the threshold derivation from Phase 1.

---

## Experiment D — Precision@Recall as an Operating Point

**Script:** `scripts/experiment_d_precision_at_recall.py`
**Figure:** `figures/exp_d_precision_at_recall.png`

**Theoretical claim (Phase 2, §2.3):**
Precision@75recall and Precision@85recall represent meaningfully different business
decisions. Higher recall requirements force precision down and FP counts up. The
choice between them is a business decision, not a technical one.

**What the experiment does:**
Uses the best model from Experiment B. Computes Precision and confusion matrix
counts at recall targets 0.75, 0.80, 0.85, 0.90. Displays the trade-off as a
side-by-side bar chart (precision + FP/FN counts). Computes expected cost at
each operating point.

**Expected results:**

| Recall target | Precision | FP count | FN count | Total expected cost |
|---------------|-----------|----------|----------|---------------------|
| r = 0.75 | | | | |
| r = 0.80 | | | | |
| r = 0.85 | | | | |
| r = 0.90 | | | | |

**Connection to article:** Section 8 (Precision@Recall — the Operating Point).
This is the section's primary empirical evidence. The figure will be accompanied
by the "business memo" exercise from Phase 3 paper exercises.

---

## Experiment E — F1 vs Precision@75Recall: Model Ranking Can Differ

**Script:** `scripts/experiment_e_ranking_comparison.py`
**Figure:** `figures/exp_e_ranking_comparison.png`

**Theoretical claim (Phase 2, §2.4):**
Ranking models by F1 can give a different answer than ranking by Precision@75recall.
Metric choice changes which model gets deployed. This is the core empirical result
of the article.

**What the experiment does:**
Trains five model variants (Logistic Regression at C=0.01/1/10, Random Forest at
depth=5/15). Ranks all models by F1, AUC-PR, P@75R, P@80R, P@85R, P@90R.
Identifies model pairs where F1 and Precision@75recall give opposite rankings.

**Expected results:**

| Model | F1 (best) | AUC-PR | P@75R | P@80R | P@85R | P@90R |
|-------|-----------|--------|-------|-------|-------|-------|
| LR (balanced) | | | | | | |
| LR (C=0.01) | | | | | | |
| LR (C=10) | | | | | | |
| RF (depth=5) | | | | | | |
| RF (depth=15) | | | | | | |

**Expected pattern:** At least one model pair where F1 ranking and P@75R ranking
are reversed. The heatmap makes this divergence visually explicit.

**Actual results:** *(fill in after running)*

| Metric | Top-ranked model |
|--------|-----------------|
| F1 (best) | |
| AUC-PR | |
| P@75R | |
| P@80R | |
| P@85R | |
| P@90R | |

**Connection to article:** Section 8 (Precision@Recall) and Section 9 (Experiments).
This is the article's central empirical finding: the choice between F1 and
Precision@recall is not just philosophical — it changes which model you deploy.
