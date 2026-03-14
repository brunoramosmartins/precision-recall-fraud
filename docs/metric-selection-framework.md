# Metric Selection Framework

> Article-ready decision framework for choosing evaluation metrics in binary
> classification. Intended to become a figure or reference table in the article.
> Full working notes are in `notes/phase2-metrics.md`.

---

## The Core Question

Before choosing a metric, answer three questions:

1. **Are the classes balanced?**
   If yes, standard metrics (Accuracy, F1) are appropriate.
   If no, prefer metrics that are not dominated by the majority class.

2. **Are the costs symmetric?**
   If yes, F1 or Accuracy are defensible.
   If no, the cost structure must be reflected in both the metric and the threshold.

3. **Is the operating point fixed or variable?**
   If fixed (a specific recall requirement exists), use Precision@r.
   If variable (comparing models before deployment), use AUC-PR.

---

## Decision Table

| Condition | Recommended metric | Rationale |
|-----------|-------------------|-----------|
| Balanced classes, symmetric costs | Accuracy or F1 | Base rate ~50%; errors equally costly; standard metrics apply without distortion |
| Imbalanced, no cost spec., ranking focus | AUC-ROC | Threshold-free; measures ranking quality; use when positive/negative prevalence is comparable in the deployment context |
| Imbalanced, no cost spec., precision focus | AUC-PR | Not distorted by class imbalance; random baseline = base rate; rewards precision across all recall levels |
| Imbalanced, specific recall requirement known | Precision@r recall | Directly encodes the operational constraint; most actionable metric for deployment decisions |
| Calibration quality matters | Brier score or log-loss | Measures probabilistic accuracy; required when scores are used in downstream cost calculations |
| Multiple operating points need to be compared | Full PR curve | Visualize the tradeoff; annotate the cost-optimal threshold and the operational recall requirement |

---

## Fraud Detection Column

For credit card fraud detection specifically, where:
- Fraud prevalence: 0.1%–1% (severely imbalanced)
- Cost of missed fraud (C_FN) >> Cost of false alarm (C_FP)
- Fraud operations teams have a finite capacity to review flagged transactions

| Evaluation context | Recommended metric | Why |
|-------------------|--------------------|-----|
| Model selection (recall requirement unknown) | AUC-PR | Honest summary; random baseline = fraud rate; not fooled by imbalance |
| Model deployment (operational recall fixed) | Precision@r at the operational r | "Of all flagged transactions, how many are actually fraud?" at the recall target the team must meet |
| Regulatory or compliance reporting | Recall (TPR) only | Regulator cares about miss rate — how many frauds reached customers |
| Internal business review | Expected cost: C_FP × FP count + C_FN × FN count | The true decision metric; derived from the confusion matrix at the chosen threshold |
| Stakeholder communication | PR curve with annotated operating points | Interpretable; shows the precision-recall tradeoff at a glance; honest about what the model cannot do |

---

## The Cost-to-Threshold-to-Recall Chain

These three quantities are connected. When cost structure is known, the operational
recall requirement follows directly:

$$\tau^* = \frac{C_{FP}}{C_{FP} + C_{FN}} \implies r^* = \text{Recall}(\tau^*)$$

**Example** with C_FP = $5, C_FN = $200:

$$\tau^* = \frac{5}{205} \approx 0.024$$

The model should flag transactions with fraud probability > 2.4%. The recall
achieved at that threshold — call it r* — is the operationally correct recall
requirement. Precision@r* is then the right summary metric for this deployment.

This is why the metric is not "just a metric" — it is the business cost structure
encoded in mathematical form.

---

## Common Mistakes and Corrections

| Mistake | Correction |
|---------|-----------|
| Using Accuracy on an imbalanced dataset | A classifier that predicts "legitimate" for every transaction achieves >99% accuracy on a 0.1% fraud dataset. Accuracy is meaningless here. |
| Reporting only AUC-ROC for a fraud model | AUC-ROC can exceed 0.95 while Precision is below 0.05. Report AUC-PR alongside AUC-ROC, or instead of it. |
| Selecting the model with the highest F1 for deployment | F1 optimizes over all thresholds with no recall constraint. A model with lower F1 may have higher Precision@r at the operational recall. Always evaluate at the deployment operating point. |
| Using the default threshold of 0.5 | The optimal threshold depends on the cost ratio, not on the probability midpoint. With C_FP = $5 and C_FN = $200, the optimal threshold is 0.024 — not 0.5. |
| Treating Precision@75recall as if 75% is a fixed standard | The 75 is a business parameter. It reflects the tolerable miss rate for a specific deployment context. It is not a universal standard. |

---

## Visual Summary

```
Is the dataset balanced?
│
├── YES → Are costs symmetric?
│         │
│         ├── YES → Accuracy or F1
│         └── NO  → F-beta (β chosen from cost ratio)
│
└── NO  → Is a specific recall requirement known?
          │
          ├── YES → Precision@r recall
          │         (r derived from cost structure or operational constraint)
          │
          └── NO  → AUC-PR
                    (+ AUC-ROC as supplementary, with caveat)
```

---

## Reference: Metric Definitions at a Glance

| Metric | Formula | Range | Random baseline |
|--------|---------|-------|----------------|
| Accuracy | (TP + TN) / N | [0, 1] | = majority class fraction |
| Precision | TP / (TP + FP) | [0, 1] | = base rate |
| Recall (TPR) | TP / (TP + FN) | [0, 1] | = 0.5 (threshold-dependent) |
| F1 | 2PR / (P + R) | [0, 1] | Near base rate for imbalanced data |
| FPR | FP / (FP + TN) | [0, 1] | = threshold value for random classifier |
| AUC-ROC | Area under ROC curve | [0, 1] | 0.5 |
| AUC-PR | Area under PR curve | [0, 1] | = base rate |
| Precision@r | Precision at Recall = r | [0, 1] | = base rate |
