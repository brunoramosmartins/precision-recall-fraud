# Phase 2 Study Notes — The Metric Landscape

> Personal working notes. Full metric ecosystem: ROC, PR curve, AUC variants,
> Precision@recall, and the decision framework.
> The article-ready framework lives in `docs/metric-selection-framework.md`.

---

## 2.1 — The ROC Curve: Origin, Definition, and Honest Caveats

### Definitions

**True Positive Rate (TPR)** — also called Recall or Sensitivity:

$$TPR = \frac{TP}{TP + FN} = P(\hat{Y}=1 \mid Y=1)$$

**False Positive Rate (FPR)**:

$$FPR = \frac{FP}{FP + TN} = P(\hat{Y}=1 \mid Y=0)$$

Note: FPR conditions on the negatives (legitimate transactions). With severe class
imbalance, the denominator (FP + TN) is very large. A model can have a low FPR
in absolute terms while still generating thousands of false alarms in absolute counts
— because the legitimate transaction pool is enormous.

This is the structural reason AUC-ROC is misleading for imbalanced problems.

### The ROC Curve

As we lower the decision threshold τ from 1 to 0, the model flags progressively
more transactions as fraud. Both TPR and FPR increase. The ROC curve is the trace
of all (FPR, TPR) pairs as τ sweeps from 1 to 0.

Key reference points:
- τ = 1: model flags nothing → TPR = 0, FPR = 0 → top-left corner of the curve
  starts at (0, 0)
- τ = 0: model flags everything → TPR = 1, FPR = 1 → bottom-right point (1, 1)
- Random classifier: diagonal line from (0,0) to (1,1) — TPR = FPR at every point

A good model hugs the top-left corner: high TPR at low FPR.

### AUC-ROC: Area Under the ROC Curve

$$\text{AUC-ROC} = \int_0^1 TPR(FPR) \, d(FPR)$$

**Probabilistic interpretation** (Wilcoxon-Mann-Whitney statistic):

$$\text{AUC-ROC} = P(\hat{p}(x^+) > \hat{p}(x^-))$$

where $x^+$ is a randomly drawn positive (fraud) and $x^-$ is a randomly drawn
negative (legitimate). This is the probability that the model assigns a higher
fraud score to a random fraud than to a random legitimate transaction.

This interpretation is elegant and threshold-free: it measures ranking quality
without committing to any operating point.

**AUC-ROC = 0.5** → random ranking (model is useless as a ranker).
**AUC-ROC = 1.0** → perfect ranking (every fraud ranked above every legitimate).

### Historical origin — Signal Detection Theory (SDT)

ROC analysis was developed in the 1940s–1950s for radar operators during World
War II. The problem: a radar operator receives a noisy signal and must decide
whether it represents an enemy aircraft (signal present) or random noise
(signal absent). The operator controls a threshold — high threshold means
few false alarms but also few real detections; low threshold means more
detections but also more false alarms.

The "receiver operating characteristic" was a tool to characterize the
detection-versus-false-alarm tradeoff of a human operator or a detection system,
across all possible thresholds, under different noise conditions.

The connection to ML classification is direct: the radar operator is the model,
enemy aircraft are positive instances, and noise is the negative class. The ROC
curve was imported from psychophysics and signal processing into ML in the 1980s
and 1990s. It was designed for settings where positives and negatives are roughly
comparable in prevalence — the conditions under which radar was calibrated.

The lesson: metrics are not neutral tools. They were designed with specific
assumptions about the problem. When those assumptions do not hold (as in fraud
detection, with 0.1% positive rate), using the metric without adjustment gives
a misleadingly optimistic picture.

### The honest caveat: AUC-ROC and class imbalance

**Claim:** A model can achieve AUC-ROC > 0.95 while still having very low precision
in a severely imbalanced dataset.

**Why:** FPR is normalized by the number of negatives. With 0.1% fraud rate
(10 frauds in 10,000 transactions, so 9,990 negatives), FP = 50 gives
FPR = 50 / 9,990 ≈ 0.005 — a very small number. The ROC curve will look excellent.

But Precision = TP / (TP + FP) = 9 / (9 + 50) = 0.15. Nearly 85% of flags are
wrong. The PR curve will show this clearly; the ROC curve will not.

This is not a flaw in AUC-ROC per se — it correctly measures ranking quality
among the population it was designed for. The issue is deploying it as the
primary metric for a highly imbalanced problem without acknowledging this limitation.

---

## 2.2 — The Precision-Recall Curve: The Right Tool for Imbalanced Data

### Definition

The PR curve is the trace of all (Recall, Precision) pairs as the decision
threshold τ sweeps from 1 to 0:

- τ = 1: model flags nothing. In the limit, Precision is undefined (no flags),
  conventionally set to 1 or handled as a special point.
- τ = 0: model flags everything. Recall = 1, Precision = base rate.

The PR curve lives in the [0,1] × [0,1] square, with Recall on the x-axis
and Precision on the y-axis.

### Why PR curves are more informative for imbalanced data

The key difference from ROC: **PR curves use the positive class as the reference
in both axes.** Precision = TP / (TP + FP) counts false positives in the
numerator directly, without normalizing by the (large) negative pool. This makes
precision sensitive to false positives in an absolute sense, not a relative one.

Davis & Goadrich (2006) proved a formal relationship: a curve that dominates in
ROC space also dominates in PR space, but the converse is not true. In other
words, PR curves can reveal differences between models that ROC curves flatten
out. For imbalanced problems, the "flattening out" effect of ROC is severe — PR
is the more discriminating tool.

### AUC-PR: Area Under the PR Curve

$$\text{AUC-PR} = \int_0^1 \text{Precision}(\text{Recall}) \, d(\text{Recall})$$

There is no clean probabilistic interpretation analogous to AUC-ROC. AUC-PR is
a summary of average precision across all recall levels — the area under the
precision-recall tradeoff curve.

### The AUC-PR baseline — a critical difference from AUC-ROC

- **AUC-ROC baseline** (random classifier): 0.5, regardless of base rate.
- **AUC-PR baseline** (random classifier): approximately equal to the base rate π.
  For 0.1% fraud rate: AUC-PR ≈ 0.001.

This matters enormously. An AUC-ROC of 0.97 on a 0.1% fraud dataset sounds
impressive. An AUC-PR of 0.20 on the same dataset — 200x above the random
baseline — is the more honest signal that the model is useful.

AUC-PR is a much more demanding metric, which is precisely why it is more
appropriate for fraud detection.

### The trade-off shape of the PR curve

A typical PR curve for a fraud model has this shape:
- **High precision, low recall (upper-left region):** at high thresholds, the
  model only flags the most obvious frauds — the easy cases. Few flags, but most
  are correct.
- **Declining precision as recall increases:** to catch more frauds, the model
  must lower its threshold and flag borderline transactions. More false positives
  enter, precision falls.
- **Low precision, recall near 1.0 (lower-right region):** to catch almost all
  frauds, the model flags a large fraction of transactions — mostly legitimate.

This trade-off shape reflects the fundamental structure of fraud detection: the
"easy" frauds look very different from legitimate transactions; the "hard" frauds
look similar. Any model that tries to catch all frauds will necessarily flag many
legitimate transactions.

### Comparison: ROC curve vs PR curve for the same model

On an imbalanced dataset:
- ROC curve: hugs the top-left corner, AUC ≈ 0.95+. Looks great.
- PR curve: falls steeply as recall increases, AUC ≈ 0.05–0.30. Looks harder.

Both are correct. They answer different questions:
- ROC: "How well does the model rank frauds above non-frauds?"
- PR: "At each recall level, how much noise does the model produce?"

For operational deployment, the PR question is the one that determines whether
the model is actually usable by a fraud operations team.

---

## 2.3 — Precision@K Recall: Definition, Interpretation, and Variability

### Formal definition

Let τ_r be the decision threshold that produces recall equal to r:

$$\tau_r = \arg\min_\tau \left| \text{Recall}(\tau) - r \right|$$

Then:

$$\text{Precision@}r = \text{Precision}(\tau_r) = P(Y=1 \mid \hat{Y}=1, \tau = \tau_r)$$

Precision@r is not a curve. It is a single point on the PR curve — the precision
achievable when the model is calibrated to catch exactly r × 100% of frauds.

### The value of r is a business parameter, not a metric property

This is the central insight of this section. Common notation like
"Precision@75recall" embeds a business decision in the metric name: we have
decided that missing more than 25% of frauds is unacceptable.

Examples of how business requirements translate to r:
- "We cannot miss more than 10% of frauds" → r = 0.90
- "We can tolerate missing up to 20% of frauds to limit customer friction" → r = 0.80
- "We need to catch at least three-quarters of all fraud" → r = 0.75

The choice of r reflects the cost structure. From Phase 1, the optimal threshold
under costs C_FP and C_FN is τ* = C_FP / (C_FP + C_FN). The recall at that
threshold is the operationally correct r for those costs. Precision@r with that
r is the precision that is achievable given the business cost structure.

This is how Phase 1 (cost theory) and Phase 2 (metric landscape) connect.

### Why Precision@r is more actionable than AUC-PR in a business context

AUC-PR summarizes precision across all possible recall levels. But in practice,
a fraud operations team does not operate at all recall levels simultaneously —
they operate at one specific threshold that reflects their current capacity to
handle alerts, their cost structure, and their risk tolerance.

AUC-PR answers: "How good is this model in general, across all possible operating
points?"

Precision@r answers: "How good is this model at the specific operating point we
actually use?"

For model selection in a production fraud system, Precision@r at the intended
operating r is the most direct evaluation signal.

### Interpolation

In practice, no threshold will give exactly recall = r. Two standard approaches:

1. **Nearest neighbor:** use the threshold whose recall is closest to r.
2. **Linear interpolation:** find the two adjacent points (r1, P1) and (r2, P2)
   where r1 < r < r2, and interpolate:
   $$\text{Precision@}r \approx P_1 + (P_2 - P_1) \cdot \frac{r - r_1}{r_2 - r_1}$$

Linear interpolation is standard in scikit-learn's implementation of average
precision (though note: `sklearn.metrics.average_precision_score` computes
AUC-PR, not Precision@r directly — computing Precision@r requires finding the
threshold on the PR curve manually).

---

## 2.4 — F1 vs Precision@Recall: The Key Distinction

### Two different optimization problems

**F1:** Find the threshold τ that maximizes the harmonic mean of Precision and
Recall. Report that maximum.

$$F1^* = \max_\tau \frac{2 \cdot P(\tau) \cdot R(\tau)}{P(\tau) + R(\tau)}$$

There is no constraint on recall. The model is free to trade any amount of recall
for precision or vice versa, as long as F1 is maximized. The resulting threshold
may be anywhere on the PR curve.

**Precision@r:** Fix a recall target r first. Find the threshold that achieves
that recall. Report the precision at that threshold.

$$\text{Precision@}r = P(\tau_r), \quad \tau_r = \arg\min_\tau |R(\tau) - r|$$

Recall is a constraint, not a variable. Precision is the objective given that
constraint.

### The analogy

F1 is like asking: "Drive from A to B as fast as possible, with no other
constraints."

Precision@r is like asking: "Drive from A to B, but you must stop at checkpoint
C (recall = r). Given that constraint, how fast can you go?"

These are different problems. A car that wins the first race (highest F1) may
not win the second (highest precision at the recall checkpoint).

### Concrete example: model ranking reversal

Consider two models on the same test set:

| Model | Threshold at F1-optimal | Precision | Recall | F1 | Precision@80recall |
|-------|------------------------|-----------|--------|----|--------------------|
| A | 0.35 | 0.72 | 0.68 | 0.70 | 0.52 |
| B | 0.28 | 0.60 | 0.79 | 0.68 | 0.60 |

By F1: Model A wins (0.70 > 0.68).
By Precision@80recall: Model B wins (0.60 > 0.52).

Which ranking is correct? It depends on the business requirement. If the
operations team must catch at least 80% of frauds (r = 0.80), Model B is the
right choice — Model A cannot even reach that recall at any threshold without
drastically reducing precision. Model B, operating at its optimal point for
r = 0.80, generates fewer false alarms than Model A would at the same recall.

F1 would have directed the team to deploy Model A. That is the wrong decision
given the stated business constraint.

### When to use F1, when to use Precision@r

Use F1 when:
- No specific recall requirement is known at model selection time.
- The goal is a general comparison across models before operational parameters
  are fixed.
- Costs are approximately symmetric (rare in fraud, common in other domains).

Use Precision@r when:
- A specific recall requirement exists (which it almost always does in
  production fraud systems).
- You are selecting a model for deployment, not just comparing on a benchmark.
- You want the metric to directly reflect the cost structure of the problem.

---

## 2.5 — Metric Selection Framework

### Full decision table

| Condition | Recommended primary metric | Why |
|-----------|---------------------------|-----|
| Balanced classes, symmetric costs | Accuracy or F1 | Base rate ~50%, errors equally costly — standard metrics apply |
| Imbalanced, no cost spec., ranking focus | AUC-ROC | Threshold-free; measures ranking quality; appropriate when positives and negatives are comparably important |
| Imbalanced, no cost spec., precision focus | AUC-PR | Not fooled by imbalance; random baseline = base rate; rewards precision at all recall levels |
| Imbalanced, specific recall requirement | Precision@r recall | Directly encodes the operational constraint; most actionable for deployment decisions |
| Calibration quality matters | Brier score or log-loss | Measures probabilistic accuracy; needed when model outputs are used as input to downstream cost calculations |
| Multiple operating points needed | Full PR curve | Visualize the tradeoff; annotate key business constraints |

### Fraud detection column

| Condition | Fraud detection recommendation | Rationale |
|-----------|-------------------------------|-----------|
| Model selection (no fixed recall requirement) | AUC-PR | Imbalanced data; random baseline = fraud rate; honest summary |
| Model deployment (operational recall set) | Precision@r at the operational recall | Directly answers "how many false alarms will the ops team handle?" |
| Regulatory or compliance audit | Recall (TPR) only | Regulator cares about miss rate, not false alarm rate |
| Internal cost reporting | Expected cost = C_FP × FP + C_FN × FN | The true business metric; derive from confusion matrix at the chosen threshold |
| Initial EDA and stakeholder communication | PR curve with annotated operating points | Visual; interpretable; honest about the precision-recall tradeoff |

### The universal caveat

No metric is universally correct. Every metric encodes assumptions about:
1. What errors cost (C_FP vs C_FN).
2. What the base rate is (affects Precision but not Recall or FPR).
3. Whether the operating point is fixed or variable.

Choosing a metric without stating these assumptions is choosing them implicitly.
The goal of a disciplined evaluation is to make the assumptions explicit —
in the metric choice, in the threshold choice, and in the way results are
communicated to stakeholders.

---

## Connections Between Phase 1 and Phase 2

| Phase 1 result | Phase 2 application |
|----------------|---------------------|
| Precision = posterior probability | Why Precision is sensitive to base rate; why PR curve uses Precision on the y-axis |
| Bayes: Precision degrades at low base rates | Why AUC-PR baseline = base rate; why AUC-ROC is misleading |
| Optimal threshold τ* = C_FP / (C_FP + C_FN) | How to choose r in Precision@r: r = Recall(τ*) |
| F1 assumes symmetric costs at the optimal threshold | Why F1 selects a different operating point than cost-optimal threshold |
| F-beta generalizes the cost weighting | F-beta is still unconstrained; Precision@r is the constrained alternative |

---

## Open Questions (to resolve in Phase 3 experiments)

1. On the Kaggle fraud dataset, what is the AUC-PR for a logistic regression vs
   a random forest? Does the difference appear in ROC space or only in PR space?
2. At what recall level does F1 peak for a typical fraud classifier? Is that
   threshold close to τ* = 0.024 from Phase 1?
3. Can we construct a real model pair where F1 and Precision@80recall rank
   differently? (Exercise 2-C suggests yes — Experiment E will confirm it.)
