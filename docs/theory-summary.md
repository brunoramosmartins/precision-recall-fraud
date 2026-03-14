# Theoretical Summary — Phase 1

> Article-ready summary of the theoretical foundation.
> Full derivations and working notes are in `notes/phase1-theory.md`.

---

## Block 1 — The Confusion Matrix as a Probability Space

Every transaction scored by a binary classifier belongs to one of four mutually
exclusive events. Using Y for the true label and Ŷ for the model's prediction:

| | Predicted fraud (Ŷ=1) | Predicted legitimate (Ŷ=0) |
|---|---|---|
| **True fraud (Y=1)** | True Positive: {Y=1, Ŷ=1} | False Negative: {Y=1, Ŷ=0} |
| **True legitimate (Y=0)** | False Positive: {Y=0, Ŷ=1} | True Negative: {Y=0, Ŷ=0} |

This is not just a counting table. Each cell is an event in a probability space,
and the standard evaluation metrics are conditional probabilities defined on these
events.

**Precision** is the posterior probability that a flagged transaction is truly
fraudulent:

$$\text{Precision} = P(Y=1 \mid \hat{Y}=1) = \frac{TP}{TP + FP}$$

**Recall** (also: sensitivity, true positive rate) is the likelihood that the
model catches a truly fraudulent transaction:

$$\text{Recall} = P(\hat{Y}=1 \mid Y=1) = \frac{TP}{TP + FN}$$

The distinction matters. Precision conditions on the model's output — it answers
*"of everything the model flagged, how much was real fraud?"* Recall conditions on
the ground truth — it answers *"of all actual frauds, how many did the model
find?"* These are not interchangeable questions. A business that cares primarily
about customer experience asks about Precision. A business that cares primarily
about financial loss asks about Recall. Usually both matter, at different weights.

---

## Block 2 — Bayes' Theorem and the Base Rate Problem

Applying Bayes' theorem to the fraud detection setting connects Precision, Recall,
and the prevalence of fraud in a single equation:

$$\text{Precision} = \frac{\text{Recall} \cdot P(\text{fraud})}{\text{Recall} \cdot P(\text{fraud}) + \text{FPR} \cdot (1 - P(\text{fraud}))}$$

where $P(\text{fraud})$ is the **base rate** — the fraction of transactions in the
scoring population that are genuinely fraudulent — and FPR is the false positive
rate: $P(\hat{Y}=1 \mid Y=0)$.

The equation reveals a structural property of fraud detection: **Precision is
bounded above by the base rate, regardless of model quality.** When fraud is
rare, even a high-recall model with low FPR will generate predominantly false
alarms.

**Numerical illustration.** Consider a market with 10,000 daily transactions and
a fraud rate of 0.1% (10 genuine frauds). A model with Recall = 0.90 and
FPR = 0.05 produces:

- True Positives: 9 (catches 90% of 10 frauds)
- False Positives: 500 (flags 5% of 9,990 legitimate transactions)
- Precision: 9 / (9 + 500) ≈ **0.018**

For every fraud the model correctly flags, it generates approximately 55 false
alarms. Whether this is acceptable depends entirely on the relative cost of each
error type — a question the metric itself cannot answer.

This is not an argument against the model. It is an argument against evaluating
the model without reference to the base rate and cost structure of the deployment
context. Class imbalance in fraud detection is not a dataset problem to be
corrected — it is a reflection of the actual prevalence of fraud in the real world.
Treating it as such changes how we interpret every metric we compute.

---

## Block 3 — Cost-Sensitive Decision Theory and the Optimal Threshold

The binary classification decision reduces to: for each transaction with estimated
fraud probability $p = P(Y=1 \mid x)$, should we flag it (Ŷ=1) or approve it
(Ŷ=0)?

Define the cost matrix:

- $C_{FP}$: cost incurred by blocking a legitimate transaction (customer friction,
  lost interchange revenue, potential churn).
- $C_{FN}$: cost incurred by approving a fraudulent transaction (transaction loss,
  chargeback fees, dispute costs, reputational damage with card networks).

The expected cost of predicting fraud is $C_{FP} \cdot (1-p)$.
The expected cost of approving the transaction is $C_{FN} \cdot p$.

We should flag when the first is less than the second:

$$C_{FP}(1-p) < C_{FN} \cdot p \implies p > \frac{C_{FP}}{C_{FP} + C_{FN}}$$

The **optimal decision threshold** under asymmetric costs is therefore:

$$\tau^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$$

With representative values $C_{FP} = \$5$ (friction cost) and
$C_{FN} = \$200$ (fraud loss):

$$\tau^* = \frac{5}{205} \approx 0.024$$

The model should flag a transaction whenever its estimated fraud probability
exceeds **2.4%**, not 50%.

The default threshold of $\tau = 0.5$ is optimal if and only if $C_{FP} = C_{FN}$
(symmetric costs) — a condition that never holds in fraud detection. Evaluating
a fraud model at threshold 0.5 and reporting F1 as the primary metric embeds the
assumption of symmetric costs without stating it. Making this assumption explicit
is one of the goals of this article.

---

## Block 4 — F1, the F-beta Family, and Their Limits

F1 is the harmonic mean of Precision and Recall:

$$F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

The harmonic mean is always less than or equal to the arithmetic mean, with
equality only when Precision = Recall. This property penalizes extreme imbalance
between the two: a model with Precision = 0.99 and Recall = 0.01 receives
F1 ≈ 0.02, not 0.50. In this sense the harmonic mean is more conservative than
the arithmetic mean, and more appropriate for summarizing a precision-recall
tradeoff.

F1 treats Precision and Recall as equally important. For problems where errors
have asymmetric costs, the $F_\beta$ score generalizes this by assigning weight
$\beta^2$ to Recall relative to Precision:

$$F_\beta = \frac{(1+\beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

Setting $\beta = 2$ (F2) weights missed frauds four times more heavily than false
alarms, which is more defensible than F1 for most fraud detection contexts.
Setting $\beta = 0.5$ (F0.5) has the opposite effect — appropriate when false
alarms are the dominant concern.

**The critical distinction between F1 and Precision@recall:** F1 optimizes
freely across all thresholds and reports the best achievable harmonic mean. It
imposes no constraint on Recall. Precision@75recall asks a different question
entirely: "given that our operational requirement is to catch at least 75% of
frauds, what is the best precision we can achieve at that constraint?"

This is the difference between an unconstrained metric and a constrained one.
In practice, fraud operations teams do not say "maximize F1." They say "we cannot
miss more than 20% of frauds — what is our false alarm rate?" Precision@recall
is the metric that reflects that requirement. F1 is not.

---

## Connections to Article Structure

| Theoretical result | Where it appears in the article |
|---|---|
| Precision = posterior, Recall = likelihood | Section 2: The Confusion Matrix |
| Bayes formula in metric language | Section 3: Bayes and the Base Rate Problem |
| Base rate numerical example | Section 3: Numerical illustration |
| Optimal threshold derivation | Section 4: Asymmetry of Errors |
| Default threshold critique | Section 4: The τ=0.5 assumption |
| F1 harmonic mean definition | Section 5: F1 and the F-beta Family |
| F-beta generalization | Section 5: F-beta family |
| F1 vs Precision@recall distinction | Section 8: Precision@Recall — the Operating Point |
