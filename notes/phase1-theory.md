# Phase 1 Study Notes — Theoretical Foundation

> Personal working notes. Derivations, intuition, and commentary.
> The clean version of this material lives in `docs/theory-summary.md`.

---

## 1.1 — Conditional Probability: The Language Underneath Precision and Recall

### Formal definition

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

Read: "the probability of A given that B has occurred." The condition P(B) > 0 is
not pedantry — it matters. If the model never predicts fraud (P(Ŷ=1) = 0),
Precision is undefined. This edge case appears in practice when thresholds are
set too high.

### The confusion matrix as a probability space

Let Y be the true label (1 = fraud, 0 = legitimate) and Ŷ be the model's
prediction. Every transaction belongs to exactly one of four events:


| Event                                          | Notation   | Name                |
| ---------------------------------------------- | ---------- | ------------------- |
| Model predicts fraud, truly is fraud           | {Y=1, Ŷ=1} | True Positive (TP)  |
| Model predicts fraud, truly is legitimate      | {Y=0, Ŷ=1} | False Positive (FP) |
| Model predicts legitimate, truly is fraud      | {Y=1, Ŷ=0} | False Negative (FN) |
| Model predicts legitimate, truly is legitimate | {Y=0, Ŷ=0} | True Negative (TN)  |


These four events are mutually exclusive and exhaustive. Their counts sum to N
(total transactions). Written as probabilities:

$$P(Y=1, \hat{Y}=1) + P(Y=0, \hat{Y}=1) + P(Y=1, \hat{Y}=0) + P(Y=0, \hat{Y}=0) = 1$$

### Precision as a conditional probability

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{P(Y=1, \hat{Y}=1)}{P(\hat{Y}=1)} = P(Y=1 \mid \hat{Y}=1)$$

**What it means:** given that the model flagged a transaction as fraud, what is
the probability it actually is fraud?

**Why it is a posterior:** we observe the prediction Ŷ=1 first, then ask about
the true state Y. The prediction is the evidence; the true label is what we are
estimating. This is precisely the Bayesian posterior structure: P(truth | evidence).

### Recall as a conditional probability

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{P(Y=1, \hat{Y}=1)}{P(Y=1)} = P(\hat{Y}=1 \mid Y=1)$$

**What it means:** given that a transaction is truly fraud, what is the
probability the model catches it?

**Why it is a likelihood:** we condition on the true state Y=1 and ask how likely
the model is to produce a positive prediction. This is the likelihood P(evidence | truth).

### The asymmetry in plain language

- Precision answers: "of everything the model accused, how often was it right?"
- Recall answers: "of everything that was actually guilty, how much did the model catch?"

They ask fundamentally different questions. Optimizing one at the expense of the
other has different business consequences. This is why using a single number (F1)
to collapse both is a decision that requires justification, not a default.

---

## 1.2 — Bayes' Theorem: Connecting Precision, Recall, and Base Rate

### Bayes' theorem (standard form)

$$P(Y=1 \mid \hat{Y}=1) = \frac{P(\hat{Y}=1 \mid Y=1) \cdot P(Y=1)}{P(\hat{Y}=1)}$$

### Rewrite in metric language

Substituting the metric names:

$$\text{Precision} = \frac{\text{Recall} \cdot P(\text{fraud})}{P(\hat{Y}=1)}$$

where:

- $P(\text{fraud}) = P(Y=1)$ is the **base rate** — the fraction of transactions
that are actually fraudulent in the population being scored.
- $P(\hat{Y}=1)$ is the fraction of transactions the model flags as fraud.

Expanding the denominator using the law of total probability:

$$P(\hat{Y}=1) = P(\hat{Y}=1 \mid Y=1) \cdot P(Y=1) + P(\hat{Y}=1 \mid Y=0) \cdot P(Y=0)$$
$$= \text{Recall} \cdot P(\text{fraud}) + \text{FPR} \cdot (1 - P(\text{fraud}))$$

So the full form is:

$$\text{Precision} = \frac{\text{Recall} \cdot P(\text{fraud})}{\text{Recall} \cdot P(\text{fraud}) + \text{FPR} \cdot (1 - P(\text{fraud}))}$$

This formula contains everything. It shows that Precision depends on three things:

1. How good the model is at catching fraud (Recall).
2. How often the model falsely flags legitimate transactions (FPR).
3. How common fraud actually is (base rate).

A model can have excellent Recall and very low FPR and *still* have low Precision
if the base rate is low enough. This is not a model failure — it is a mathematical
consequence of Bayes' theorem.

### Key consequence: low base rate destroys precision

**Claim:** A model with high recall can have very low precision when the base rate
is low.

**Proof:** Fix Recall = 0.90 and FPR = 0.05. Vary the base rate π = P(fraud):

$$\text{Precision}(\pi) = \frac{0.90 \cdot \pi}{0.90 \cdot \pi + 0.05 \cdot (1 - \pi)}$$

As π → 0: numerator → 0, denominator → 0.05. So Precision → 0.
As π → 1: numerator → 0.90, denominator → 0.90. So Precision → 1.

The base rate controls the precision ceiling. When fraud is rare, even a good
model produces mostly false alarms.

### Numerical example (10,000 transactions, 0.1% fraud rate)

Given:

- N = 10,000 total transactions
- P(fraud) = 0.001 → 10 actual frauds, 9,990 legitimate transactions
- Recall = 0.90 → model catches 9 of 10 frauds (TP = 9, FN = 1)
- FPR = 0.05 → model flags 5% of legitimate transactions (FP = 0.05 × 9,990 ≈ 500)

Computing Precision:

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{9}{9 + 500} = \frac{9}{509} \approx 0.018$$

**Interpretation:** approximately 98% of the transactions the model flags as fraud
are actually legitimate. For every genuine fraud caught, the system generates ~55
false alarms.

Is this acceptable? That depends entirely on the costs. If blocking a legitimate
transaction costs $5 in customer friction and missing a fraud costs $200:

- Cost of the 500 false positives: 500 × $5 = $2,500
- Cost of the 1 missed fraud: 1 × $200 = $200
- Total cost: $2,700

Compare to a model with Recall = 0.50, FPR = 0.001:

- TP = 5, FN = 5, FP = 10
- Cost of 10 false positives: 10 × $5 = $50
- Cost of 5 missed frauds: 5 × $200 = $1,000
- Total cost: $1,050

The first model flags more fraud but costs more overall. The second model misses
more fraud but is cheaper. This is a **business decision**, not a model quality
judgment.

### Connection to accuracy

Accuracy on this dataset: (TP + TN) / N = (9 + 9,490) / 10,000 = 99.99%.
A model that predicts "legitimate" for every transaction gets:
(0 + 9,990) / 10,000 = 99.90% accuracy — barely worse.

Accuracy is useless here because the base rate makes the denominator misleading.
This is not an edge case; it is the norm in fraud detection.

---

## 1.3 — The Asymmetry of Errors: Cost-Sensitive Decision Theory

### The cost matrix


|                           | Predicted fraud (Ŷ=1) | Predicted legitimate (Ŷ=0) |
| ------------------------- | --------------------- | -------------------------- |
| **True fraud (Y=1)**      | 0 (correct)           | C_FN (missed fraud)        |
| **True legitimate (Y=0)** | C_FP (false alarm)    | 0 (correct)                |


Where:

- $C_{FP}$: cost of a false positive — blocking a legitimate transaction.
Includes customer friction, potential churn, lost interchange revenue.
- $C_{FN}$: cost of a false negative — letting fraud through.
Includes the full transaction value lost, chargeback fees, operational
dispute costs, and reputational damage with card networks.

In real fraud detection, $C_{FN} \gg C_{FP}$ (typically by a factor of 20–100x).

### Deriving the optimal threshold

A classifier outputs a probability score $p = P(Y=1 \mid x)$ for each transaction.
We predict fraud if $p \geq \tau$ for some threshold $\tau \in (0,1)$.

The expected cost of predicting fraud (Ŷ=1) for a transaction with score p:

$$E[\text{cost} \mid \hat{Y}=1] = C_{FP} \cdot P(Y=0 \mid x) = C_{FP} \cdot (1-p)$$

The expected cost of predicting legitimate (Ŷ=0) for the same transaction:

$$E[\text{cost} \mid \hat{Y}=0] = C_{FN} \cdot P(Y=1 \mid x) = C_{FN} \cdot p$$

We should predict fraud when it is cheaper to do so:

$$C_{FP} \cdot (1-p) < C_{FN} \cdot p$$

Expanding:

$$C_{FP} - C_{FP} \cdot p < C_{FN} \cdot p$$
$$C_{FP} < p \cdot (C_{FP} + C_{FN})$$
$$p > \frac{C_{FP}}{C_{FP} + C_{FN}}$$

Therefore, the **optimal threshold** is:

$$\boxed{\tau^* = \frac{C_{FP}}{C_{FP} + C_{FN}}}$$

### Interpretation

With $C_{FP} = 5$ and $C_{FN} = 200$:

$$\tau^* = \frac{5}{5 + 200} = \frac{5}{205} \approx 0.024$$

The model should flag a transaction as fraud whenever its fraud probability
exceeds **2.4%** — not 50%.

Using the default threshold of 0.5 with these costs means: "I will only flag
a transaction if I am more than 50% sure it is fraud." But given that missing a
fraud costs 40 times more than a false alarm, flagging at 2.4% confidence is the
mathematically correct behavior.

### Why default threshold = 0.5 is wrong in fraud detection

$\tau = 0.5$ is optimal if and only if:

$$\frac{C_{FP}}{C_{FP} + C_{FN}} = 0.5 \implies C_{FP} = C_{FN}$$

That is, it assumes symmetric costs. It also implicitly assumes that the
decision is made without reference to the base rate (the prior $P(Y=1)$ is folded
into the model's output probability, but the threshold calibration ignores costs).

In fraud detection: costs are never symmetric, and the base rate is rarely 50%.
The default threshold is wrong by construction.

### The direction of the error

When $C_{FN} \gg C_{FP}$ (as in fraud): $\tau^*$ is small → the model flags more
transactions → Recall increases, Precision decreases. This is the correct behavior.
It is not a bug. It reflects a deliberate choice to accept more false alarms in
exchange for fewer missed frauds — a choice that is justified by the cost structure.

---

## 1.4 — Why F1 Is Not "50/50 Precision and Recall"

### F1 as harmonic mean

$$F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

Why harmonic mean and not arithmetic mean?

The arithmetic mean of Precision and Recall gives equal credit to extreme values.
The harmonic mean is always ≤ the arithmetic mean, with equality only when
Precision = Recall. It punishes large imbalances between the two.

Example: Model with Precision = 0.99, Recall = 0.01.

- Arithmetic mean: (0.99 + 0.01) / 2 = 0.50
- Harmonic mean (F1): 2 × 0.99 × 0.01 / (0.99 + 0.01) = 0.0198 ≈ 0.02

A model that catches almost nothing but is precise when it does — F1 correctly
scores this near zero. The arithmetic mean would give it a 0.50, which is
misleading.

### What "equal weight to Precision and Recall" actually means

F1 treats Precision and Recall as equally important in the sense that it is
symmetric: F1(P, R) = F1(R, P). It does not fix either at 50%.

The "50/50" confusion comes from misreading symmetry as balance. What F1 actually
does is: it searches across all possible thresholds and reports the best F1
achievable. It does not constrain recall — it optimizes over the full tradeoff.

**Precision@75recall** is a different problem statement entirely: "I require at
least 75% recall. Given that constraint, what is the best precision the model can
achieve?" This is a constrained optimization. F1 is an unconstrained one.

### The F-beta family

$$F_\beta = \frac{(1+\beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

The parameter $\beta^2$ is the weight given to Recall relative to Precision.


| beta | Interpretation                                | Use case                                                   |
| ---- | --------------------------------------------- | ---------------------------------------------------------- |
| 0.5  | Precision counts twice as much as Recall      | Spam filter — false positives are costly                   |
| 1.0  | Equal weight                                  | General purpose, no cost prior                             |
| 2.0  | Recall counts four times as much as Precision | Fraud, medical diagnosis — false negatives are very costly |


For fraud detection, $F_2$ (β=2) is more defensible than $F_1$ as an aggregate
metric. But it still does not fix the recall constraint — it just re-weights the
tradeoff. Precision@recall makes the constraint explicit.

### Proof that harmonic mean ≤ arithmetic mean (AM-HM inequality)

For any two positive reals a, b:

$$\frac{2ab}{a+b} \leq \frac{a+b}{2}$$

Proof: Cross-multiply (valid since both sides are positive):
$$4ab \leq (a+b)^2 = a^2 + 2ab + b^2$$
$$0 \leq a^2 - 2ab + b^2 = (a-b)^2$$

This is always true, with equality iff a = b. Therefore F1 ≤ arithmetic mean of
Precision and Recall, with equality only when they are equal.

---

## Summary of Key Results


| Result                   | Formula                                          | Business meaning                           |
| ------------------------ | ------------------------------------------------ | ------------------------------------------ |
| Precision = posterior    | P(Y=1 | Ŷ=1)                                     | Of all flags, how many were real fraud?    |
| Recall = likelihood      | P(Ŷ=1 | Y=1)                                     | Of all frauds, how many did we catch?      |
| Bayes in metric language | Precision = Recall × P(fraud) / P(Ŷ=1)           | Precision degrades when fraud is rare      |
| Optimal threshold        | τ* = C_FP / (C_FP + C_FN)                        | With $5 FP and $200 FN costs: flag at 2.4% |
| F1 vs P@recall           | F1 optimizes freely; P@r constrains recall first | P@r reflects operational requirements      |


---

## Open Questions (to revisit in Phase 2)

- How does the PR curve encode the Bayes relationship across all thresholds?
- What does it mean for AUC-PR that its baseline equals the base rate?
- The numerical example above shows Precision ≈ 0.018 at 0.1% fraud rate.
At what base rate does this model become useful (Precision > 0.50)?
Solve: 0.90π / (0.90π + 0.05(1-π)) = 0.50. This gives π ≈ 5%.
A market where fraud is below 5% will have low precision regardless of model quality.
This has major implications for deploying a fraud model in a new geography.

