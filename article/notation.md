# Notation Glossary

All symbols used in the article, consistent across all sections and figures.

---

## Random Variables and Labels

| Symbol | Definition |
|--------|------------|
| $Y$ | True label. $Y = 1$ denotes a fraudulent transaction; $Y = 0$ denotes a legitimate one. |
| $\hat{Y}$ | Model prediction. $\hat{Y} = 1$ means the model flags the transaction as fraud. |
| $x$ | Feature vector for a single transaction. |
| $\hat{p}(x)$ | Model's estimated fraud probability: $\hat{p}(x) = P(Y=1 \mid x)$. |
| $\pi$ | Base rate (prevalence of fraud): $\pi = P(Y=1)$. |

---

## Confusion Matrix Counts

| Symbol | Definition |
|--------|------------|
| $N$ | Total number of transactions. |
| $TP$ | True Positives: transactions where $Y=1$ and $\hat{Y}=1$. |
| $FP$ | False Positives: transactions where $Y=0$ and $\hat{Y}=1$. |
| $FN$ | False Negatives: transactions where $Y=1$ and $\hat{Y}=0$. |
| $TN$ | True Negatives: transactions where $Y=0$ and $\hat{Y}=0$. |

---

## Evaluation Metrics

| Symbol | Formula | Definition |
|--------|---------|------------|
| $P$ | $TP / (TP + FP)$ | Precision: posterior probability that a flagged transaction is fraud. |
| $R$ | $TP / (TP + FN)$ | Recall (= Sensitivity = TPR): likelihood that a fraud is flagged. |
| $FPR$ | $FP / (FP + TN)$ | False Positive Rate. |
| $F_1$ | $2PR / (P + R)$ | Harmonic mean of Precision and Recall. |
| $F_\beta$ | $(1+\beta^2)PR / (\beta^2 P + R)$ | Generalised F-score weighting Recall $\beta^2$ times over Precision. |
| $P\text{@}r$ | $P(\tau_r)$ | Precision at recall level $r$: precision when $R(\tau) = r$. |
| AUC-ROC | $\int_0^1 TPR \, d(FPR)$ | Area under the ROC curve. |
| AUC-PR | $\int_0^1 P \, dR$ | Area under the Precision-Recall curve. |

---

## Decision Threshold and Costs

| Symbol | Definition |
|--------|------------|
| $\tau$ | Decision threshold: predict fraud if $\hat{p}(x) \geq \tau$. |
| $\tau^*$ | Optimal decision threshold under asymmetric costs: $C_{FP} / (C_{FP} + C_{FN})$. |
| $\tau_r$ | Threshold producing recall exactly $r$: $\arg\min_\tau \lvert R(\tau) - r \rvert$. |
| $C_{FP}$ | Cost of a false positive (blocking a legitimate transaction). |
| $C_{FN}$ | Cost of a false negative (approving a fraudulent transaction). |

---

## Probability Notation

| Symbol | Definition |
|--------|------------|
| $P(A)$ | Probability of event $A$. |
| $P(A \mid B)$ | Conditional probability of $A$ given $B$. |
| $P(A \cap B)$ | Joint probability of $A$ and $B$. |

---

## Conventions

- All probabilities are in $[0, 1]$.
- Subscripts on model scores: $\hat{p}(x^+)$ denotes the score for a positive (fraud) instance; $\hat{p}(x^-)$ for a negative (legitimate) instance.
- Log-base is natural unless stated otherwise.
- Expected values use standard $\mathbb{E}[\cdot]$ notation when they appear.
