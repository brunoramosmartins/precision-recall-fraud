---
title: "Not All Errors Cost the Same"
description: "Bayesian foundations of evaluation metrics for fraud detection: from F1 to Precision@Recall."
date: 2026-03-19
category: machine-learning
reading_time: "25 min"
---

# Not All Errors Cost the Same

**Bayesian Foundations of Evaluation Metrics for Fraud Detection: from F1 to Precision@Recall**

*Bruno Ramos Martins — 2026*

---

## Abstract

Fraud detection is an imbalanced classification problem in which two types of errors carry fundamentally different consequences: missing a fraud costs far more than blocking a legitimate transaction. Despite this asymmetry, F1 score — which treats both error types as equally costly — remains the default evaluation metric in most practitioner workflows. This article argues that F1 is not wrong, but that its assumptions are rarely made explicit; and that when those assumptions are violated, it produces misleading model rankings. Starting from a probabilistic reading of the confusion matrix, we derive precision and recall as conditional probabilities and connect them to Bayes' theorem. We show that class imbalance is not a data problem but a base rate problem, and that the optimal decision threshold follows directly from the ratio of error costs. We then map the full metric landscape — ROC curves, Precision-Recall curves, and Precision@Recall — showing that each metric encodes a different assumption about the operating context. A suite of reproducible experiments on a synthetic fraud dataset confirms that metric choice changes model rankings, that the default threshold of 0.5 is rarely optimal, and that ranking models by F1 versus Precision@Recall can select different winners — meaning that the choice of metric, not the choice of algorithm, determines which model gets deployed. The article closes with a decision framework: a set of practical questions whose answers lead unambiguously to the right metric for a given deployment context.

---

## 1. Introduction

The motivation for this article emerged from a technical interview centered on a fraud detection case. The format was straightforward: at each stage of building a solution, it was necessary to explain what to do, how to do it, and why.

This type of problem exposes an important gap. When preparing for roles in machine learning, it is common to focus heavily on algorithms, hyperparameter tuning, and optimization techniques. There is a substantial amount of theory to cover, and much of the learning process revolves around understanding model behavior and improving performance. However, when the problem is presented in a more realistic, contextualized setting, this “checklist-driven” approach often breaks down.

At one point in the interview, a seemingly simple question was asked: among several evaluation metrics, which one would you choose and why. The answer was based on my knowledge at the time, but it lacked depth. In retrospect, the issue was not the choice itself, but the absence of an explicit reasoning process. The metric selection was largely automatic, without a clear articulation of the assumptions being made.

This realization motivated this article.

The question is not whether F1 is a bad metric. It is whether F1 is an *honest* metric in a specific context: binary classification under severe class imbalance, where the two types of errors carry asymmetric costs. In fraud detection, a false negative — approving a fraudulent transaction — typically costs an order of magnitude more than a false positive — blocking a legitimate one. F1 weights these two error types equally by construction. That implicit assumption is rarely made explicit and, in practice, is rarely valid.

This article makes those assumptions explicit, derives what the appropriate metric should look like under asymmetric costs, and shows empirically that the choice of metric changes model rankings and deployment decisions in ways that materially affect outcomes.

### What drives this article

Every evaluation metric is a compressed answer to the question: *"What should the model optimise for?"* Different metrics encode different answers. Accuracy encodes: *"Every correct prediction counts equally."* F1 encodes: *"Precision and Recall matter equally, and I care more about the minority class than accuracy alone."* Precision@Recall encodes: *"I have a non-negotiable recall requirement, and I want the highest precision achievable within it."*

The central thesis is this: in fraud detection, the last formulation is the honest one. The mathematical derivation that justifies it connects to Bayes' theorem, cost-sensitive decision theory, and the geometry of Precision-Recall curves. All of that machinery is here, built bottom-up.

### Who this article is for

A data scientist or machine learning engineer who uses precision, recall, F1, and AUC-ROC in daily work, and who wants to understand not just *how* these metrics are computed but *what* they measure and *when* each one is appropriate. No prior knowledge of Bayesian inference is assumed. The necessary conditional probability is developed from scratch.

### Roadmap

The article is structured as follows. Sections 2–4 build the probabilistic foundation: the confusion matrix as conditional probability, the role of the base rate through Bayes' theorem, and cost-sensitive decision theory with its optimal threshold. Sections 5–8 map the metric landscape: F1, the ROC curve, the Precision-Recall curve, and Precision@Recall. Section 9 presents five experiments that provide empirical evidence for every claim made in the theory sections. Section 10 synthesises the metric selection framework. Section 11 closes with practical takeaways.

---

## 2. The Confusion Matrix — A Probabilistic Reading

Most practitioners encounter the confusion matrix early in their training, usually presented as a 2×2 table of counts. This section reframes it as a table of *conditional probabilities*. The reframing is small but changes everything.

### 2.1 The four cells as events

Let $Y \in \lbrace 0, 1 \rbrace$ be the true label — 1 for fraud, 0 for legitimate — and $\hat{Y} \in \lbrace 0, 1 \rbrace$ the model's prediction. The four cells of the confusion matrix correspond to the four intersections of the event space $\lbrace Y = 0, Y = 1 \rbrace \times \lbrace \hat{Y} = 0, \hat{Y} = 1 \rbrace$:

| | Predicted: Fraud ($\hat{Y}=1$) | Predicted: Legitimate ($\hat{Y}=0$) |
|---|---|---|
| **Actual: Fraud** ($Y=1$) | True Positive (TP) | False Negative (FN) |
| **Actual: Legitimate** ($Y=0$) | False Positive (FP) | True Negative (TN) |

Each count is proportional to a joint probability. $TP \propto P(Y=1 \cap \hat{Y}=1)$, and so on for each cell.

What makes this framing useful is that every standard metric is a conditional probability derived from these joint probabilities.

### 2.2 Precision as a posterior probability

Precision answers the question: *"Given that the model flagged a transaction as fraud, how likely is it to actually be fraud?"*

$$P = \frac{TP}{TP + FP} = \frac{P(Y=1 \cap \hat{Y}=1)}{P(\hat{Y}=1)} = P(Y=1 \mid \hat{Y}=1)$$

In Bayesian language, Precision is the **posterior probability** of fraud given the model's positive prediction. It answers the question from the operator's perspective: when the alarm sounds, how often is it real?

### 2.3 Recall as a likelihood

Recall answers the question: *"Given that a transaction is actually fraudulent, how likely is it to be flagged?"*

$$R = \frac{TP}{TP + FN} = \frac{P(Y=1 \cap \hat{Y}=1)}{P(Y=1)} = P(\hat{Y}=1 \mid Y=1)$$

In Bayesian language, Recall is the **likelihood** of receiving a positive prediction given that the true class is positive. It answers the question from the fraud team's perspective: of all the frauds in the dataset, how many are we catching?

### 2.4 Why this framing matters

Precision and Recall condition on different things. Precision conditions on the model's output (what happened after prediction). Recall conditions on the ground truth (what was in the data). These are not symmetric quantities, and they cannot be combined into a single metric without an explicit weighting decision — which F1 makes silently.

A model with perfect Recall and zero Precision catches every fraud but flags every legitimate transaction too. A model with perfect Precision and zero Recall is entirely selective but misses most frauds. The right balance between these extremes depends on the cost of each error type — which varies by application. In fraud detection, missing a fraud is typically far more costly than a false alarm. The confusion matrix framing makes this concrete: FN and FP measure different conditional failure modes and warrant different costs.

---

## 3. Bayes' Theorem and the Base Rate Problem

Knowing that Precision is $P(Y=1 \mid \hat{Y}=1)$ and Recall is $P(\hat{Y}=1 \mid Y=1)$ invites the natural question: how are they related? The answer is Bayes' theorem. And the bridge between them is the **base rate** — the prevalence of fraud in the population.

### 3.1 The Bayesian derivation

By Bayes' theorem:

$$P(Y=1 \mid \hat{Y}=1) = \frac{P(\hat{Y}=1 \mid Y=1) \cdot P(Y=1)}{P(\hat{Y}=1)}$$

Translating into the metric vocabulary where $\pi = P(Y=1)$ is the base rate:

$$\text{Precision} = \frac{\text{Recall} \cdot \pi}{\text{Recall} \cdot \pi + (1 - \text{Specificity}) \cdot (1 - \pi)}$$

where Specificity $= P(\hat{Y}=0 \mid Y=0) = TN / (TN + FP)$.

This equation has a consequence that surprises many practitioners: **even a high-recall model can have very low precision if the base rate $\pi$ is very small**.

### 3.2 Numerical example: the base rate trap

Suppose a fraud detection model has Recall = 0.90 and False Positive Rate = 0.01. At first glance, this sounds like an excellent model: it catches 90% of frauds and incorrectly flags only 1% of legitimate transactions.

Now apply the Bayesian formula with a typical fraud base rate of $\pi = 0.001$ (0.1%):

$$\text{Precision} = \frac{0.90 \times 0.001}{0.90 \times 0.001 + 0.01 \times 0.999} \approx \frac{0.0009}{0.0009 + 0.00999} \approx 0.082$$

With only 8.2% precision, roughly 9 of every 10 transactions flagged by the model are legitimate. Despite excellent Recall and a small False Positive Rate, nearly 92% of the model's alerts are false alarms. This is not a failure of the model in any traditional sense — the model is genuinely good. It is a consequence of the base rate.

### 3.3 Class imbalance is a base rate problem

This numerical example reframes the "class imbalance problem" in an important way. Class imbalance is typically presented as a data problem requiring techniques such as oversampling, undersampling, or synthetic data generation. These techniques may help during training, but they do not resolve the fundamental issue.

The fundamental issue is that when $\pi$ is very small, Recall and Precision have a structurally adversarial relationship. A model that tries to achieve high Recall on a rare class will inevitably produce many False Positives relative to True Positives, because there are far more negatives to be mistakenly classified. This is not a bug in the data pipeline. It is Bayes' theorem operating on the population's true class distribution.

The right response to class imbalance in evaluation is not to pretend the imbalance does not exist (which accuracy does) or to average over both error types without acknowledging their costs (which F1 does). The right response is to explicitly incorporate the base rate and the asymmetric error costs into the metric itself.

---

## 4. The Asymmetry of Errors

The previous section established that class imbalance, interpreted as a base rate phenomenon, creates a structural tension between Precision and Recall. This section formalises what that tension costs, and derives the optimal decision threshold under asymmetric error costs.

### 4.1 The cost matrix

Every binary prediction system implicitly or explicitly encodes costs for each cell of the confusion matrix. In fraud detection:

- **True Positive ($C_{TP}$)**: A fraud is flagged. The investigation identifies and blocks it. Cost: investigation overhead, often small or offset by loss prevention.
- **True Negative ($C_{TN}$)**: A legitimate transaction is approved. Standard operating outcome. Cost: zero.
- **False Negative ($C_{FN}$)**: A fraud is approved. The loss falls on the institution (or the cardholder, depending on the liability regime). Typical costs include the transaction amount, dispute handling, chargeback fees, and reputational damage.
- **False Positive ($C_{FP}$)**: A legitimate transaction is blocked. The immediate cost is the lost transaction revenue. The indirect cost is customer friction, potential account abandonment, and lost relationship value.

Concretely: if the average transaction value at risk in a fraud is \$500 and the cost of a false alarm (investigation + customer service + lost transaction value) is \$15, then $C_{FN} \approx 500$ and $C_{FP} \approx 15$. The ratio is approximately 33:1.

These numbers are illustrative. The exact values vary by institution, card network rules, and liability agreements. What matters for this derivation is the **ratio** $C_{FN} / C_{FP}$, not the absolute values. (The experiments in Section 9 use $C_{FN} = 200$ and $C_{FP} = 5$ — a 40:1 ratio — as defined in `config.yaml`.)

### 4.2 The optimal threshold

A standard probabilistic classifier produces a score $\hat{p}(x) = P(Y=1 \mid x)$ for each transaction. At deployment, a threshold $\tau$ converts this score into a binary prediction: flag as fraud if $\hat{p}(x) \geq \tau$.

The conventional choice is $\tau = 0.5$. This choice minimises expected cost only when the two error types cost the same amount — which is almost never true in fraud detection.

To derive the optimal threshold, consider the expected cost of predicting fraud versus predicting legitimate for a transaction with score $\hat{p}(x)$:

- **Expected cost of predicting fraud** ($\hat{Y}=1$): $C_{FP} \cdot P(Y=0 \mid x) = C_{FP} \cdot (1 - \hat{p}(x))$
- **Expected cost of predicting legitimate** ($\hat{Y}=0$): $C_{FN} \cdot P(Y=1 \mid x) = C_{FN} \cdot \hat{p}(x)$

Predict fraud when the first is less than the second:

$$C_{FP} \cdot (1 - \hat{p}(x)) < C_{FN} \cdot \hat{p}(x)$$

Solving for $\hat{p}(x)$:

$$\hat{p}(x) > \frac{C_{FP}}{C_{FP} + C_{FN}} \equiv \tau^*$$

The optimal threshold is the ratio of the false positive cost to the total error cost. With $C_{FN} = 500$ and $C_{FP} = 15$:

$$\tau^* = \frac{15}{15 + 500} \approx 0.029$$

This means: flag any transaction for which the model assigns at least a 2.9% probability of fraud. The default threshold of 0.5 would demand 50% posterior probability — an astronomically conservative standard given that the prior probability is only 0.1%.

### 4.3 The default threshold encodes an implicit assumption

The threshold $\tau = 0.5$ is not neutral. It is the optimal threshold only when $C_{FP} = C_{FN}$. Using it in a context where this assumption is violated is equivalent to implicitly declaring that blocking a legitimate transaction and missing a fraud cost the same — a declaration that no fraud team would endorse explicitly but that many evaluation pipelines make by default.

This observation motivates the rest of the article. Once the optimal threshold is derived, the evaluation metric should be designed to reflect performance at and around $\tau^*$, not at the arbitrary $\tau = 0.5$.

---

## 5. F1 and the F-beta Family

F1 is the most widely used composite metric for imbalanced classification. This section derives it from first principles, explains what it actually measures, and states its limits honestly.

### 5.1 The harmonic mean

The arithmetic mean of two numbers weights each value linearly. The harmonic mean penalises extreme imbalance: if one value is very small, the harmonic mean is pulled strongly downwards.

$$F_1 = \frac{2 \cdot P \cdot R}{P + R} = \frac{2}{\frac{1}{P} + \frac{1}{R}}$$

This is precisely the harmonic mean of Precision and Recall. Its key property is that a model cannot achieve a high F1 by excelling at one metric while ignoring the other. A model with $P = 0.99, R = 0.01$ has $F_1 \approx 0.02$, not 0.50.

F1 is also expressible in terms of confusion matrix counts:

$$F_1 = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

This formulation makes the implicit weighting explicit: FP and FN are treated symmetrically. Each false prediction of either type contributes equally to the denominator.

### 5.2 What "equal weight to Precision and Recall" means

The phrase "equal weight" sounds balanced. In cost terms, it is not neutral. Weighting Precision and Recall equally means penalising False Positives and False Negatives identically — which is another way of saying $C_{FP} = C_{FN}$. This is exactly the same assumption embedded in the default threshold $\tau = 0.5$.

F1 is consistent with that threshold and that cost assumption. When those assumptions are violated — which they are in fraud detection — F1 provides a coherent answer to the *wrong question*.

### 5.3 The F-beta generalisation

The F-beta family generalises F1 by introducing a parameter $\beta$ that controls the relative importance of Recall over Precision:

$$F_\beta = \frac{(1 + \beta^2) \cdot P \cdot R}{\beta^2 \cdot P + R}$$

When $\beta = 1$, this reduces to F1. When $\beta > 1$, Recall is weighted more heavily — appropriate when missing a positive is more costly. When $\beta < 1$, Precision is weighted more heavily — appropriate when false alarms are more costly.

For fraud detection with $C_{FN} \gg C_{FP}$, values of $\beta$ in the range 2–5 are common, reflecting that missing a fraud matters more than generating a false alarm. However, the mapping from cost ratios to $\beta$ values is not direct — $\beta$ controls metric weighting, not cost weighting — so Precision@Recall (Section 8) provides a more principled connection to the actual operational constraint.

### 5.4 An honest statement about F1

F1 is a useful summary statistic. It avoids the accuracy trap on imbalanced data, penalises one-sided models, and provides a single number for model comparison. These are genuine advantages.

Its limitation is that its implicit cost assumption — equal costs for FP and FN — is rarely examined. Most practitioners who use F1 are not consciously endorsing equal costs; they are following a convention. The convention is fine when costs are similar. In fraud detection and similar high-asymmetry domains, it is not fine — and the experiments in Section 9 demonstrate that it changes model rankings.

---

## 6. The ROC Curve

The Receiver Operating Characteristic (ROC) curve provides a visual summary of classifier performance across all possible thresholds. Understanding its geometry and its limitations is essential for the argument that follows.

### 6.1 Definition and AUC-ROC

As the decision threshold $\tau$ sweeps from 1 (predict nothing) to 0 (predict everything), both the True Positive Rate (Recall) and the False Positive Rate change. The ROC curve plots $TPR(\tau)$ against $FPR(\tau)$ for all values of $\tau$:

$$TPR = \frac{TP}{TP + FN} = R \qquad FPR = \frac{FP}{FP + TN}$$

A random classifier traces the diagonal from $(0,0)$ to $(1,1)$. A perfect classifier reaches $(0,1)$ — zero false positives, all true positives captured — before any false positives occur.

The Area Under the ROC Curve (AUC-ROC) is the integral of this curve:

$$\text{AUC-ROC} = \int_0^1 TPR \, d(FPR)$$

AUC-ROC has a clean probabilistic interpretation: it is the probability that the model assigns a higher score to a randomly chosen positive (fraud) instance than to a randomly chosen negative (legitimate) instance. A value of 0.5 indicates random ranking; 1.0 indicates perfect discrimination.

### 6.2 Historical origin: Signal Detection Theory

ROC analysis originated not in machine learning but in **Signal Detection Theory**, developed during World War II to evaluate radar operators: how often does an operator correctly detect an enemy aircraft (TPR) versus report a false alarm (FPR)? The context is instructive because the operator faced the same asymmetric cost problem we are addressing — a missed aircraft had far graver consequences than a false alarm — and the theory already established that the optimal operating point on the ROC curve depends on costs and the base rate, the same quantities derived in Sections 3 and 4.

### 6.3 A critical caveat: ROC and class imbalance

Despite its probabilistic elegance, AUC-ROC has a well-documented limitation in the context of imbalanced datasets. The False Positive Rate $FPR = FP / (FP + TN)$ is normalised by the number of negatives. When the negative class is overwhelmingly large (as in fraud detection), even a large absolute number of false positives produces a small FPR.

This means that classifiers with substantially different precision profiles — and therefore different real-world costs — can appear indistinguishable on a ROC curve. The next section demonstrates this with the Precision-Recall curve, which does not have this masking property.

---

## 7. The Precision-Recall Curve

The Precision-Recall (PR) curve plots Precision against Recall as the threshold $\tau$ varies. Unlike the ROC curve, it has no FPR term and is therefore fully sensitive to false positive performance even when the negative class dominates.

### 7.1 Definition and AUC-PR

The PR curve plots $P(\tau)$ against $R(\tau)$:

$$\text{AUC-PR} = \int_0^1 P \, dR$$

A random (uninformative) classifier on a dataset with base rate $\pi$ produces a flat PR curve at height $\pi$ — because random predictions yield a fraction of True Positives proportional to the base rate, regardless of the threshold. This makes the baseline for AUC-PR approximately equal to $\pi$.

For fraud detection with $\pi = 0.001$, a random classifier achieves AUC-PR $\approx 0.001$. A useful model must achieve AUC-PR substantially above this. For ROC, by contrast, the random baseline is always 0.5, regardless of base rate — which makes it harder to visually assess whether a model is doing anything useful.

### 7.2 Why PR curves are more informative for fraud detection

Davis and Goadrich (2006) proved a formal relationship between ROC and PR curves: a model that dominates another in ROC space also dominates in PR space, but the converse is not true. Models can be separated in PR space that appear identical in ROC space.

This happens precisely in the imbalanced regime. When there are many more negatives than positives, the ROC curve's FPR axis compresses the distinction between classifiers that differ meaningfully in their precision profiles. The PR curve makes this distinction visible.

### 7.3 The trade-off shape and its business meaning

The shape of the PR curve encodes a specific type of business information. As Recall increases (the model is pushed to catch more frauds by lowering $\tau$), Precision typically decreases (more false alarms are generated). The PR curve is the trade-off frontier.

A concave PR curve indicates that the model can be operated at multiple points on the frontier by adjusting $\tau$. The business choice — how much precision can be sacrificed to achieve a given recall target — is a decision that belongs to the stakeholder, not the model. The PR curve provides the information needed to make that decision explicitly.

This observation sets up the final metric.

---

## 8. Precision@Recall — Pinning the Operating Point

Precision@Recall is the theoretical and narrative climax of this article. It is where the probabilistic machinery from Sections 2–4, the metric landscape from Sections 5–7, and the business reality of fraud detection converge.

### 8.1 Formal definition

Given a recall target $r \in (0, 1)$, Precision@Recall is defined as:

$$P\text{@}r = P(\tau_r) \quad \text{where} \quad \tau_r = \arg\min_\tau \lvert R(\tau) - r \rvert$$

In words: find the threshold $\tau_r$ that achieves recall level $r$; then report the precision at that threshold. On the PR curve, this is the Precision coordinate of the point where the curve crosses the horizontal line $R = r$.

### 8.2 The business parameter: r is not fixed, it is chosen

The value $r$ is not a technical parameter — it is a **business decision**. It encodes the answer to the question: *"What fraction of frauds are we willing to miss?"*

In a retail bank with aggressive fraud prevention, $r = 0.90$ might be the minimum acceptable recall: at least 90% of frauds must be caught. In a higher-friction, lower-volume environment, $r = 0.95$ or even $r = 0.99$ might be appropriate. Conversely, in a context where false positives are very costly (high-value transactions with sensitive customers), $r = 0.70$ might be chosen to preserve precision.

The crucial point is that this decision is made by the business, not by the algorithm. The role of the data scientist is to provide the PR curve (the trade-off frontier) and to compute Precision@$r$ for the stakeholder-defined $r$. This separation of roles — the model provides the frontier; the business chooses the operating point — is the most honest possible interface between machine learning and deployment.

### 8.3 Connection to cost-sensitive decision theory

Precision@Recall connects directly to the optimal threshold derivation in Section 4. Recall that $\tau^* = C_{FP} / (C_{FP} + C_{FN})$. The recall achieved at $\tau^*$ defines the cost-optimal operating recall:

$$r^* = R(\tau^*)$$

If the business has made an explicit cost specification (Section 4.1), then $r^*$ is the economically justified recall target, and $P\text{@}r^*$ is the precision the model achieves at the cost-optimal operating point. This is a more complete description of model performance than any single summary statistic over all thresholds.

### 8.4 F1 versus Precision@Recall: a key distinction

F1 is an unconstrained optimisation: it identifies the threshold that maximises the harmonic mean of Precision and Recall. This threshold may or may not coincide with the business's required operating point.

Precision@$r$ is a constrained formulation: it fixes the recall requirement and maximises precision subject to that constraint. This is the correct formulation when the recall requirement is non-negotiable (e.g., regulatory or contractual minimums on fraud detection rates).

The distinction matters for model ranking. Section 9 (Experiment E) demonstrates that ranking models by F1 and ranking them by Precision@$r$ can produce different orderings. The model that maximises F1 is not necessarily the model that delivers the highest precision at the required recall target. When a recall floor is fixed by the business, Precision@$r$ is the honest metric; F1 is not wrong, but it answers a different question.

---

## 9. Experiments

The following experiments were conducted on a synthetic imbalanced dataset (n = 100,000; 20 features, 5 informative; fraud rate = 0.1%). All code, configuration, and figures are reproducible by running `python scripts/run_all.py` from the repository root with a correctly configured environment. Seeds, hyperparameters, cost parameters, and dataset settings are specified in `config.yaml`.

All experiments use the same dataset split: 80% training, 20% test, stratified by class. Random seeds are fixed throughout. Models are trained on the training set; all metrics are reported on the held-out test set.

---

### Experiment A — The Accuracy Trap

**Theoretical claim**: On a severely imbalanced dataset, a degenerate classifier (predict always-negative) achieves very high accuracy but zero recall. Accuracy is therefore an invalid evaluation metric for fraud detection.

**Setup**: A `DummyClassifier` (always predicts the majority class) and a `LogisticRegression` model are evaluated on the synthetic fraud dataset (n = 100,000; fraud rate = 0.1%). Six metrics are reported: Accuracy, Precision, Recall, F1, AUC-ROC, and AUC-PR.

**Figure**:

![Experiment A: The Accuracy Trap — comparing DummyClassifier vs LogisticRegression across six evaluation metrics.](../figures/exp_a_accuracy_trap.png)

*Figure 1. Bar chart comparing a DummyClassifier (always predicts the majority class) with a LogisticRegression model on six metrics. The DummyClassifier achieves Accuracy = 0.999 but Precision, Recall, and F1 are all zero (highlighted with labelled annotations). Its AUC-ROC is 0.500 (random chance) and its AUC-PR matches the base rate (0.001). LogisticRegression sacrifices accuracy (0.926) in exchange for meaningful Recall (0.850) and AUC-PR (0.142).*

**Observation**: The DummyClassifier achieves accuracy close to $1 - \pi \approx 0.999$ while achieving Precision = 0, Recall = 0, and F1 = 0. Its AUC-ROC of 0.500 correctly identifies it as a random ranker, and its AUC-PR of 0.001 matches the base rate — the expected performance of a classifier with no discrimination ability. The LogisticRegression model accepts a reduction in accuracy (from 0.999 to 0.926) in exchange for meaningful Recall (0.850) and an AUC-PR two orders of magnitude above the baseline.

**Theoretical connection**: The DummyClassifier exploits the base rate problem (Section 3.3). Its accuracy equals $1 - \pi$, which is very high when fraud is rare. The F1 score of zero correctly identifies it as useless — not because F1 is the optimal metric, but because it conditions on the positive class (through Recall and Precision), which accuracy does not. Notably, AUC-PR provides the most honest single-number summary: the DummyClassifier scores 0.001 (the base rate) while LogisticRegression scores 0.142.

Having established that accuracy is unreliable, the next question is whether the metrics that replace it — AUC-ROC and AUC-PR — always agree on which model is better.

---

### Experiment B — ROC vs. Precision-Recall Curves

**Theoretical claim**: In the imbalanced regime, models that appear similar on a ROC curve are revealed to be substantially different on a Precision-Recall curve. Moreover, the model ranking itself can differ between the two spaces.

**Setup**: Three models are evaluated — `LogisticRegression`, `RandomForestClassifier`, and a weak `DecisionTreeClassifier` (depth = 2) — on the synthetic fraud dataset (n = 100,000; fraud rate = 0.1%; 5 informative features out of 20). Both ROC and PR curves are plotted with identical square aspect ratios for direct visual comparison. Curves include markers for distinction in greyscale printing.

**Figure**:

![Experiment B — Part 1: ROC curves for three models.](../figures/exp_b_roc_curves.png)

*Figure 2. ROC curves for three classifiers. AUC-ROC values are compressed: Logistic Regression (0.951), Random Forest (0.917), Weak Classifier (0.876). All models appear competitive in ROC space. The ranking suggests Logistic Regression is the best model.*

**Figure**:

![Experiment B — Part 2: Precision-Recall curves for three models.](../figures/exp_b_pr_curves.png)

*Figure 3. Precision-Recall curves for the same three classifiers. The horizontal baseline marks the base rate (annotated with arrow). AUC-PR values are far more spread: Random Forest (0.426), Weak Classifier (0.409), Logistic Regression (0.122). The ranking is reversed: the model that appeared best in ROC space is worst in PR space. Vertical lines mark the recall targets used in Experiments D and E.*

**Observation**: The ROC and PR curves produce **opposite model rankings**. In ROC space, Logistic Regression leads (AUC-ROC = 0.951); in PR space, it is the weakest model (AUC-PR = 0.122). This inversion occurs because the linear model struggles with the non-linear decision boundary created by the reduced feature set (5 of 20 features are informative), but its discrimination across the full score distribution — which ROC measures — remains strong. The PR curve, which is sensitive to precision at the relevant operating range, reveals this weakness.

**Theoretical connection**: This result is a stronger-than-expected confirmation of Section 7.2. Davis and Goadrich (2006) proved that models can be separated in PR space while appearing identical in ROC space. Here we observe the more dramatic case: models are **inversely ranked** across the two spaces. The ROC curve's FPR axis normalises false positives by the number of negatives, which are overwhelmingly numerous (99,900 legitimate transactions). A large absolute increase in false positives — which directly affects precision — produces only a marginal increase in FPR.

The ranking inversion shows that *which* metric we use matters. But even within a single model, the choice of *threshold* produces dramatically different operating profiles — as the next experiment demonstrates.

---

### Experiment C — Threshold Selection

**Theoretical claim**: The default threshold of $\tau = 0.5$ is rarely optimal under asymmetric costs. The cost-optimal threshold $\tau^*$ shifts the operating point towards higher Recall at the cost of lower Precision, reducing expected cost.

**Setup**: A `LogisticRegression` model is evaluated at three operating points: the default threshold ($\tau = 0.5$), the F1-optimal threshold (found by grid search), and the cost-optimal threshold $\tau^* = C_{FP} / (C_{FP} + C_{FN})$ using the cost values in `config.yaml` ($C_{FP}$ = \$5, $C_{FN}$ = \$200). A threshold sweep plot and confusion matrices for all three thresholds are shown.

**Figure**:

![Experiment C — Part 1: Threshold sweep showing Precision, Recall, and F1 as a function of threshold.](../figures/exp_c_threshold_sweep.png)

*Figure 4. Threshold sweep for a LogisticRegression model. Precision, Recall, and F1 are plotted as functions of the decision threshold. Vertical lines mark three operating points: the default threshold ($\tau = 0.5$), the F1-optimal threshold ($\tau = 0.990$), and the cost-optimal threshold ($\tau^* = 0.024$). Precision and F1 remain near zero for most of the threshold range and only rise steeply above $\tau = 0.9$, indicating that on this highly imbalanced problem, the model's score distribution is concentrated near 1.0 for fraud cases.*

**Figure**:

![Experiment C — Part 2: Confusion matrices at three different thresholds.](../figures/exp_c_confusion_matrices.png)

*Figure 5. Confusion matrices (row-normalised) at the three operating points. At $\tau = 0.5$ (left): Recall = 0.850 but Precision = 0.011 (1,470 false positives). At $\tau = 0.990$ / F1-optimal (centre): Recall drops to 0.400 and Precision rises to 0.078 (94 false positives). At $\tau^* = 0.024$ / cost-optimal (right): Recall increases to 0.900 with Precision = 0.003 (5,824 false positives). Each matrix shows both raw counts and row-normalised rates.*

**Observation**: The three thresholds produce qualitatively different confusion matrices. The default threshold ($\tau = 0.5$) achieves 85% recall but at the cost of 1,470 false positives — a Precision of just 1.1%. The F1-optimal threshold ($\tau = 0.990$) reduces false positives drastically (to 94) but sacrifices recall to 40%, missing most frauds. The cost-optimal threshold ($\tau^* = 0.024$) maximises recall (90%) at the expense of 5,824 false positives — the correct trade-off given that each missed fraud costs 40 times more than a false alarm.

A notable result is that the F1-optimal threshold is 0.990 — far above the default of 0.5. This is not an error; it reflects the model's score distribution under extreme class imbalance. The vast majority of scores are near zero (for legitimate transactions), and the F1 harmonic mean peaks only when the threshold is high enough to substantially reduce false positives.

**Theoretical connection**: This is a direct empirical confirmation of Section 4.3: the default threshold encodes the implicit assumption $C_{FP} = C_{FN}$, which is false here. The cost-optimal threshold follows from the derivation in Section 4.2 and produces a qualitatively different confusion matrix. The gap between the F1-optimal and cost-optimal thresholds (0.990 vs. 0.024) illustrates how far the F1 criterion diverges from the economically justified operating point.

The threshold experiment shows that the operating point matters enormously. In practice, however, the business does not choose a threshold directly — it chooses a recall target. The next experiment translates recall targets into precision and cost.

---

### Experiment D — Precision@Recall at Business-Defined Targets

**Theoretical claim**: Different business contexts imply different recall requirements, and the precision achievable at each requirement is a model property that should be evaluated explicitly. The trade-off has direct financial consequences.

**Setup**: A `RandomForestClassifier` (the best model from Experiment B by AUC-PR) is evaluated at five recall targets: $r \in \lbrace 0.75, 0.80, 0.85, 0.90, 0.95 \rbrace$. Precision, false positive count, and false negative count are computed at each operating point. The expected business cost ($C_{FP}$ = \$5, $C_{FN}$ = \$200) is annotated on the figure.

**Figure**:

![Experiment D: Precision@Recall at five business-defined recall targets with cost annotations.](../figures/exp_d_precision_at_recall.png)

*Figure 6. Left panel: Precision at each recall target. Precision declines from 0.072 at r = 0.75 to 0.011 at r = 0.95. Right panel: False positive and false negative counts, with expected cost annotations above each group. At r = 0.75, the model generates 193 false positives and 5 false negatives (total cost: \$1,965). At r = 0.90, false positives jump to 1,480 and the cost quadruples to \$8,000. The transition between r = 0.85 and r = 0.90 marks a sharp inflection point in the cost curve.*

**Observation**: Precision declines steadily from r = 0.75 to r = 0.85 (7.2% to 5.7%), but drops sharply at r = 0.90 (to 0.8%) as the model exhausts its high-confidence positive predictions and begins flagging low-confidence transactions. The cost analysis reveals an inflection point: below r = 0.85, the total expected cost is approximately \$2,000; above r = 0.90, it jumps to \$8,000. This non-linearity means that the business decision between 85% and 90% recall has a 4x cost impact — information that is invisible in a summary metric like AUC-PR.

**Theoretical connection**: This experiment operationalises Section 8.2: $r$ is a business decision, and Precision@$r$ is the metric that reflects model quality at that decision point. The cost annotations connect directly to the cost matrix framework in Section 4.1, translating abstract confusion matrix counts into monetary terms. The full PR curve (Figure 3) is the complete trade-off frontier; Precision@$r$ is the specific operating point on that frontier.

Experiment D evaluated a single model at multiple recall targets. The final experiment asks the decisive question: when multiple models compete, does the choice of metric change which model wins?

---

### Experiment E — Metric Choice Changes Model Rankings

**Theoretical claim**: Ranking models by F1 and ranking them by Precision@$r$ can produce different orderings. When a recall floor is fixed by the business, the F1-based ranking may recommend the wrong model.

**Setup**: Five model variants are trained — three Logistic Regression configurations (balanced, C = 0.01, C = 10) and two Random Forest configurations (depth = 5, depth = 15) — all with balanced class weights. Each model is ranked by seven metrics: F1 (best achievable across thresholds), AUC-PR, and Precision@$r$ for $r \in \lbrace 0.75, 0.80, 0.85, 0.90, 0.95 \rbrace$. Results are displayed as an annotated heatmap with per-cell rankings and divergence highlighting.

**Figure**:

![Experiment E: Model ranking comparison across seven evaluation metrics.](../figures/exp_e_ranking_comparison.png)

*Figure 7. Heatmap of model scores across seven metrics. Each cell shows the metric value and ordinal rank (e.g., #1, #2). Orange-bordered cells highlight where the top-1 model differs from the F1 ranking. RF (depth = 15) ranks #1 by F1 (0.519); RF (depth = 5) ranks #1 by P@75R (0.146). The divergence is concentrated at the P@75R column — the models that achieve the best F1 trade-off are not the same models that deliver the highest precision at a fixed recall floor.*

**Observation**: The ranking diverges at the P@75R column. RF (depth = 15) is the best model by F1 (0.519) and AUC-PR (0.449), but RF (depth = 5) delivers nearly twice the precision at 75% recall (0.146 vs. 0.074). This means that if the business requires at least 75% of frauds to be caught, the F1-recommended model would produce roughly twice as many false alarms as the P@75R-recommended model. The three Logistic Regression variants are consistently weaker across all metrics, confirming that the ranking divergence is not an artefact of weak models — it occurs between the two best-performing models in the pool.

**Theoretical connection**: This experiment is the empirical core of the article's thesis. It ties together Sections 5 through 8: F1 (Section 5) and Precision@Recall (Section 8) answer different questions, and when those questions have different answers, the choice of metric determines which model gets deployed. The heatmap makes this concrete: the same set of models, the same dataset, the same train-test split — but a different metric selects a different model. This is not noise; it is information about the misalignment between F1's implicit equal-cost assumption and the business's actual operating constraints.

---

## 10. A Framework for Choosing the Right Metric

The theory sections established the mathematical foundations; the experiments provided empirical evidence. This section synthesises a practical decision framework.

### 10.1 The decision table

The choice of evaluation metric follows from answering a small set of questions about the deployment context:

| Question | Answer | Recommended Metric |
|---|---|---|
| Are class distributions balanced? | Yes | Accuracy, F1 |
| Are class distributions balanced? | No | Continue below |
| Is a business recall floor defined? | Yes | Precision@r, AUC-PR |
| Is a business recall floor defined? | No | AUC-PR, F1 |
| Are error costs explicitly estimated? | Yes | Cost-optimal threshold + Precision@r\* |
| Are error costs explicitly estimated? | No | AUC-PR (ranking); F1 at F1-optimal threshold |
| Is model ranking the goal (no fixed threshold)? | Yes | AUC-PR |
| Is threshold selection the goal? | Yes | PR curve + cost-optimal threshold |

### 10.2 Specific recommendations for fraud detection

Given the structural properties of fraud detection — severe class imbalance, asymmetric error costs, and a regulatory or commercial minimum recall requirement — the recommended approach is:

1. **During development**: rank models by AUC-PR. This provides a threshold-free comparison that is sensitive to performance on the positive (fraud) class.
2. **During threshold selection**: compute the cost-optimal threshold $\tau^* = C_{FP} / (C_{FP} + C_{FN})$ from the institution's cost estimates.
3. **For deployment evaluation**: report Precision@$r$ where $r$ is the business-defined recall floor. This is the number that directly answers the operational question: at the required detection rate, what fraction of our alerts are real frauds?
4. **For communication with stakeholders**: express results in terms of FP and FN counts and their associated costs, not percentages alone. "At 90% recall, we generate approximately 47 false alarms per day at a combined investigation cost of $X" is more actionable than "P@90R = 0.15."

### 10.3 Common mistakes and how to avoid them

**Mistake 1: Using accuracy on an imbalanced dataset.** Experiment A demonstrates that accuracy rewards models that ignore the minority class. Always check whether a DummyClassifier would achieve competitive accuracy before using accuracy as a metric.

**Mistake 2: Using AUC-ROC as the primary metric for imbalanced datasets.** AUC-ROC is a useful complement but can mask large differences in precision profiles (Experiment B). Use AUC-PR as the primary ranking metric.

**Mistake 3: Evaluating at the default threshold of 0.5.** The default threshold is only optimal when costs are symmetric and the base rate is 0.5. In fraud detection, $\tau = 0.5$ typically produces near-zero recall (Experiment C). Always report which threshold was used and why.

**Mistake 4: Choosing a metric without a stated business requirement.** F1 is a reasonable default *when no other information is available*. If a recall floor is known, Precision@$r$ is more informative. If costs are known, cost-optimal threshold analysis is more informative. The metric should be matched to the information available.

### 10.4 The closing principle

The right metric is a business decision encoded in mathematics. Every evaluation metric encodes assumptions about costs, class distributions, and operating requirements. Making those assumptions explicit — rather than inheriting them from convention — is the most important practice an ML practitioner can adopt.

The mathematical tools to do this are not exotic. They are Bayes' theorem, conditional probability, and a cost matrix. This article has assembled them bottom-up. The assembly was the point.

---

## 11. Conclusion

### What was demonstrated

This article started from a concrete question — why F1 in fraud detection? — and traced its answer through four linked bodies of knowledge:

1. **Conditional probability** revealed that Precision and Recall measure qualitatively different things: the posterior probability of fraud given a positive prediction, and the likelihood of a positive prediction given fraud. Combining them requires making a weighting decision explicit.

2. **Bayes' theorem** connected the base rate to the tension between Precision and Recall, showing that class imbalance is a base rate problem and that any metric which ignores the base rate cannot fully characterise model performance in imbalanced settings.

3. **Cost-sensitive decision theory** derived the optimal threshold from first principles, showing that the conventional threshold of 0.5 is a special case that holds only when the two error types cost the same.

4. **The metric landscape** — ROC, PR, and Precision@Recall — mapped the space of evaluation choices and demonstrated empirically that metric choice changes model rankings and deployment decisions.

### Practical takeaways

These takeaways are conditional. Apply them when the conditions are met.

- When class imbalance is severe, never use accuracy as the primary evaluation metric.
- When comparing models without a fixed threshold, prefer AUC-PR over AUC-ROC.
- When a business recall floor is defined, evaluate with Precision@$r$ at that floor.
- When error costs are estimated, derive the cost-optimal threshold and evaluate at it.
- When reporting results to stakeholders, translate metric values into FP/FN counts and their associated business costs.
- When choosing between two models, ask which metric best represents the deployment context before consulting the leaderboard.

### What comes next

For the reader: the natural extension of this work is to dynamic and sequential settings — fraud streams with non-stationary distributions, where the base rate changes over time and the optimal threshold must be recalibrated. The Bayesian framework developed here extends naturally to online learning and sequential decision-making.

For the model and dataset used here: the experiments ran on a static snapshot. Production fraud detection operates on evolving transaction data with concept drift, label delay (chargebacks may arrive weeks after the transaction), and adversarial feedback loops. The metric framework is correct for the static case; extending it to the dynamic case is a richer problem.

For the author: this article was written in parallel with learning. The next iteration will incorporate calibration — the degree to which the model's probability estimates $\hat{p}(x)$ are trustworthy — because the cost-optimal threshold derivation in Section 4 depends entirely on $\hat{p}(x)$ being a well-calibrated posterior probability.

---

## References

[1] Bayes, T. (1763). *An Essay towards Solving a Problem in the Doctrine of Chances*. Philosophical Transactions of the Royal Society of London, 53, 370–418.

[2] Davis, J. & Goadrich, M. (2006). The Relationship Between Precision-Recall and ROC Curves. *Proceedings of the 23rd International Conference on Machine Learning (ICML)*, 233–240. https://doi.org/10.1145/1143844.1143874

[3] Fawcett, T. (2006). An Introduction to ROC Analysis. *Pattern Recognition Letters*, 27(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010

[4] Saito, T. & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. *PLOS ONE*, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

[5] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

[6] Green, D.M. & Swets, J.A. (1966). *Signal Detection Theory and Psychophysics*. Wiley. *(Original formulation of ROC analysis from Signal Detection Theory.)*

[7] Dal Pozzolo, A., Caelen, O., Johnson, R.A., & Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. *2015 IEEE Symposium Series on Computational Intelligence*, 159–166. https://doi.org/10.1109/SSCI.2015.33

[8] ULB Machine Learning Group (2018). *Credit Card Fraud Detection* [Dataset]. Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud — *Note: this article uses a synthetic dataset rather than the ULB dataset. The synthetic design allows full control over the number of informative features, base rate, and class separation, making metric behaviour easier to isolate and reproduce. The ULB dataset is cited as the canonical real-world reference for imbalanced fraud detection.*

---

*Source code and data: [github.com/brunoramosmartins/precision-recall-fraud](https://github.com/brunoramosmartins/precision-recall-fraud)*

*Notation glossary: [article/notation.md](notation.md)*

*BibTeX references: [article/references.bib](references.bib)*
