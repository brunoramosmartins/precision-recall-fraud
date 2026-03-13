# Article Outline — Not All Errors Cost the Same

> **Status:** Draft — sections are placeholders until Phase 4 writing begins.

---

## Section Map


| #   | Section Title                                  | Type                | Bayes?          | Business framing? |
| --- | ---------------------------------------------- | ------------------- | --------------- | ----------------- |
| 0   | Abstract                                       | Summary             | —               | —                 |
| 1   | Introduction                                   | Narrative           | hook            | hook              |
| 2   | The Confusion Matrix — A Probabilistic Reading | Theory-heavy        | yes             | no                |
| 3   | Bayes' Theorem and the Base Rate Problem       | Theory-heavy        | yes (central)   | light             |
| 4   | The Asymmetry of Errors                        | Theory + framing    | yes (applied)   | yes (central)     |
| 5   | F1 and the F-beta Family                       | Theory-heavy        | no              | partial           |
| 6   | The ROC Curve                                  | Theory + experiment | no              | yes (caveat)      |
| 7   | The Precision-Recall Curve                     | Theory + experiment | no              | yes (central)     |
| 8   | Precision@Recall — Pinning the Operating Point | Theory + experiment | yes (ties back) | yes (climax)      |
| 9   | Experiments                                    | Experiment-driven   | no              | yes               |
| 10  | A Framework for Choosing the Right Metric      | Framework           | no              | yes (close)       |
| 11  | Conclusion                                     | Narrative           | summary         | summary           |
| 12  | References                                     | —                   | —               | —                 |


---

## Section Descriptions

### Abstract (write last)

150–250 words. Problem → mathematical insight → experiments → conclusion. One sentence per major finding.

---

### 1. Introduction (write second-to-last)

**Type:** Narrative
**Opens with:** A technical case interview asked which metric to use for a fraud detection model. The answer given — F1 — was not wrong, but it was chosen on autopilot, without connecting it to business cost or error asymmetry. This article is the study that should have happened before that interview.
**Closes with:** The guiding question: *when costs are not symmetric and classes are not balanced, which metric is honest?* Statement of thesis and roadmap of what the article demonstrates.

---

### 2. The Confusion Matrix — A Probabilistic Reading

**Type:** Theory-heavy
**Bayes connection:** Yes — sets up the probabilistic notation for Section 3.
**Summary:** The four cells of the confusion matrix are events in a probability space. Precision is a posterior probability P(Y=1 | Ŷ=1). Recall is a likelihood P(Ŷ=1 | Y=1). This framing is not academic decoration — it determines how the metrics respond to class imbalance.

---

### 3. Bayes' Theorem and the Base Rate Problem

**Type:** Theory-heavy (the mathematical core of the article)
**Bayes connection:** Central — the full derivation lives here.
**Summary:** Bayes' theorem rewrites Precision as a function of Recall and the base rate (fraud prevalence). The consequence: high recall does not imply high precision when the base rate is low. Class imbalance is the base rate problem in disguise. A numerical example (10,000 transactions, 0.1% fraud) makes this concrete.

---

### 4. The Asymmetry of Errors

**Type:** Theory + business framing
**Bayes connection:** Applied — the cost-weighted threshold is a direct application of Bayesian decision theory.
**Summary:** Not all errors cost the same. A cost matrix formalizes this. The optimal decision threshold under asymmetric costs is derived from first principles: τ* = C_FP / (C_FP + C_FN). Default threshold 0.5 is implicitly assuming symmetric costs and a 50/50 base rate — neither holds in fraud detection.

---

### 5. F1 and the F-beta Family

**Type:** Theory-heavy
**Summary:** F1 is the harmonic mean of Precision and Recall. It treats both as equally important. The harmonic mean penalizes extreme imbalance between them more than the arithmetic mean would. F-beta generalizes this by giving weight β² to Recall relative to Precision. F1 is a useful summary but makes implicit assumptions that should be made explicit before using it as a primary metric.

---

### 6. The ROC Curve

**Type:** Theory + experiment (Experiment B contributes here)
**Summary:** The ROC curve traces (FPR, TPR) pairs across all thresholds. AUC-ROC has a clean probabilistic interpretation: P(model ranks a fraud higher than a legitimate transaction). Historical context: Signal Detection Theory, WWII radar operators. Honest caveat: AUC-ROC can be misleadingly optimistic under severe class imbalance.

---

### 7. The Precision-Recall Curve

**Type:** Theory + experiment (Experiment B contributes here)
**Summary:** The PR curve traces (Recall, Precision) pairs across all thresholds. AUC-PR baseline is the base rate — not 0.5 like ROC. For imbalanced problems, PR curves reveal differences between models that ROC curves hide. The trade-off shape reflects the fundamental tension between catching more fraud and generating fewer false alarms.

---

### 8. Precision@Recall — Pinning the Operating Point

**Type:** Theory + experiment (Experiment D is the core evidence)
**Bayes connection:** Yes — ties back to the base rate and cost-structure discussions in Sections 3 and 4.
**Summary:** Precision@r is a specific point on the PR curve: the precision achievable when recall equals the business-required minimum r. The value of r is not fixed — it is a business parameter reflecting the tolerable miss rate. This is a constrained optimization view vs F1's unconstrained view. This section is the theoretical and narrative climax: all prior sections converge here.

---

### 9. Experiments

**Type:** Experiment-driven
**Summary:** Five experiments connecting theory to evidence:

- **Exp A — The Accuracy Trap:** DummyClassifier vs Logistic Regression. Shows accuracy is meaningless for imbalanced data.
- **Exp B — ROC vs PR Side by Side:** Three models. Differences invisible in ROC become visible in PR space.
- **Exp C — Threshold Selection:** Default 0.5 vs F1-optimal vs cost-optimal threshold. Confusion matrices at each.
- **Exp D — Precision@Recall as Operating Point:** [P@0.75](mailto:P@0.75), [P@0.80](mailto:P@0.80), [P@0.85](mailto:P@0.85), [P@0.90](mailto:P@0.90) with their confusion matrices. The business trade-off in numbers.
- **Exp E — Model Ranking Reversal:** At least one model pair where F1 and P@75recall give opposite rankings.

---

### 10. A Framework for Choosing the Right Metric

**Type:** Framework / decision guide
**Summary:** Decision table: for each problem structure (balanced, imbalanced with no cost spec, imbalanced with recall requirement, ranking quality, calibration), the recommended metric and why. Fraud detection column highlighted. Closes with: "the right metric is a business decision encoded in mathematics."

---

### 11. Conclusion

**Type:** Narrative
**Summary:** Restates the thesis and what was demonstrated. Five practical takeaways, each conditional and honest. What comes next — for the reader (how to apply this framework), and optionally for the author.

---

### 12. References

Minimum citations:

- Bayes (1763)
- Davis & Goadrich (2006) — PR vs ROC
- Fawcett (2006) — ROC analysis introduction
- Saito & Rehmsmeier (2015) — PR more informative than ROC for imbalance
- Hastie, Tibshirani, Friedman (2009) — ESL
- scikit-learn documentation
- Kaggle Credit Card Fraud dataset

---

## Bayes Connection Map


| Section | How Bayes appears                                                                                                            |
| ------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 2       | Precision = P(Y=1 | Ŷ=1) is established as a posterior probability                                                           |
| 3       | Full Bayes derivation: Precision = Recall × P(fraud) / P(Ŷ=1)                                                                |
| 4       | Cost-sensitive threshold τ* = C_FP / (C_FP + C_FN) is Bayesian decision theory                                               |
| 8       | Precision@recall fixes the recall constraint and asks for the best posterior precision — Bayes and business framing converge |


---

## Business Decision Framing Map


| Section | Business framing                                                                                |
| ------- | ----------------------------------------------------------------------------------------------- |
| 1       | Interview catalyst; "which metric is honest?"                                                   |
| 3       | Low base rate → low precision → frustrated customers or financial loss                          |
| 4       | Cost matrix; τ* derivation; blocking legitimate transactions vs letting frauds through          |
| 8       | Recall target as a business parameter; the number in "P@75recall" is the operational constraint |
| 9       | Experiment D: the confusion matrix at each recall target in dollar terms                        |
| 10      | Full decision framework; "the right metric is a business decision"                              |
| 11      | Takeaways are conditional and business-framed                                                   |


