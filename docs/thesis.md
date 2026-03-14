# Thesis, Scope & Reader Contract

> **Status:** Draft — refine before starting Phase 1.

---

## Final Thesis

> *"Not all errors cost the same: how Bayes' theorem explains why precision@recall is a more honest evaluation metric than F1 when the cost of missing a fraud far exceeds the cost of a false alarm."*

**Core claim:** F1 treats all errors as equally costly and implicitly assumes a balanced class distribution. In fraud detection, neither assumption holds. Precision@recall, grounded in Bayes' theorem and cost-sensitive decision theory, is a more honest metric because it makes the cost structure and the operating constraint explicit — not hidden.

**Evidence that will support it:**
1. Formal derivation showing Precision as a posterior probability and Recall as a likelihood — both conditional probabilities connected by Bayes' theorem.
2. Mathematical proof that high recall can coexist with very low precision when the base rate (fraud prevalence) is low.
3. Derivation of the optimal decision threshold under asymmetric costs — showing that default threshold 0.5 is only optimal when costs are symmetric and classes are balanced.
4. Empirical experiments on a fraud dataset demonstrating that F1 and precision@recall can rank models differently, and that the difference matters for business decisions.

---

## Scope Boundaries

### In Scope
- Formal definitions of Precision, Recall, F1, F-beta, AUC-ROC, AUC-PR, Precision@recall.
- Bayesian interpretation of the confusion matrix (Precision as posterior, Recall as likelihood).
- Cost-sensitive decision theory: cost matrix, optimal threshold derivation, business framing.
- ROC curve vs PR curve: what each reveals, what each hides, and why.
- Reproducible experiments on a fraud detection dataset.
- A practical framework for choosing the right metric given the problem's cost structure.

### Out of Scope
- Gradient boosting internals (XGBoost, LightGBM, CatBoost).
- Model training and hyperparameter tuning (models are a means to an end, not the focus).
- Feature engineering for fraud detection.
- Multi-class classification.
- Deep learning or neural network architectures.
- Real-time fraud detection systems and infrastructure.
- SMOTE and other resampling techniques (mentioned in passing, not explored).

---

## Narrative Thread

**Opening — the catalyst:**
A technical case interview revealed that F1 had been chosen on autopilot — without consciously connecting it to business cost, error asymmetry, or threshold choice. This is framed as intellectual curiosity, not failure. The question it raises: *when costs are not symmetric and classes are not balanced, which metric is honest?*

**Core — the mathematical insight:**
Precision and Recall are not arbitrary ML jargon. They are conditional probabilities. Their relationship is governed by Bayes' theorem. Class imbalance is the base rate problem in disguise. Understanding this changes how you choose, interpret, and communicate evaluation metrics.

**Close — the practical framework:**
A structured decision framework for choosing the right metric given the business cost structure. The close is both technical and honest: no metric is universally best. The right metric is a business decision encoded in mathematics.

---

## Reader Persona

The primary reader is a data scientist or ML engineer with 1–4 years of experience who has used F1, AUC-ROC, and scikit-learn's classification report routinely, but has not rigorously connected these metrics to probability theory or business cost structures. They have encountered class imbalance problems. They may have used precision@recall in a job description or technical interview without fully internalizing why it exists. They are comfortable with Python and basic statistics. They respond well to rigorous derivations when the intuition is scaffolded first, and they appreciate honest framing of limitations over confident oversimplification.

A secondary reader is a technical recruiter or hiring manager who will skim the introduction, look at the figures, and read the conclusion — seeking evidence of depth, communication skill, and honest reasoning.

---

## Tone

- **Theory sections:** technically rigorous. Derivations are shown, not waved away. Notation is consistent and defined.
- **Narrative sections (introduction, conclusion):** honest, direct, and personal without being confessional. First person is appropriate in the introduction.
- **Experiment sections:** precise and factual. Claims are bounded by the dataset and setup.
- **General:** no hedging for its own sake. No overclaiming. Every conclusion is supported by a derivation or figure.

---

## Promise to the Reader

After reading this article, you will:

1. Understand why Precision and Recall are conditional probabilities and how Bayes' theorem explains the relationship between them.
2. Know why high AUC-ROC does not guarantee high precision in imbalanced problems — and how to spot this in practice.
3. Be able to derive the optimal decision threshold under asymmetric costs and explain why default threshold 0.5 is almost never optimal in fraud detection.
4. Understand the precise difference between F1 (unconstrained optimization) and Precision@recall (constrained optimization anchored to a business requirement).
5. Have a practical decision framework for choosing between Accuracy, F1, AUC-ROC, AUC-PR, and Precision@recall based on the cost structure of your problem.

---

## Publishing Plan

| Format | Output | Channel | Priority |
|--------|--------|---------|----------|
| Markdown | `article/metrics-that-matter.md` | [Personal portfolio site](https://brunoramosmartins.github.io) (GitHub Pages, rendered via custom Python pipeline) | Primary |
| HTML | `docs/index.html` | This repository's GitHub Pages (landing page / project card — links to full article) | Secondary |
| LaTeX / PDF | `article/latex/metrics-that-matter.tex` | Repository download + future academic submissions | Optional |
| LinkedIn post | Summary only — no full article | LinkedIn | Promotion |

**Format decision rationale:**

Markdown is the canonical source. It is version-controlled, diff-friendly, and consumed directly by the portfolio site's rendering pipeline. It does not require a build step to read.

The `docs/index.html` in this repository is a standalone landing page for the GitHub repository itself — it is not the article. It provides a project summary and links to the full article on the portfolio site. This avoids duplication while maintaining a professional presence at the repo's GitHub Pages URL.

The LaTeX version, if produced, will be generated from the Markdown source using **Pandoc** to avoid writing the content twice:
```bash
pandoc article/metrics-that-matter.md \
  --bibliography article/latex/references.bib \
  --citeproc \
  -o article/latex/metrics-that-matter.pdf
```
The LaTeX source (`article/latex/metrics-that-matter.tex`) will be committed. The compiled PDF may also be committed for direct download. LaTeX build artifacts are excluded by `.gitignore`.
