# Not All Errors Cost the Same

### Bayesian Foundations of Evaluation Metrics for Fraud Detection — from F1 to Precision@Recall

> A technical case interview revealed I had been using F1 on autopilot — without consciously connecting it to business cost, error asymmetry, or threshold choice. This article is the study I should have done before that interview.

---

## What This Is

A technical article + reproducible experiments on evaluation metrics for imbalanced binary classification. The central argument:

> *Precision and Recall are conditional probabilities. Their relationship is governed by Bayes' theorem. Class imbalance is the base rate problem in disguise. F1 treats all errors as equally costly; Precision@recall does not. In fraud detection, the difference matters.*

**Covered:**
- Confusion matrix as a probability space — Precision as a posterior, Recall as a likelihood.
- Bayes' theorem connecting Precision, Recall, and base rate.
- Cost-sensitive decision theory: optimal threshold derivation under asymmetric costs.
- F1 and the F-beta family — what "equal weight" actually means.
- ROC curve vs Precision-Recall curve — what each reveals and what each hides.
- Precision@recall — definition, business interpretation, and why it beats F1 for fraud detection.
- 5 reproducible experiments connecting theory to evidence.

**Not covered:** model training, feature engineering, gradient boosting internals, deep learning, SMOTE.

---

## Repository Structure

```
precision-recall-fraud/
├── article/
│   ├── metrics-that-matter.md   <- Canonical article source (Markdown)
│   ├── notation.md              <- Notation glossary
│   ├── references.bib           <- BibTeX references
│   └── latex/                   <- Optional LaTeX/PDF version (via Pandoc)
├── docs/                        <- GitHub Pages: landing page + project docs
├── src/                         <- Reusable Python modules
├── scripts/                     <- Experiment scripts (one per experiment)
├── notebooks/                   <- Exploratory analysis (clearly labeled)
├── figures/                     <- Generated figures (300dpi PNG)
├── data/                        <- Data README + synthetic data
├── notes/                       <- Phase study notes
├── config.yaml                  <- All parameters, seeds, and paths
└── requirements.txt             <- Python dependencies
```

---

## Reproduce Everything

```bash
# 1. Clone
git clone https://github.com/brunoramosmartins/precision-recall-fraud.git
cd precision-recall-fraud

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate      # Git Bash / macOS / Linux
# .venv\Scripts\activate       # Windows CMD / PowerShell
pip install -r requirements.txt

# 3. Download the Kaggle dataset (see data/README.md for instructions)

# 4. Run all experiments and regenerate all figures
python scripts/run_all.py
```

All random seeds are controlled by `config.yaml`. The synthetic dataset experiments run without any external data download.

---

## Publishing Formats

| Format | Source | Purpose |
|--------|--------|---------|
| Markdown | `article/metrics-that-matter.md` | Canonical source — rendered by the [portfolio site](https://brunoramosmartins.github.io) |
| HTML (landing page) | `docs/index.html` | Repository GitHub Pages — project summary and link to full article |
| LaTeX / PDF | `article/latex/` | Optional academic version, generated via Pandoc from the Markdown source |

---

## Licenses

| Content | License |
|---------|---------|
| Source code (`src/`, `scripts/`, `notebooks/`) | [MIT License](LICENSE) |
| Article text (`article/`, `docs/`) | [CC BY 4.0](LICENSE-TEXT) |

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Foundation & Project Setup | Complete |
| Phase 1 | Theoretical Foundation | Complete |
| Phase 2 | Metric Landscape | Complete |
| Phase 3 | Experiments & Code | Complete |
| Phase 4 | Article Writing | In progress |
| Phase 5 | Review & Polish | Pending |
| Phase 6 | Publishing | Pending |

---

## References

- Fawcett, T. (2006). *An Introduction to ROC Analysis.* Pattern Recognition Letters.
- Davis, J. & Goadrich, M. (2006). *The Relationship Between Precision-Recall and ROC Curves.* ICML.
- Saito, T. & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLOS ONE.
- Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.).
