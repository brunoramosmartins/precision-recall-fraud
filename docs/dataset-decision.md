# Dataset Decision

> **Status:** Draft — finalize at the start of Phase 3 before writing `src/data.py`.

---

## Candidate Datasets

### Option A — Kaggle Credit Card Fraud Detection


| Attribute       | Value                                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------- |
| Source          | [Kaggle / ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Samples         | 284,807 transactions                                                                           |
| Fraud rate      | 492 frauds / 284,807 total ≈ **0.172%**                                                        |
| Features        | 30 (V1–V28: anonymized PCA components; Time; Amount)                                           |
| License         | DbCL v1.0 — open for research and non-commercial use                                           |
| Citability      | Most cited fraud detection dataset in ML literature                                            |
| Reproducibility | Requires manual download from Kaggle                                                           |


**Pros:** Real transaction data, established benchmark, highly cited, realistic imbalance ratio.
**Cons:** Features are anonymized (PCA-transformed) — no domain interpretability. Requires Kaggle account for download.

---

### Option B — `sklearn.datasets.make_classification`


| Attribute       | Value                                                   |
| --------------- | ------------------------------------------------------- |
| Source          | scikit-learn built-in                                   |
| Samples         | Configurable (default: 100,000)                         |
| Fraud rate      | Configurable — set to 0.1% to mirror Kaggle dataset     |
| Features        | Configurable — interpretable (no anonymization)         |
| License         | BSD 3-Clause (scikit-learn)                             |
| Citability      | Not a citable dataset — only for controlled experiments |
| Reproducibility | Perfect — fully seeded, no download required            |


**Pros:** Full reproducibility without external downloads, controlled imbalance ratio, no license concerns.
**Cons:** Not a real dataset — results are not externally validated, and cannot be cited as fraud detection evidence.

---

### Option C — Both

Use both datasets for complementary purposes:

- **Synthetic** for controlled experiments where the ground truth of the generative process is known.
- **Kaggle** for practical demonstration with real-world data.

---

## Selection Criteria


| Criterion                     | Weight | Kaggle                   | Synthetic             |
| ----------------------------- | ------ | ------------------------ | --------------------- |
| Imbalance ratio realism       | High   | Yes — 0.17%              | Yes — configurable    |
| Dataset size                  | Medium | Yes — 284k               | Yes — configurable    |
| License for public article    | High   | Yes — DbCL v1.0          | Yes — BSD             |
| Citability                    | Medium | Yes — standard benchmark | No — not citable      |
| Reproducibility (no download) | High   | No — requires download   | Yes — zero-dependency |
| Feature interpretability      | Low    | No — anonymized PCA      | Yes — interpretable   |


---

## Decision

**Selected: Option C — Both.**

**Reasoning:**

The article has two audiences and two purposes: (1) theoretical rigor and controlled experiments to demonstrate the mathematical claims, and (2) practical demonstration that results hold on real fraud data.

The **synthetic dataset** will be used for experiments where reproducibility is non-negotiable — specifically Experiments A (accuracy trap) and E (model ranking reversal), where exact control over the generative process is needed to construct the cleanest possible illustration of the theoretical claims. Any reader can reproduce these results with `pip install scikit-learn`.

The **Kaggle dataset** will be used for the primary PR curve and Precision@recall experiments (B, C, D), where the benchmark status of the dataset adds credibility to the conclusions. A clear download instruction in `data/README.md` ensures reproducibility. The anonymized features are not a concern because feature interpretability is out of scope for this article.

This dual-dataset approach will be disclosed explicitly in the article's Experiments section, with a brief justification for each choice.

---

## Dataset Statistics (to be filled in Phase 3)

### Kaggle Credit Card Fraud


| Statistic        | Value |
| ---------------- | ----- |
| n_samples        | —     |
| n_fraud          | —     |
| n_legitimate     | —     |
| fraud_rate       | —     |
| train_size (80%) | —     |
| test_size (20%)  | —     |


### Synthetic Fraud (config.yaml defaults)


| Statistic     | Value      |
| ------------- | ---------- |
| n_samples     | 100,000    |
| n_fraud       | 100 (0.1%) |
| n_features    | 20         |
| n_informative | 10         |
| random_seed   | 42         |


---

## References

- Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE Symposium on Computational Intelligence and Data Mining.
- [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

