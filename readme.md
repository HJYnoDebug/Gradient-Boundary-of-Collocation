# The Gradient Boundary of Collocational Knowledge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LLM: Llama-3 / GPT-4o](https://img.shields.io/badge/LLM-Llama--3%20%7C%20GPT--4o-green.svg)](https://meta.ai)

This repository contains the dataset, evaluation framework, and statistical analysis for our study on the **Scaling Laws** and **Probabilistic Representations** of academic collocations in Large Language Models (LLMs).

## 📖 Project Overview

How do LLMs internalize the "middle ground" of language—collocations? This project explores whether model scale (from 3B to 70B+ parameters) predictably improves collocational sensitivity and investigates if this knowledge is represented as discrete symbolic rules or a gradient statistical continuum.

### Key Insights
1. **Scaling Laws**: We identify a robust logarithmic correlation between parameter count and accuracy. In the Llama-3 series, scaling from 3B to 70B reduces the relative error rate by over **80%**.
2. **Gradient Boundaries**: Using GPT-4o-mini as a probe, we found that even with a perfect AUC (1.00), the internal probability distribution is continuous, suggesting knowledge is **gradient** rather than categorical.
3. **Interference Zones**: Models remain susceptible to "probabilistic pull" from high-frequency colloquialisms (e.g., *"in the beginning"*) even when they conflict with formal academic norms (*"at the beginning"*).

---

## 🛠 Methodology

### 1. Dataset: ColloCaid-2AFC
We transformed the *ColloCaid* academic error corpus into a **Two-Alternative Forced Choice (2AFC)** task.
- **Total Samples**: 370 distinct academic collocation pairs.
- **Error Categories**: Prepositions, Formality, Articles, Lexical choice, Infinitives, etc.

### 2. Evaluation Metric: Confidence Margin ($\Delta P$)
We extract token-level log-probabilities to calculate the relative preference:
$$\Delta P = \log P(\text{canonical solution}) - \log P(\text{erroneous problem})$$
- $\Delta P > 0$: Indicates a preference for the academic norm.
- **Threshold**: We utilized Youden’s J statistic to identify the optimal discriminative boundary.

---

## 📊 Experimental Results

### Overall Performance Summary
| Model Family | Model ID | Parameters | Accuracy | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Llama 3** | Llama-3.2-3B | 3B | 81.62% | 0.824 |
| **Llama 3** | Llama-3.1-8B | 8B | 89.46% | 0.898 |
| **Llama 3** | Llama-3.3-70B | 70B | 96.49% | 0.966 |
| **Proprietary** | Mistral-Medium | Unknown | 97.57% | 0.976 |
| **Proprietary** | DeepSeek-V3 | 671B (MoE) | 98.65% | 0.986 |
| **Proprietary** | GPT-4o-Mini | Unknown | 98.38% | 0.983 |
| **Proprietary** | GPT-4o | Unknown | **99.19%** | **0.992** |

### Probabilistic Discrimination (Probe: GPT-4o-mini)
- **AUC**: 1.0000 (Perfect linear separability at scale).
- **Optimal Threshold**: 0.6225 (Indicates a required margin for confident classification).

---

## 📂 Repository Structure & Usage

### Structure
```text
├── data/
│   └── analysis_summary.csv   # Processed results from all 7 models
├── src/
│   ├── probe.py               # Logic for extracting log-probs via API
│   └── scaling_plot.py        # Script to generate Scaling Law visualizations
├── results/
│   ├── margin_density.png     # Distribution of Delta P
│   └── roc_curve.png          # ROC analysis results
└── paper/
    └── manuscript.tex         # ACL format LaTeX source

