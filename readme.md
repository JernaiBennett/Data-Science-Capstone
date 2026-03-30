# Automated Code Documentation & Bug Report Analysis Using Fine-Tuned CodeT5

**MS in Data Science & AI Capstone Project**  
Florida International University — Spring/Fall 2026  
Mentor: Dr. Olaoluwa Adigun

---

## Overview

This project develops an AI-powered pipeline that automatically generates code documentation (docstrings, inline comments) and produces structured summaries of bug reports using a fine-tuned **CodeT5+ (220M parameters)** model. The system targets real-world developer productivity by reducing the manual overhead of documentation and standardizing bug triage at scale.

The project is divided into two phases:
- **Spring 2026** — Data collection, preprocessing, and model selection
- **Fall 2026** — Model fine-tuning, evaluation, and final presentation

---

## Project Structure

```
data-science-capstone/
│
├── data/
│   ├── raw/                    # Original downloaded datasets (not committed — see below)
│   ├── processed/              # Cleaned, filtered, and split datasets
│   └── samples/                # Small sample files for quick testing
│
├── notebooks/
│   ├── 01_eda_exploration.ipynb             # Exploratory data analysis
│   ├── 02_data_cleaning.ipynb               # Preprocessing pipeline
│   ├── 03_model_comparison.ipynb            # Baseline model comparisons
│   └── codet5_eda_model_comparison_FIXED_v2.ipynb  # Latest fixed notebook
│
├── src/
│   ├── data/
│   │   ├── download.py         # Dataset download scripts
│   │   ├── preprocess.py       # Cleaning and filtering logic
│   │   └── split.py            # Train/val/test splitting
│   ├── models/
│   │   ├── tokenizer_utils.py  # Defensive tokenizer loading (load_tokenizer_safe)
│   │   └── train.py            # Fine-tuning scripts (Fall 2026)
│   └── evaluation/
│       └── metrics.py          # BLEU, ROUGE-L, METEOR scoring
│
├── reports/
│   ├── spring2026_intro_slides.pptx
│   └── figures/                # Charts, EDA plots
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Datasets

| Dataset | Source | Size | Purpose |
|---|---|---|---|
| [CodeSearchNet (Python)](https://huggingface.co/datasets/code_search_net) | Hugging Face | ~455K function-doc pairs | Code documentation generation |
| [GitBugs](https://huggingface.co/datasets/gitbugs) | Hugging Face | 150K+ bug reports | Bug report summarization |

**Combined training corpus:** ~200K samples after filtering and quality control.

> ⚠️ **Raw data is not committed to this repository** due to file size. See [Data Setup](#data-setup) below to reproduce the dataset locally.

---

## Models Compared

| Model | Parameters | Notes |
|---|---|---|
| **CodeT5+ 220M** *(primary)* | 220M | `Salesforce/codet5p-220m` |
| CodeT5-small | 60M | Baseline comparison |
| T5-small | 60M | General-purpose baseline |
| CodeT5+ 220M + LoRA | ~220M | Parameter-efficient fine-tuning variant |

---

## Evaluation Metrics

- **BLEU** — n-gram precision for generated documentation
- **ROUGE-L** — Longest common subsequence recall
- **METEOR** — Alignment-aware generation quality
- Developer satisfaction targets (qualitative, Fall 2026)

---

## Data Setup

To reproduce the dataset locally, run the following in a Colab or local environment:

```python
# CodeSearchNet (Python)
from datasets import load_dataset
ds = load_dataset("code_search_net", "python")

# GitBugs
from datasets import load_dataset
bugs = load_dataset("gitbugs")
```

Or use the provided notebook: `notebooks/02_data_cleaning.ipynb`

---

## Environment

This project was developed and tested in **Google Colab** using the following key libraries:

```
transformers>=4.38.0
datasets
torch
evaluate
rouge-score
nltk
sacrebleu
```

Install dependencies:
```bash
pip install -r requirements.txt
```

> **Note:** This project uses a `load_tokenizer_safe()` utility function to handle tokenizer version mismatches between different `transformers` releases. See `src/models/tokenizer_utils.py`.

---

## Known Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `extra_special_tokens` type error | CodeT5 tokenizer version mismatch | `load_tokenizer_safe()` in `tokenizer_utils.py` |
| `evaluation_strategy` keyword error | Newer `transformers` API renamed arg | Use `eval_strategy` in `TrainingArguments` |

---

## Roadmap

- [x] Project scoping and mentor alignment (Dr. Adigun)
- [x] Dataset identification and access
- [x] Exploratory data analysis
- [x] Model comparison notebook (CodeT5+, CodeT5-small, T5-small, LoRA)
- [ ] Finalize GitBugs dataset (confirm split/version)
- [ ] Complete preprocessing pipeline
- [ ] Submit Canvas project description
- [ ] **Fall 2026:** Fine-tune CodeT5+ on combined corpus
- [ ] **Fall 2026:** Evaluate with BLEU / ROUGE-L / METEOR
- [ ] **Fall 2026:** Final committee presentation

---

## Citation

```bibtex
@inproceedings{husain2019codesearchnet,
  title={CodeSearchNet Challenge: Evaluating the State of Semantic Code Search},
  author={Husain, Hamel and Wu, Ho-Howard and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  year={2019}
}

@article{wang2021codet5,
  title={CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation},
  author={Wang, Yue and Wang, Weishi and Joty, Shafiq and Hoi, Steven C.H.},
  year={2021}
}
```

---

## Author

**Jernai**  
MS in Data Science & AI — Florida International University  
Capstone Mentor: Dr. Olaoluwa Adigun
