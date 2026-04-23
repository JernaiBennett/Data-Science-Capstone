# Automated Code Documentation & Bug Report Analysis Using Fine-Tuned CodeT5

**MS in Data Science & AI Capstone Project**  
Florida International University — Spring/Fall 2026  
Mentor: Dr. Olaoluwa Adigun

---

## Overview

This project investigates fine-tuned transformer-based models for two software engineering automation tasks:

1. **Code documentation generation** — producing Python docstrings from function source code
2. **Bug report analysis** — summarizing GitHub issues into structured natural-language descriptions

Three architectures are benchmarked across Spring and Fall 2026:

| Model | Params | Trainable | Tokenizer | Semester |
|---|---|---|---|---|
| CodeT5-small | 60M | 60M (100%) | RobertaTokenizer | Spring 2026 |
| CodeT5+ 220M | 220M | 220M (100%) | AutoTokenizer | Fall 2026 |
| CodeT5+ 220M + LoRA | 220M | ~1.1M (0.5%) | AutoTokenizer | Fall 2026 |

---

## Results (Spring 2026 — CodeT5-small)

Evaluated on the ConcodePlus enriched-target task (CodeSearchNet Python, 15K comparison subset, 3 epochs):

| Metric | Score |
|---|---|
| BLEU-4 | 98.57 |
| ROUGE-L | 99.09 |
| METEOR | 99.14 |
| CodeBLEU | N/A* |
| Eval loss | 0.041 |
| Train time | 53.1 min (A100) |

> **Note on scores:** Near-perfect values reflect the ConcodePlus task framing — targets include a deterministically extracted `Logic:` suffix from `extract_logic_summary()`, which inflates n-gram metrics. Fall 2026 evaluation will run against plain docstring references for externally comparable benchmarks.

---

## Repository Structure

```
.
├── semester1_codet5.ipynb     # Full pipeline notebook (Spring 2026)
├── requirements.txt           # Python dependencies
├── .gitignore
├── README.md
└── outputs/
    ├── eda_csn.png            # CodeSearchNet EDA (6-panel)
    ├── eda_github.png         # GitHub Issues EDA (3-panel)
    ├── codet5_small_results.png   # Comparison run metrics
    ├── before_after_finetune.png  # Before vs after full fine-tuning
    └── training_results.json  # Saved metrics
```

---

## Pipeline

```
CodeSearchNet Python (412K)          GitHub Issues (GitBugs)
         |                                     |
  Quality filtering                    Quality filtering
  (~200K retained)                     (30K retained)
         |                                     |
  ConcodePlus enrichment               Tokenization
  (Logic suffix appended)              (256 in / 128 out)
         |                                     |
         +------------- Merge -----------------+
                             |
                    Combined training set
                             |
               Fine-tuning — CodeT5-small
               (A100 · 50K samples · 2 epochs)
                             |
                        Evaluation
              BLEU-4 · ROUGE-L · METEOR · CodeBLEU
```

---

## Setup

### Requirements

- Python 3.10+
- Google Colab Pro (A100 recommended) or local GPU with 16GB+ VRAM
- Google Drive (for checkpoint persistence)

### Install

```bash
pip install -r requirements.txt
```

Key dependencies:

```
transformers>=4.40
peft>=0.10
torch>=2.1
datasets>=2.18
evaluate>=0.4
codebleu
rouge_score
sacrebleu
sentencepiece
```

### Run

Open `semester1_codet5.ipynb` in Google Colab and run sections in order:

| Section | Description |
|---|---|
| 0 | Environment setup and GPU verification |
| 1 | Data loading, EDA, quality filtering, PyPlus enrichment |
| 2 | Metric suite (BLEU-4, ROUGE-L, METEOR, CodeBLEU) |
| 3 | Model definition and tokenizer loading |
| 4 | Training (CodeT5-small comparison run) |
| 5 | Results and analysis |
| 6 | Full fine-tuning on combined dataset |
| 7 | Final evaluation and inference demo |

> **Already trained?** Run the "Load from Drive" cell in Section 6 to skip retraining and jump straight to evaluation.

---

## Model Details

### CodeT5-small (baseline)

- **HF ID:** `Salesforce/codet5-small`
- **Architecture:** Encoder-decoder T5
- **Pre-training:** CodeSearchNet (6 languages, 8.35M functions)
- **Tokenizer note:** Uses `RobertaTokenizer` directly — `AutoTokenizer` triggers an `extra_special_tokens` TypeError with newer versions of `transformers`. A monkey-patch is included in Section 3 of the notebook.

### CodeT5+ 220M (Fall 2026)

- **HF ID:** `Salesforce/codet5p-220m`
- **Architecture:** Improved encoder-decoder with better pre-training objectives
- **Tokenizer:** `AutoTokenizer` (no compatibility issues)

### CodeT5+ 220M + LoRA (Fall 2026)

- **Base model:** `Salesforce/codet5p-220m`
- **LoRA rank:** r=16, alpha=32, dropout=0.10
- **Target modules:** `q`, `v` (query and value projections in all attention heads)
- **Trainable params:** ~1.1M of 220M (~0.5%)

---

## Training Configuration

| Hyperparameter | Comparison run | Full fine-tuning |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Learning rate | 5e-5 | 3e-5 |
| LR schedule | Linear warmup | Cosine |
| Warmup ratio | 0.10 | 0.10 |
| Batch size | 16 | 32 |
| Epochs | 3 | 2 |
| Training samples | 15,000 | 50,000 |
| Max input length | 512 tokens | 512 tokens |
| Max target length | 128 tokens | 128 tokens |
| Primary metric | ROUGE-L | ROUGE-L |
| Hardware | A100 40GB | A100 40GB |

---

## Dataset Details

### CodeSearchNet Python

- **Source:** `code_search_net` on Hugging Face Hub
- **Raw splits:** 412K train / 23K val / 22K test
- **After filtering:** ~200K train / 15K val / 15K test
- **Quality filters:** docstring 10-200 words, code 50-8,000 chars, ≥3 lines, ≥3 identifiers, no boilerplate stubs

### GitHub Issues (GitBugs)

- **Source:** `sharjeelyunus/github-issues-dataset` on Hugging Face Hub
- **After filtering:** 30K samples
- **Quality filters:** title ≥3 words, body 20-600 words, no generic single-word titles

### ConcodePlus Enrichment

Training targets are enriched with a `Logic:` suffix extracted deterministically from the function AST by `extract_logic_summary()`. This produces richer supervision signals aligned with the ConcodePlus task framing (Shaalan & Zakaria 2023). **Note:** this suffix inflates standard n-gram metrics — see the Results note above.

---

## Evaluation Metrics

| Metric | Paper | What it measures |
|---|---|---|
| BLEU-4 | Papineni et al. 2002 | 4-gram precision vs. reference |
| ROUGE-L | Lin 2004 | Longest common subsequence recall (primary checkpoint metric) |
| METEOR | Banerjee & Lavie 2005 | Synonym-aware semantic matching |
| CodeBLEU | Ren et al. 2020 | Code-aware: n-gram + AST + dataflow |

ROUGE-L is the primary checkpoint selection metric, consistent with the CodeT5+ paper (Wang et al. 2023).

---

## Known Issues / Limitations

- **Runtime constraint:** Full fine-tuning was limited to 50K samples over 2 epochs due to Colab session time limits (~3 hours per run). Full 200K training is planned for Fall 2026.
- **Single model this semester:** Only CodeT5-small was evaluated in Spring 2026. Three-model comparison is Fall 2026.
- **Score inflation:** ConcodePlus enrichment inflates BLEU/ROUGE/METEOR — not directly comparable to published CodeSearchNet benchmarks.
- **CodeBLEU unavailable** in the current Colab environment.

---

## References

- Wang, Y. et al. (2021). CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models. EMNLP 2021.
- Wang, Y. et al. (2023). CodeT5+: Open Code Large Language Models. EMNLP 2023.
- Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Shaalan, K. & Zakaria, L. (2023). Automated code documentation. Journal of Systems and Software.
- Husain, H. et al. (2019). CodeSearchNet Challenge. arXiv:1909.09436.
- Ren, S. et al. (2020). CodeBLEU. arXiv:2009.10297.
