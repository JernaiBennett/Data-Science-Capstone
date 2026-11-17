# Automated Code Documentation and Bug Report Analysis Using Fine-Tuned CodeT5+ Models

**Author:** Jernai Bennett, MS in Data Science and AI, Florida International University
**Advisor:** Dr. Olaoluwa Adigun

This project fine-tunes **CodeT5+ 220M**, a code-specific language model, to draft docstrings for source code and summaries for bug reports. The repository compares **full fine-tuning**, **LoRA** (low-rank adaptation), and a **CodeT5-small** baseline on the same data, metrics, and GPU, and includes `docgen`, a command-line tool built on the best model (see [docgen_cli/](docgen_cli/)).

## Command-line tool

`docgen` proposes docstrings for undocumented Python functions (as a diff you review), summarizes snippets in six languages, and writes titles for bug reports. Install and usage are in [docgen_cli/README.md](docgen_cli/README.md). Model weights are not stored in the repository.

## Results

Test set: 19,988 examples (six programming languages plus GitBugs). Higher is better. Hardware: NVIDIA A100 (40 GB) on Google Colab.

| Model | BLEU-4 | ROUGE-L | METEOR | Trainable parameters | Peak GPU memory | Saved model size |
|---|---|---|---|---|---|---|
| CodeT5+ 220M, full fine-tune | 16.40 | 39.89 | 37.54 | 222.9M (100%) | 12.7 GB | about 890 MB |
| CodeT5+ 220M, LoRA | 13.32 | 37.79 | 35.29 | 0.88M (0.4%) | 7.9 GB | 3.6 MB adapter |
| CodeT5-small (baseline) | 13.44 | 37.07 | 34.71 | 60.5M (100%) | 4.5 GB | 242 MB |

- Full fine-tuning scores highest on every metric and in every language group; LoRA is within about 2 ROUGE-L points while using 37.5% less peak GPU memory.
- Training time per epoch is nearly the same (27.7 vs 29.0 minutes). Once merged into the base model, LoRA inference speed matches the full fine-tune.
- Paired bootstrap intervals (below) show the full fine-tune is ahead of both other models on every metric. LoRA is slightly ahead of CodeT5-small on ROUGE-L and METEOR (about 0.6 to 0.7 points) and level on BLEU-4. The intervals reflect test-set sampling only; each model was trained once, so training-seed variation is not measured.
- Per-language scores, first-sentence scores, and the efficiency table are in `results/comparisons/`.

## Statistical comparison

95% intervals from a paired bootstrap over the 19,988 test examples (2,000 resamples, seed 42; the same resampled examples are used for every model). Produced by `scripts/final_evaluation.py`; tables and examples are in `results/final_comparison/`.

| Comparison | BLEU-4 | ROUGE-L | METEOR |
|---|---|---|---|
| Full fine-tune minus LoRA | +3.09 (2.86 to 3.32) | +2.09 (1.90 to 2.28) | +2.24 (2.05 to 2.44) |
| Full fine-tune minus CodeT5-small | +2.97 (2.76 to 3.18) | +2.82 (2.62 to 3.03) | +2.82 (2.63 to 3.03) |
| LoRA minus CodeT5-small | -0.12 (-0.29 to 0.04) | +0.73 (0.56 to 0.90) | +0.58 (0.41 to 0.76) |

- The full fine-tune beats both models on every metric; no interval comes near zero.
- LoRA beats CodeT5-small on ROUGE-L and METEOR by a small margin; on BLEU-4 the difference is not distinguishable from zero.
- The full fine-tune scores higher than LoRA on 40.6% of examples, ties on 27.7%, and scores lower on the rest.
- These intervals cover which test examples were drawn. They do not cover run-to-run variation from training, because each model was trained once.

## Key finding: a data leak, found and fixed

In CodeSearchNet's Python subset the function code includes its own docstring, which is also the target. The first full fine-tune therefore learned to copy the answer from the input (98.6% of Python test inputs contained the target; Python reached 72.6 BLEU-4 and 96.6 ROUGE-L). Docstrings were stripped from Python inputs with the `ast` module, 122 rows that still leaked were dropped, and everything was retrained on the fixed splits (Python fell to a realistic 9.6 BLEU-4 and 28.1 ROUGE-L). The leaked run is kept in `results/full_finetune_v1_leaked/` as documentation. All reported results use the fixed data.

## Repository structure

```
notebooks/
  01_build_dataset.ipynb             Build and filter the combined dataset, create v1 splits
  02_fix_python_docstring_leak.ipynb Strip Python docstrings from inputs, create v2 splits
  03_full_finetune.ipynb             CodeT5+ 220M full fine-tune, evaluation, results export
  04_lora_finetune.ipynb             CodeT5+ 220M LoRA, evaluation, comparison with full fine-tune
  05_codet5_small_baseline.ipynb     CodeT5-small on the fixed splits, matched evaluation, three-model comparison
results/
  full_finetune_v1_leaked/           First run (leaked data), kept for the record
  full_finetune_v2/                  Full fine-tune on the fixed data
  lora_v2/                           LoRA on the fixed data
  comparisons/                       Side-by-side metrics and efficiency tables
docgen_cli/                          The `docgen` command-line tool, tests, demo project, and model card
docs/                                Presentation and loss-curve figures
data/                                Data description (datasets are not stored in the repository)
```

Prediction files are gzipped (`.csv.gz`); read them with `pandas.read_csv("file.csv.gz")`.

## How to reproduce

1. Open the notebooks in Google Colab and mount Google Drive. All data and checkpoints are saved under `MyDrive/capstone/`.
2. Run `01_build_dataset.ipynb` on a CPU runtime.
3. Run `02_fix_python_docstring_leak.ipynb` to create `splits_v2`.
4. Run `03_full_finetune.ipynb` on an A100 runtime (about 2 hours; checkpoints resume automatically).
5. Run `04_lora_finetune.ipynb` on the same GPU type (about 1.5 hours of training plus generation).
6. Run `05_codet5_small_baseline.ipynb` (about 40 minutes of training plus generation). It also rescores the full fine-tune and LoRA predictions with the same code and prints the three-model tables.

Run each notebook from top to bottom in a single session. See `data/README.md` for the files each step creates.

### Shared settings

3 epochs, effective batch size 32, bf16 mixed precision, seed 42, 512 input and 128 output tokens, beam search with 4 beams. Learning rate 5e-5 for full fine-tuning and 3e-4 for LoRA (rank 8, alpha 32, dropout 0.1, query and value layers). The LoRA learning rate is higher on purpose because only a small set of weights trains.

### Recording exact package versions

Run this in Colab after a successful run and paste the output into `requirements.txt`:

```python
import torch, transformers, datasets, peft, evaluate
print({m.__name__: m.__version__ for m in (torch, transformers, datasets, peft, evaluate)})
```

## Limitations and next steps

- One training run per model (a single seed), so the intervals above do not include training variation; LoRA settings are not tuned yet.
- Reference docstrings include parameter and returns sections, which lowers n-gram scores for every model.
- The Spring 2026 CodeT5-small baseline was trained on Python only and likely on leaked data. It was replaced by a retrain on the fixed splits (`05_codet5_small_baseline.ipynb`).
- The CLI is a prototype. Parameter and returns text in generated docstrings is often generic or wrong, so it keeps only the summary line by default. Bug titles and non-Python snippet summaries are weak. See `docgen_cli/docs/MODEL_CARD.md`.
- Next: written error analysis from `results/final_comparison/` and an optional LoRA sweep.

## References

- Wang et al. (2021). CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation.
- Wang et al. (2023). CodeT5+: Open Code Large Language Models for Code Understanding and Generation.
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- Husain et al. (2019). CodeSearchNet Challenge: Evaluating the State of Semantic Code Search.
- Patil et al. (2025). GitBugs: Bug Reports for Duplicate Detection, Retrieval Augmented Generation, Triage, and More.
- Papineni et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation.
- Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
- Banerjee and Lavie (2005). METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments.
