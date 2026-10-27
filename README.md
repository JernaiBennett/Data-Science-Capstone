# Automated Code Documentation and Bug Report Analysis Using Fine-Tuned CodeT5+ Models

**Author:** Jernai Bennett, MS in Data Science and AI, Florida International University
**Advisor:** Dr. Olaoluwa Adigun

This project fine-tunes **CodeT5+ 220M**, a code-specific language model, to draft docstrings for source code and summaries for bug reports. The long-term goal is a command-line tool that automates both tasks for software engineers. The repository compares **full fine-tuning** against **LoRA** (low-rank adaptation) on the same data, metrics, and GPU.

## Results

Test set: 19,988 examples (six programming languages plus GitBugs). Higher is better. Hardware: NVIDIA A100 (40 GB) on Google Colab.

| Model | BLEU-4 | ROUGE-L | METEOR | Trainable parameters | Peak GPU memory | Saved model size |
|---|---|---|---|---|---|---|
| CodeT5+ 220M, full fine-tune | 16.40 | 39.89 | 37.54 | 222.9M (100%) | 12.7 GB | about 890 MB |
| CodeT5+ 220M, LoRA | 13.32 | 37.79 | 35.29 | 0.88M (0.4%) | 7.9 GB | 3.6 MB adapter |
| CodeT5-small (baseline) | pending | pending | pending | n/a | n/a | n/a |

- Full fine-tuning scores highest on every metric and in every language group; LoRA is within about 2 ROUGE-L points while using 37.5% less peak GPU memory.
- Training time per epoch is nearly the same (27.7 vs 29.0 minutes). Once merged into the base model, LoRA inference speed matches the full fine-tune.
- Per-language scores, first-sentence scores, and the efficiency table are in `results/comparison/`.

## Key finding: a data leak, found and fixed

In CodeSearchNet's Python subset the function code includes its own docstring, which is also the target. The first full fine-tune therefore learned to copy the answer from the input (98.6% of Python test inputs contained the target; Python reached 72.6 BLEU-4 and 96.6 ROUGE-L). Docstrings were stripped from Python inputs with the `ast` module, 122 rows that still leaked were dropped, and everything was retrained on the fixed splits (Python fell to a realistic 9.6 BLEU-4 and 28.1 ROUGE-L). The leaked run is kept in `results/full_finetune_v1_leaked/` as documentation. All reported results use the fixed data.

## Repository structure

```
notebooks/
  01_build_dataset.ipynb             Build and filter the combined dataset, create v1 splits
  02_fix_python_docstring_leak.ipynb Strip Python docstrings from inputs, create v2 splits
  03_full_finetune.ipynb             CodeT5+ 220M full fine-tune, evaluation, results export
  04_lora_finetune.ipynb             CodeT5+ 220M LoRA, evaluation, comparison with full fine-tune
results/
  full_finetune_v1_leaked/           First run (leaked data), kept for the record
  full_finetune_v2/                  Full fine-tune on the fixed data
  lora_v2/                           LoRA on the fixed data
  comparison/                        Side-by-side metrics and efficiency tables
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

- One training run per model (a single seed); LoRA settings are not tuned yet.
- Reference docstrings include parameter and returns sections, which lowers n-gram scores for every model.
- The Spring 2026 CodeT5-small baseline was trained on Python only and likely on leaked data; it will be retrained on the fixed splits.
- Next: retrain CodeT5-small, error analysis across all three models, an optional LoRA sweep, and the command-line demo.

## References

- Wang et al. (2021). CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation.
- Wang et al. (2023). CodeT5+: Open Code Large Language Models for Code Understanding and Generation.
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- Husain et al. (2019). CodeSearchNet Challenge: Evaluating the State of Semantic Code Search.
- Patil et al. (2025). GitBugs: Bug Reports for Duplicate Detection, Retrieval Augmented Generation, Triage, and More.
- Papineni et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation.
- Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
- Banerjee and Lavie (2005). METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments.
