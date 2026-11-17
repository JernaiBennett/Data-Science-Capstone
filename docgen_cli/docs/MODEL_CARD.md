# Model card: docgen fine-tuned CodeT5+ 220M

## Summary

CodeT5+ 220M (`Salesforce/codet5p-220m`) fine-tuned for two summarization tasks with one model and no task prefix:

- **Code summarization:** a function in Python, Java, JavaScript, Go, PHP, or Ruby to its documentation comment.
- **Bug report summarization:** a bug report body to its issue title.

Author: Jernai Bennett, MS Data Science and AI, Florida International University. Mentor: Dr. Olaoluwa Adigun.

## Training data

- **Code:** CodeSearchNet (function code paired with its docstring or comment), six languages.
- **Bug reports:** GitBugs. The input is the report body (steps to reproduce, actual and expected behavior) and the target is the issue title, about 9 words at the median.
- **Splits:** a corrected version (`splits_v2`) in which an earlier train/test leak was fixed. All reported numbers use it. Test set: 19,988 examples (19,015 code, 973 bug reports).
- Targets keep the language's own comment style (Go `// Name ...`, PHP and Ruby tags, JSDoc in JavaScript). Some targets are not in English.

## Models compared

| Model | Trainable parameters | Notes |
|---|---|---|
| CodeT5+ 220M full fine-tune | all | Default in the CLI |
| CodeT5+ 220M LoRA | rank 8, alpha 32, on `q` and `v`; 3.6 MB adapter | Merged before inference |
| CodeT5-small full fine-tune | all (60M) | Baseline |

Evaluation used 4 beams, 128 maximum new tokens, early stopping, and 512 input tokens, with BLEU (max order 4), ROUGE-L, and METEOR from the Hugging Face `evaluate` library.

## Results (v2 test set)

| Model | BLEU-4 | ROUGE-L | METEOR | Latency ms/example | Peak GPU GB | Train min | Size |
|---|---|---|---|---|---|---|---|
| Full fine-tune | 16.40 | 39.89 | 37.54 | 97.4 | 12.68 | 83.0 | ~890 MB |
| LoRA | 13.32 | 37.79 | 35.29 | 90.5 | 7.93 | 86.9 | 3.6 MB adapter |
| CodeT5-small | 13.44 | 37.07 | 34.71 | 47.6 | 4.50 | 41.4 | 242 MB |

Reading the table:

- The full fine-tune scores highest on every metric. That is a clear gap over both other models.
- LoRA and CodeT5-small differ by less than one point on each metric. Each was trained once, so these results do not show that either is better.
- The CodeT5-small first-sentence-only scores (14.50 BLEU-4, 33.92 ROUGE-L, 30.61 METEOR) are a separate check and are not comparable to the rows above.
- Scores are low in absolute terms because references often include parameter and returns sections the model rarely reproduces. Compare models with each other.

## Intended use

Drafting documentation that a person reviews: docstring summary lines, snippet summaries, and bug report titles. It is a research prototype built for a capstone project.

## Limitations

- **Parameter and returns text is unreliable.** The model writes NumPy-style sections even though only a small share of Python training targets used them. The text is often generic (`price : float  The price to add.`) or wrong (two parameters both described as "Number of bills to split"). The CLI therefore keeps the summary line by default.
- **Summaries can miss what the code does.** In the demo, snippet summaries matched the language's comment style but not always the content. A PHP snippet produced tags only.
- **Bug titles are weak.** In the demo, one of three titles was on topic. Others copied a step from the report. Title quality tracks the low ROUGE-L for this task (about 29 on the bug subset).
- **Inputs:** one function at a time, truncated at 512 tokens. Classes and whole files are not summarized.
- **Python only for insertion.** Other languages work through `docgen snippet`.
- **Single run per model, one test set.** There are no confidence intervals or multiple seeds.
- **Not evaluated by humans.** Automatic metrics reward overlap with the reference, not correctness.

## Licenses

The base model, CodeSearchNet, and GitBugs each carry their own licenses and terms. Check them before redistributing weights or data. This repository's code is MIT licensed.
