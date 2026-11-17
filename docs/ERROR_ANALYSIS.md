# Error analysis

Companion to the statistical comparison in the README. It describes what the three models produce on the fixed v2 test set (19,988 examples) and where they score low. All numbers come from the files in `results/final_comparison/` (`output_shape.csv`, `rouge_by_input_length.csv`, `rouge_non_english_references.csv`, `examples_bug_reports.csv`), produced by `scripts/final_evaluation.py`. Shape flags are regular-expression checks on the raw model output, not a judgment of correctness.

## 1. What the outputs look like

Share of outputs by shape, per model (percent of test examples):

| Group | Model | Tags or sections present | Tag-only (no summary sentence) | Median words (reference) |
|---|---|---|---|---|
| All | Full fine-tune | 67.3 | 9.7 | 11 (16) |
| All | LoRA | 71.5 | 10.3 | 11 (16) |
| All | CodeT5-small | 72.4 | 11.9 | 10 (16) |
| Python | Full fine-tune | 68.3 | 0.1 | 19 (19) |
| Python | LoRA | 78.5 | 0.0 | 18 (19) |
| Python | CodeT5-small | 81.8 | 0.1 | 20 (19) |
| PHP | Full fine-tune | 96.2 | 36.9 | 8 (15) |
| PHP | LoRA | 99.3 | 38.9 | 8 (15) |
| PHP | CodeT5-small | 99.6 | 45.5 | 8 (15) |
| GitBugs | Full fine-tune | 0.1 | 0.1 | 7 (9) |

"Tags or sections" means the output contains `@param`, `:param`, `Args:`, `Returns:`, or a NumPy-style header such as `Parameters` followed by a dashed line. No model ever produced an empty output.

Findings:

- **No empty outputs and almost no degenerate repeats.** Repeated sections are rare in every model on the test set (0.00% full fine-tune, 0.01% LoRA, 0.21% CodeT5-small). Repeated lines are more common in CodeT5-small (1.16% overall, 3.17% in Python) than in the full fine-tune (0.29% overall, 0.18% in Python).
- **PHP is the weakest language for usable summaries.** Between 37% and 46% of PHP outputs contain only tags, with no sentence describing the function, and the median PHP output is 8 words against 15 for the reference. Models learned this form from PHP references, so a tag-only output can still earn overlap credit while telling a reader little. This matches the PHP snippet in the CLI demo, which returned only `@param` and `@return` lines.
- **The full fine-tune writes fewer tag and section blocks than the other two** in Python (68% vs 79% and 82%), Java (91% vs 94% and 95%), and JavaScript (89% vs 94% and 94%). I did not test whether this explains its higher scores, and fewer sections is not better by itself.
- **Go is the clean case.** No Go output contains tags, and Go outputs are mostly one line, as the references are (a `// Name does ...` comment).
- **Bug titles are short.** The median bug title is 7 words (full fine-tune) against 9 for the reference, and almost all are a single line.
- **Test-set repeats do not match the demo.** The CLI demo produced a duplicated `Returns` section from the full fine-tune, which the test set almost never does. The demo functions are short toy functions that differ from the test inputs. I have not established why, which is why the CLI keeps its clean-up step.

## 2. Input length

ROUGE-L by quartile of input length (characters):

| Model | Shortest 25% | 25-50% | 50-75% | Longest 25% |
|---|---|---|---|---|
| Full fine-tune | 45.17 | 40.60 | 38.52 | 35.25 |
| LoRA | 42.75 | 38.30 | 36.85 | 33.27 |
| CodeT5-small | 42.61 | 37.85 | 35.80 | 32.01 |

All three models lose about 10 ROUGE-L points from the shortest to the longest quarter. The full fine-tune leads LoRA in every quarter by 1.7 to 2.4 points, so its advantage does not come from one input size. Input length is mixed up with language and task (for example Java inputs are longer than Go inputs), so this table does not isolate length as the cause.

## 3. Non-English references

146 test examples (0.7%) have mostly non-ASCII references.

| Model | Mostly non-ASCII reference | ASCII reference |
|---|---|---|
| Full fine-tune | 47.73 | 39.83 |
| LoRA | 41.22 | 37.77 |
| CodeT5-small | 39.21 | 37.05 |

These scores are higher than on English references, which should not be read as better multilingual quality. The default ROUGE tokenizer keeps only ASCII letters and digits, so most non-English text is dropped before scoring and the remaining ASCII tokens (names, identifiers) drive the result. I have not verified this on the 146 examples. The subset is also small. Treat these rows as uninterpretable rather than as a result.

## 4. Bug reports

A random sample of 40 of the 973 test bug reports (seed 42) is in `results/final_comparison/examples_bug_reports.csv`. Counts below describe this sample only and are illustrative, not estimates for the test set.

- Mean ROUGE-L in the sample: full fine-tune 33.1, LoRA 25.9, CodeT5-small 26.7. The full fine-tune scored higher than LoRA on 21 of 40, lower on 8, and tied on 11. The subset-wide comparison with intervals is in `pairwise_differences.csv` (group `gitbugs`).
- Seven of 40 full-fine-tune titles scored zero (nine for CodeT5-small).
- In five of 40 reports all three models produced the same title.

Failure patterns seen in the sample:

1. **Nothing to summarize.** Several reports contain only a screenshot or a few words. For an image-only body the models invent a title ("Markdown editor is broken" for a screenshot of a path-matching bug) or echo the body.
2. **Symptom copied instead of the title's abstraction.** For a report whose body says memory usage climbed to 87 GB, all three produced "Memory usage by Thunderbird climbed to 87 GB"; the reference is "Large memory leakage on Mac Silicon". The first sentence of the body is a common output. At least 7 of 40 full-fine-tune titles appear verbatim in the first 500 characters of the body, and the file stores only that much, so the true count is higher.
3. **Plausible but wrong content.** "Tasks.json file cannot be created" for a report about environment variables not persisting across task chains; "Fix bug 1696895" for a report asking to add missing message-header information.
4. **Copying makes some scores look better than the understanding behind them.** The highest-scoring example (ROUGE-L 98.2 for all three models) is an automated intermittent-failure report whose title is a log line that also appears in the body. A copy of the right line scores almost perfectly.
5. **Small identifier errors.** One LoRA output mis-copied an issue key ("CASSURE_17056" for "CASSANDRA-17056").

## 5. What this means for the CLI

- Keeping only the summary line by default is supported by the data: parameter and returns blocks appear in about two thirds of outputs and are the part most often generic or wrong.
- A PHP-style output with tags only should be reported as having no summary sentence, which the CLI does.
- Bug titles are drafts. They are strongest when the report body already states the problem in one sentence and weakest when the body is a screenshot, a log, or a long discussion.

## 6. Limits

- ROUGE-L, BLEU, and METEOR measure overlap with one reference, not correctness or usefulness. No human evaluation was done.
- Shape flags are regular-expression checks and can miss unusual formats.
- The bug-report patterns come from 40 examples read by one person.
- Each model was trained once, so differences between models carry training-run variation that the confidence intervals do not include.
