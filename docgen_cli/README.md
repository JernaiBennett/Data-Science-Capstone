# docgen

A command-line tool that drafts docstrings for Python code and short summaries for code snippets and bug reports. It runs the CodeT5+ 220M model fine-tuned in the capstone project **"Automated Code Documentation and Bug Report Analysis Using Fine-Tuned CodeT5+ Models"** (MS Data Science and AI, Florida International University).

> **Status: research prototype.** The summary line of a generated docstring is usually plausible. Parameter and returns text is often generic or wrong, so `docgen` keeps only the summary line by default and shows a diff before changing any file. Read [docs/MODEL_CARD.md](docs/MODEL_CARD.md) before relying on the output.

## What it does

| Command | Input | Output |
|---|---|---|
| `docgen doc` | Python files, folders, or wildcards | A diff of proposed docstrings for undocumented functions; `--write` applies them |
| `docgen snippet` | Code in Python, Java, JavaScript, Go, PHP, or Ruby | A one-sentence summary |
| `docgen bug` | A bug report body (steps, actual, expected) | A short issue title |

## Install (Windows PowerShell)

```powershell
cd docgen_cli
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

macOS and Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -e .`

Optional, only for the LoRA adapter: `pip install -e ".[lora]"`.

## Get the model

The weights (about 890 MB) are not in this repository. Place the fine-tuned model folder, which holds `model.safetensors`, `config.json`, and the tokenizer files, at `models/codet5p_full_ft/`. To use another location, pass `--model PATH` or set the `DOCGEN_MODEL` environment variable.

## Use

Preview docstrings. This prints a diff and changes nothing:

```powershell
docgen doc path\to\module.py
```

Apply them after reviewing the diff:

```powershell
docgen doc path\to\module.py --write
docgen doc src\ --write                 # folders are searched recursively
docgen doc module.py --only my_function # one function
docgen doc module.py --full             # also keep Parameters and Returns sections (review carefully)
```

Summarize code in any training language. Several files load the model once:

```powershell
docgen snippet Example.go Example.rb
Get-Content snippet.txt | docgen snippet
docgen snippet Example.php --full       # keep @param / @return tags
```

Write a title for a bug report. Pass the body only, not the title, because the training data has no title in the input:

```powershell
docgen bug report.txt other_report.txt
docgen bug --text "Steps to reproduce: save a file whose name has an emoji. Actual: the app closes. Expected: the file is saved."
```

Use the LoRA adapter instead of the full fine-tune (needs the `lora` extra; the base model downloads from Hugging Face on first use):

```powershell
docgen doc module.py --adapter models\codet5p_lora_adapter
```

Try everything on the bundled demo project: `.\demo\run_demo.ps1` (see [demo/README.md](demo/README.md)). If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

## Behavior

- Runs on a GPU if one is available, otherwise on the CPU (a few seconds per function on CPU).
- Generation settings match the evaluation: 4 beams, up to 128 new tokens, early stopping, inputs truncated to 512 tokens.
- `doc` edits Python files only and adds docstrings only to functions and methods that have none. It skips one-line functions whose body shares the `def` line, and files with syntax errors. Insertion uses the `ast` module, so edited files stay valid Python.
- Generated text is cleaned before use: repeated lines and repeated sections (for example a second `Returns`) are removed.
- A snippet whose output is only tags (common for PHP) prints `(no summary sentence generated)` instead of an invented sentence.

## Debugging

```powershell
$env:DOCGEN_DEBUG = 1
docgen doc demo\shop\pricing.py
```

Each raw model output is printed to the error stream before cleaning.

## Tests

```powershell
pip install -e ".[test]"
pytest
```

The tests use a fake model, so they run without the weights. They cover function discovery, docstring insertion and escaping, diff and write modes, folder and wildcard handling, output cleaning, and error handling.

## Model results (fixed v2 test set, 19,988 examples)

| Model | BLEU-4 | ROUGE-L | METEOR | Latency (ms/example) |
|---|---|---|---|---|
| CodeT5+ 220M full fine-tune (default) | 16.40 | 39.89 | 37.54 | 97.4 |
| CodeT5+ 220M with LoRA (merged) | 13.32 | 37.79 | 35.29 | 90.5 |
| CodeT5-small full fine-tune | 13.44 | 37.07 | 34.71 | 47.6 |

Single training run per model. Details, data, and limitations are in [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## License

MIT, see [LICENSE](LICENSE). The base model and datasets have their own licenses; see the model card.
