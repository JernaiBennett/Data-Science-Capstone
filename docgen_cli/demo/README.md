# Demo files for docgen

A small shop library to document, snippets in four other languages, and three bug reports. Nothing here is part of the model's training data.

Run everything at once (uses copies, so the originals stay unchanged):

```powershell
.\demo\run_demo.ps1
```

Or run commands one at a time from the `docgen_cli` folder:

```powershell
docgen doc demo\shop\pricing.py
docgen doc demo\shop                # a folder works too (searched recursively)
docgen doc demo\shop --full          # preview with parameter sections too
docgen snippet demo\snippets\moving_average.js
docgen bug demo\bugs\crash_on_save.txt
```

What to expect from `docgen doc`:

| File | Documented by the tool | Skipped, and why |
|---|---|---|
| `shop/pricing.py` | `apply_discount`, `add_tax`, `split_bill`, `cheapest` | `bulk_price` already has a docstring |
| `shop/cart.py` | `__init__`, `add_item`, `remove_item`, `subtotal`, `total`, `item_count`, `__len__` | `clear` already has a docstring |
| `shop/textutils.py` | `slugify`, `truncate`, `is_palindrome`, `word_frequencies` | `clamp` is a one-line function |
| `shop/tasks.py` | `retry`, the inner `attempt`, `gather_limited`, the inner `run` | none |

The bug reports contain no title line, matching the training data, where the model reads a report body and writes its title. `wrong_total.txt` is deliberately a little confusing, to show how the model handles a noisy report.
