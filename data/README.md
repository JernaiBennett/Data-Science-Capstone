# Data

The datasets are not stored in this repository because of their size (the combined file is about 86 MB and the training split about 69 MB).

The notebooks build them and save them to Google Drive under `MyDrive/capstone/data/`:

| File | Created by | Contents |
|---|---|---|
| `combined_dataset_v1.parquet` | `01_build_dataset.ipynb` | Combined dataset before the leak fix (CodeSearchNet and GitBugs) |
| `dataset_filter_log_v1.json` | `01_build_dataset.ipynb` | Filtering settings used for v1 |
| `splits_v1/` | `01_build_dataset.ipynb` | Train, validation, test splits with leaked Python inputs (do not use) |
| `combined_dataset_v2.parquet` | `02_fix_python_docstring_leak.ipynb` | 199,878 records after docstring stripping |
| `splits_v2/` | `02_fix_python_docstring_leak.ipynb` | 159,902 train / 19,988 validation / 19,988 test (used for all reported results) |

Columns: `input_text`, `target_text`, `source`, `language`, `task_type`.

Sources: CodeSearchNet (Python, Java, JavaScript, Go, PHP, Ruby) and GitBugs. Follow each dataset's own license and citation requirements.
