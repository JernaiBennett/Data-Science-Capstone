"""docgen: draft docstrings and bug report summaries with a fine-tuned CodeT5+ model."""

import argparse
import os
import sys
from typing import List

import glob

from . import __version__
from .extract import document_source, process_generated, unified_diff

_SKIP_DIRS = {"__pycache__", "node_modules", "venv", "site-packages", "build", "dist"}


def _expand_paths(args_files: List[str]) -> List[str]:
    """Expand folders (recursively, Python files only) and wildcards, since PowerShell and cmd do not."""
    out: List[str] = []
    for item in args_files:
        if os.path.isdir(item):
            for root, dirs, files in os.walk(item):
                dirs[:] = sorted(d for d in dirs if not d.startswith((".", "_work")) and d not in _SKIP_DIRS)
                out += [os.path.join(root, f) for f in sorted(files) if f.endswith(".py")]
        elif any(ch in item for ch in "*?["):
            out += sorted(glob.glob(item, recursive=True))
        else:
            out.append(item)
    return out

DEFAULT_MODEL = os.environ.get("DOCGEN_MODEL", os.path.join("models", "codet5p_full_ft"))


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Folder with the fine-tuned model (env DOCGEN_MODEL). Default: %(default)s")
    p.add_argument("--adapter", default=None,
                   help="LoRA adapter folder. Uses --base-model plus this adapter instead of --model.")
    p.add_argument("--base-model", default="Salesforce/codet5p-220m",
                   help="Base model for --adapter (local folder or Hub id).")
    p.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Default: cuda if available, else cpu.")
    p.add_argument("--beams", type=int, default=4, help="Beam search width (default 4, as in evaluation).")
    p.add_argument("--batch-size", type=int, default=8)


def _summarizer(args):
    from .model import Summarizer  # imported lazily so --help is fast

    print("Loading model...", file=sys.stderr, flush=True)
    return Summarizer(args.model, adapter=args.adapter, base_model=args.base_model,
                      device=args.device, num_beams=args.beams)


def _read_text(path: str) -> str:
    if path and path != "-":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if sys.stdin.isatty():
        print("Reading from stdin. Paste text, then press Ctrl+Z and Enter (Windows) or Ctrl+D (Mac/Linux).",
              file=sys.stderr)
    return sys.stdin.read()


def cmd_doc(args) -> int:
    summ = None
    changed = 0
    files = _expand_paths(args.files)
    if not files:
        print("No files matched.", file=sys.stderr)
        return 1
    for path in files:
        if not path.endswith(".py"):
            print(f"skip {path}: docstring insertion supports Python files only "
                  f"(use `docgen snippet` for other languages)", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        try:
            from .extract import find_undocumented

            if not find_undocumented(source):
                print(f"{path}: no undocumented functions", file=sys.stderr)
                continue
        except SyntaxError as e:
            print(f"skip {path}: syntax error ({e})", file=sys.stderr)
            continue
        if summ is None:
            summ = _summarizer(args)
        new_source, sites, texts = document_source(
            source,
            lambda xs: process_generated(summ.summarize(xs, batch_size=args.batch_size), not args.full),
            only=args.only,
        )
        if not sites:
            continue
        changed += len(sites)
        if args.write:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(new_source)
            print(f"{path}: wrote {len(sites)} docstring(s)", file=sys.stderr)
        else:
            sys.stdout.write(unified_diff(source, new_source, path))
    if changed and not args.write:
        print(f"\n{changed} docstring(s) proposed. Review the diff, then rerun with --write to apply.",
              file=sys.stderr)
    return 0


def _inputs(args, files) -> List[tuple]:
    """Return (label, text) pairs from file arguments or stdin. Folders and wildcards are not expanded here."""
    if not files or files == ["-"]:
        return [("stdin", _read_text("-"))]
    return [(f, _read_text(f)) for f in files]


def _run_batch(args, pairs, only_summary: bool, empty_msg: str) -> int:
    pairs = [(label, text) for label, text in pairs if text.strip()]
    if not pairs:
        print("No input.", file=sys.stderr)
        return 1
    summ = _summarizer(args)
    results = process_generated(summ.summarize([t for _, t in pairs], batch_size=args.batch_size, progress=False),
                                only_summary)
    for (label, _), text in zip(pairs, results):
        text = text or empty_msg
        print(f"{os.path.basename(label)}: {text}" if len(pairs) > 1 else text)
    return 0


def cmd_snippet(args) -> int:
    return _run_batch(args, _inputs(args, args.files), not args.full, "(no summary sentence generated)")


def cmd_bug(args) -> int:
    if args.text:
        pairs = [("text", args.text)]
    else:
        pairs = _inputs(args, args.files)
    return _run_batch(args, pairs, False, "(no title generated)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="docgen", description=__doc__)
    p.add_argument("--version", action="version", version=f"docgen {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doc", help="Propose docstrings for undocumented Python functions (diff by default).")
    d.add_argument("files", nargs="+", help="Python files, folders (searched recursively), or wildcards")
    d.add_argument("--write", action="store_true", help="Apply the docstrings in place (default: print a diff).")
    d.add_argument("--only", default=None, help="Only document the function with this name.")
    d.add_argument("--full", action="store_true",
                   help="Keep the generated parameter and returns sections too. Default: summary line only, "
                        "because the parameter text is often generic or wrong.")
    d.add_argument("--summary-only", action="store_true", help=argparse.SUPPRESS)  # kept for old scripts; now the default
    _add_model_args(d)
    d.set_defaults(func=cmd_doc)

    s = sub.add_parser("snippet", help="Summarize code snippets in any language (files or stdin).")
    s.add_argument("files", nargs="*", help="One or more files with code, or - for stdin. Loads the model once.")
    s.add_argument("--full", action="store_true",
                   help="Keep tags such as @param and @return. Default: summary sentence only.")
    s.add_argument("--summary-only", action="store_true", help=argparse.SUPPRESS)
    _add_model_args(s)
    s.set_defaults(func=cmd_snippet)

    b = sub.add_parser("bug", help="Write a one-line title for bug report bodies (files, --text, or stdin).")
    b.add_argument("files", nargs="*", help="One or more files with bug reports, or - for stdin.")
    b.add_argument("--text", default=None, help="Bug report text given directly.")
    _add_model_args(b)
    b.set_defaults(func=cmd_bug)
    return p


def main(argv: List[str] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
