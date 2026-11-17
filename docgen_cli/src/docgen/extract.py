"""Find Python functions without docstrings and insert generated ones.

Model inputs match training: the raw function source, with no docstring in it,
dedented so methods look like top-level functions.
"""

import ast
import difflib
import re
import textwrap
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence


@dataclass
class FuncSite:
    name: str
    lineno: int          # 1-based line of the `def` keyword
    source: str          # dedented source of the whole function (decorators excluded)
    insert_index: int    # 0-based line index where the docstring is inserted
    indent: str          # indentation of the function body


def find_undocumented(source: str) -> List[FuncSite]:
    """Return every function or method that has no docstring and a body on its own line."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    sites: List[FuncSite] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if ast.get_docstring(node, clean=False) is not None:
            continue
        first = node.body[0]
        body_line = lines[first.lineno - 1]
        before_stmt = body_line[: first.col_offset]
        if before_stmt.strip():
            continue  # body shares a line with the header, for example `def f(): return 1`
        func_src = "".join(lines[node.lineno - 1 : node.end_lineno])
        sites.append(
            FuncSite(
                name=node.name,
                lineno=node.lineno,
                source=textwrap.dedent(func_src),
                insert_index=first.lineno - 1,
                indent=before_stmt,
            )
        )
    sites.sort(key=lambda s: s.lineno)
    return sites


# ---------- cleaning generated text ----------

_SECTION_SPLIT = re.compile(r"\n(?=[ \t]*[A-Za-z][A-Za-z ]*\n[ \t]*-{3,}[ \t]*(?:\n|$))")
_SECTION_HEAD = re.compile(r"[ \t]*([A-Za-z][A-Za-z ]*)\n[ \t]*-{3,}")


def _normalize(text: str) -> str:
    """Unify newlines and remove the stray common indent the model puts on lines after the first."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [l.rstrip() for l in text.split("\n")]
    if len(lines) > 1:
        rest = textwrap.dedent("\n".join(lines[1:])).split("\n")
        lines = [lines[0].strip()] + rest
    return "\n".join(lines)


def clean_generated(text: str) -> str:
    """Tidy model output: normalize indentation, drop repeated lines and repeated NumPy sections."""
    lines: List[str] = []
    for line in _normalize(text).split("\n"):
        if lines and line.strip() and line == lines[-1]:
            continue  # identical consecutive line
        lines.append(line)
    text = "\n".join(lines)

    seen = set()
    parts = []
    for part in _SECTION_SPLIT.split(text):
        m = _SECTION_HEAD.match(part)
        if m:
            key = m.group(1).strip().lower()
            if key in seen:
                continue  # a section with this title already appeared
            seen.add(key)
        parts.append(part)
    return "\n".join(parts).strip()


_STOP_LINE = re.compile(r"^[ \t]*(?:@\w+|:param|:returns?:|Args:|Returns:|Arguments:)", re.M)
_NUMPY_HEAD = re.compile(r"^[ \t]*[A-Za-z][A-Za-z ]*\n[ \t]*-{3,}", re.M)


def summary_only(text: str) -> str:
    """Keep only the opening summary: text before the first blank line, tag, or section header."""
    text = _normalize(text)
    cut = len(text)
    for pattern in (re.compile(r"\n\s*\n"), _STOP_LINE, _NUMPY_HEAD):
        m = pattern.search(text)
        if m and m.start() > 0:
            cut = min(cut, m.start())
    first = text[:cut].strip()
    if _STOP_LINE.match(first) or _NUMPY_HEAD.match(first):
        return ""
    return " ".join(first.split())


def process_generated(texts: Sequence[str], only_summary: bool = False) -> List[str]:
    out = []
    for t in texts:
        t = clean_generated(t)
        out.append(summary_only(t) if only_summary else t)
    return out


# ---------- inserting ----------

def format_docstring(text: str, indent: str, width: int = 88) -> List[str]:
    """Turn generated text into docstring lines at the given indentation, keeping its line breaks."""
    text = text.replace("\r\n", "\n").strip("\n")
    text = text.replace("\\", "\\\\").replace('"""', "'''")
    if not text.strip():
        return []

    raw = [l.rstrip() for l in text.split("\n")]
    if len(raw) == 1:
        wrapped = textwrap.wrap(" ".join(raw[0].split()), width=max(40, width - len(indent) - 6))
        raw = wrapped or [raw[0].strip()]
    else:
        first = raw[0].strip()
        rest = textwrap.dedent("\n".join(raw[1:])).split("\n")
        while rest and not rest[-1].strip():
            rest.pop()
        raw = [first] + rest

    last = raw[-1] + (" " if raw[-1].endswith('"') else "")
    if len(raw) == 1:
        return [f'{indent}"""{last}"""\n']
    out = [f'{indent}"""{raw[0]}\n']
    for line in raw[1:-1]:
        out.append(f"{indent}{line}\n" if line.strip() else "\n")
    out.append(f"{indent}{last}\n")
    out.append(f'{indent}"""\n')
    return out


def apply_docstrings(source: str, sites: Sequence[FuncSite], texts: Sequence[str]) -> str:
    """Insert docstrings bottom to top so earlier line numbers stay valid."""
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    pairs = sorted(zip(sites, texts), key=lambda p: p[0].insert_index, reverse=True)
    for site, text in pairs:
        new_lines = format_docstring(text, site.indent)
        if new_lines:
            lines[site.insert_index : site.insert_index] = new_lines
    return "".join(lines)


def document_source(
    source: str,
    summarize: Callable[[List[str]], List[str]],
    only: Optional[str] = None,
):
    """Generate docstrings for one file's source. Returns (new_source, sites, texts)."""
    sites = find_undocumented(source)
    if only:
        sites = [s for s in sites if s.name == only]
    if not sites:
        return source, [], []
    texts = summarize([s.source for s in sites])
    return apply_docstrings(source, sites, texts), sites, texts


def unified_diff(old: str, new: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
