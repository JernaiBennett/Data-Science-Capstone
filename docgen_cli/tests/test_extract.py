import ast
import sys
import types

import pytest

from docgen import cli
from docgen.extract import (
    apply_docstrings,
    document_source,
    find_undocumented,
    format_docstring,
    unified_diff,
)

SAMPLE = '''import os


def add(a, b):
    return a + b


def has_doc(x):
    """Already documented."""
    return x


def one_liner(x): return x


async def fetch(url):
    # comment before the first statement
    data = await get(url)
    return data


class Box:
    def __init__(self, v):
        self.v = v

    def get(
        self,
        default=None,
    ):
        def inner():
            return 1
        return self.v or default
'''


def fake(texts):
    return [f"Summary {i}." for i, _ in enumerate(texts)]


def test_find_undocumented():
    names = [s.name for s in find_undocumented(SAMPLE)]
    assert names == ["add", "fetch", "__init__", "get", "inner"]
    assert "has_doc" not in names and "one_liner" not in names


def test_method_source_is_dedented_and_parses():
    sites = {s.name: s for s in find_undocumented(SAMPLE)}
    src = sites["__init__"].source
    assert src.startswith("def __init__")
    ast.parse(src)


def test_insert_keeps_file_valid_and_adds_docstrings():
    new, sites, texts = document_source(SAMPLE, fake)
    tree = ast.parse(new)
    assert len(sites) == 5
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != "one_liner":
            assert ast.get_docstring(node), node.name


def test_docstring_indentation_and_position():
    new, _, _ = document_source(SAMPLE, fake)
    lines = new.splitlines()
    i = lines.index("def add(a, b):")
    assert lines[i + 1] == '    """Summary 0."""'
    j = lines.index("    # comment before the first statement")
    assert '"""' in lines[j + 1]  # docstring lands after the comment, before the first statement


def test_idempotent_second_run_finds_only_skipped():
    new, _, _ = document_source(SAMPLE, fake)
    remaining = [s.name for s in find_undocumented(new)]
    assert remaining == []  # one_liner stays skipped because its body is inline


def test_only_filter():
    new, sites, _ = document_source(SAMPLE, fake, only="add")
    assert [s.name for s in sites] == ["add"]
    assert new.count('"""') == 2 + 2  # has_doc keeps its own pair, plus the new pair


def test_escaping_and_long_text():
    out = format_docstring('Return "x" and a\\b', "    ")
    assert out == ['    """Return "x" and a\\\\b"""\n']
    out = format_docstring('ends with a quote "', "    ")
    assert out[-1].rstrip("\n").endswith('" """')
    long = " ".join(["word"] * 60)
    out = format_docstring(long, "    ")
    assert len(out) > 2 and out[0].startswith('    """') and out[-1] == '    """\n'
    assert format_docstring('has """ inside', "")[0].count('"""') == 2
    assert format_docstring("   ", "    ") == []


def test_no_trailing_newline_file():
    src = "def f(x):\n    return x"
    new, _, _ = document_source(src, lambda xs: ["Return x."])
    ast.parse(new)
    assert '"""Return x."""' in new


def test_diff_output():
    new, _, _ = document_source("def f(x):\n    return x\n", lambda xs: ["Return x."])
    d = unified_diff("def f(x):\n    return x\n", new, "f.py")
    assert d.startswith("--- a/f.py") and '+    """Return x."""' in d


class FakeSummarizer:
    def __init__(self, *a, **k):
        pass

    def summarize(self, texts, batch_size=8, progress=True):
        return [f"Summary of {t.splitlines()[0]}" for t in texts]


@pytest.fixture
def patched(monkeypatch):
    mod = types.ModuleType("docgen.model")
    mod.Summarizer = FakeSummarizer
    monkeypatch.setitem(sys.modules, "docgen.model", mod)


def test_cli_doc_diff_then_write(tmp_path, capsys, patched):
    f = tmp_path / "m.py"
    f.write_text("def f(x):\n    return x\n", encoding="utf-8")
    assert cli.main(["doc", str(f)]) == 0
    out = capsys.readouterr().out
    assert "+    \"\"\"Summary of def f(x):\"\"\"" in out
    assert f.read_text() == "def f(x):\n    return x\n"  # diff mode does not modify the file
    assert cli.main(["doc", str(f), "--write"]) == 0
    assert '"""Summary of def f(x):"""' in f.read_text()
    ast.parse(f.read_text())


def test_cli_skips_non_python_and_syntax_errors(tmp_path, capsys, patched):
    js = tmp_path / "a.js"
    js.write_text("function f(){}", encoding="utf-8")
    bad = tmp_path / "b.py"
    bad.write_text("def (:\n", encoding="utf-8")
    assert cli.main(["doc", str(js), str(bad)]) == 0
    err = capsys.readouterr().err
    assert "Python files only" in err and "syntax error" in err


def test_cli_snippet_and_bug(tmp_path, capsys, patched):
    code = tmp_path / "x.go"
    code.write_text("func Add(a, b int) int { return a + b }", encoding="utf-8")
    assert cli.main(["snippet", str(code)]) == 0
    assert "Summary of func Add" in capsys.readouterr().out
    assert cli.main(["bug", "--text", "App closes when saving."]) == 0
    assert "Summary of App closes when saving." in capsys.readouterr().out
    assert cli.main(["bug", "--text", "   "]) == 1


def test_missing_model_folder_gives_clear_error(tmp_path, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    code = tmp_path / "x.py"
    code.write_text("def f(x):\n    return x\n", encoding="utf-8")
    rc = cli.main(["snippet", str(code), "--model", str(tmp_path / "missing")])
    assert rc == 2
    assert "Model folder not found" in capsys.readouterr().err


def test_cli_accepts_folders_and_wildcards(tmp_path, capsys, patched):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def a(x):\n    return x\n", encoding="utf-8")
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "skip.py").write_text("def s(x):\n    return x\n", encoding="utf-8")
    (tmp_path / "pkg" / "readme.md").write_text("x", encoding="utf-8")
    assert cli._expand_paths([str(tmp_path / "pkg")]) == [str(tmp_path / "pkg" / "a.py")]
    assert cli._expand_paths([str(tmp_path / "pkg" / "*.py")]) == [str(tmp_path / "pkg" / "a.py")]
    assert cli.main(["doc", str(tmp_path / "nothing*.py")]) == 1


NUMPY = (
    "Apply a discount to a price.\n\nParameters\n----------\nprice : float\n    The price.\n"
    "percent : float\n    The percentage.\n\nReturns\n-------\nfloat\n    The result.\n\n"
    "Returns\n-------\nfloat\n    The result."
)


def test_clean_drops_repeated_sections_and_lines():
    from docgen.extract import clean_generated

    out = clean_generated(NUMPY)
    assert out.count("Returns") == 1
    assert clean_generated("a\na\nb") == "a\nb"


def test_summary_only():
    from docgen.extract import summary_only

    assert summary_only(NUMPY) == "Apply a discount to a price."
    assert summary_only("Returns the average.\n\n@param {Array} values") == "Returns the average."
    assert summary_only("@param array $numbers\n\n@return int") == ""
    assert summary_only("Line one\nline two\n@param x") == "Line one line two"


def test_multiline_docstring_keeps_structure_and_indentation():
    from docgen.extract import process_generated

    src = "class A:\n    def f(self, price, percent):\n        return price\n"
    new, _, _ = document_source(src, lambda xs: process_generated([NUMPY]))
    ast.parse(new)
    doc = ast.get_docstring([n for n in ast.walk(ast.parse(new)) if isinstance(n, ast.FunctionDef)][0])
    assert doc.startswith("Apply a discount to a price.")
    assert "Parameters\n----------\nprice : float\n    The price." in doc
    assert doc.count("Returns") == 1
    assert "\n        Parameters" in new  # indented at body level


def test_doc_defaults_to_summary_only_and_full_keeps_sections(tmp_path, patched, monkeypatch):
    f = tmp_path / "m.py"
    f.write_text("def f(x):\n    return x\n", encoding="utf-8")
    monkeypatch.setattr(FakeSummarizer, "summarize",
                        lambda self, texts, batch_size=8, progress=True: [NUMPY for _ in texts])
    assert cli.main(["doc", str(f), "--write"]) == 0
    assert '"""Apply a discount to a price."""' in f.read_text()
    f.write_text("def f(x):\n    return x\n", encoding="utf-8")
    assert cli.main(["doc", str(f), "--write", "--full"]) == 0
    assert "Parameters" in f.read_text() and f.read_text().count("Returns") == 1


def test_snippet_and_bug_accept_several_files(tmp_path, capsys, patched):
    a = tmp_path / "a.go"; a.write_text("func A() {}", encoding="utf-8")
    b = tmp_path / "b.go"; b.write_text("func B() {}", encoding="utf-8")
    assert cli.main(["snippet", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "a.go: Summary of func A" in out and "b.go: Summary of func B" in out
    assert cli.main(["bug", str(a), str(b)]) == 0
    assert "b.go: Summary of func B" in capsys.readouterr().out


def test_tag_only_snippet_reports_no_summary(tmp_path, capsys, patched, monkeypatch):
    monkeypatch.setattr(FakeSummarizer, "summarize",
                        lambda self, texts, batch_size=8, progress=True: ["@param array $n\n\n@return int"])
    a = tmp_path / "a.php"; a.write_text("function f($n){}", encoding="utf-8")
    assert cli.main(["snippet", str(a)]) == 0
    assert "(no summary sentence generated)" in capsys.readouterr().out
    assert cli.main(["snippet", str(a), "--full"]) == 0
    assert "@param array $n" in capsys.readouterr().out


SPACED = (
    "Apply a discount to a price.\n Parameters\n ----------\n price : float\n     The price.\n\n"
    " Returns\n -------\n price : float\n     The price.\n\n Returns\n -------\n price : float\n     The price."
)


def test_clean_handles_one_space_indent_on_later_lines():
    from docgen.extract import clean_generated, summary_only

    out = clean_generated(SPACED)
    assert out.count("Returns") == 1
    assert "\nParameters\n----------\nprice : float\n    The price." in out
    assert summary_only(SPACED) == "Apply a discount to a price."
    assert summary_only("Returns a list.\n\n @param [Integer] n\n\n @return [Array]") == "Returns a list."
