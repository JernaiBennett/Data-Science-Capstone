import ast
import os

from docgen.extract import find_undocumented

DEMO = os.path.join(os.path.dirname(__file__), "..", "demo", "shop")

EXPECTED = {
    "pricing.py": ["apply_discount", "add_tax", "split_bill", "cheapest"],
    "cart.py": ["__init__", "add_item", "remove_item", "subtotal", "total", "item_count", "__len__"],
    "textutils.py": ["slugify", "truncate", "is_palindrome", "word_frequencies"],
    "tasks.py": ["retry", "attempt", "gather_limited", "run"],
}


def test_demo_files_match_readme_table():
    for name, expected in EXPECTED.items():
        with open(os.path.join(DEMO, name), encoding="utf-8") as f:
            src = f.read()
        ast.parse(src)
        assert [s.name for s in find_undocumented(src)] == expected, name
