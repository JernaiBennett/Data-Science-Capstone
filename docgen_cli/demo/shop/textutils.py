"""Small text helpers."""

import re


def slugify(text):
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
    return re.sub(r"[\s-]+", "-", text)


def truncate(text, limit, suffix="..."):
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))] + suffix


def is_palindrome(text):
    cleaned = [c.lower() for c in text if c.isalnum()]
    return cleaned == cleaned[::-1]


def clamp(x, lo, hi): return max(lo, min(x, hi))


def word_frequencies(text):
    counts = {}
    for word in re.findall(r"[a-zA-Z']+", text.lower()):
        counts[word] = counts.get(word, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
