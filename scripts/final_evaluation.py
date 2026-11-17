"""Final evaluation: confidence intervals and error analysis for the three-model comparison.

Reads the saved test-set predictions of the three models (no GPU needed), then writes tables,
a figure, and example files to an output folder.

Usage (PowerShell, from the repository root):

    pip install pandas numpy matplotlib nltk rouge_score sacrebleu
    python scripts\\final_evaluation.py ^
        --full  results\\full_finetune_v2\\predictions_full_ft.csv.gz ^
        --lora  results\\lora_v2\\predictions_lora.csv.gz ^
        --small "PATH\\TO\\predictions_small.csv" ^
        --out   results\\final_comparison

What it computes
  1. Point scores (BLEU-4, ROUGE-L, METEOR) per model, overall and per group, computed the same way
     as the training notebooks, and a check against the numbers reported in the README.
  2. 95% confidence intervals from a paired bootstrap over test examples (same resampled examples for
     every model), and pairwise differences between models with their intervals.
  3. Error analysis: output shape (empty, tag-only, parameter sections, repeated sections, length),
     score by input length, and example files of the largest wins and losses.

Metric definitions
  BLEU-4: corpus BLEU, max order 4, no smoothing, 13a tokenization, one reference (as in Hugging Face
          `evaluate`). Computed from per-example n-gram counts so it can be resampled.
  ROUGE-L: mean per-example F-measure (rouge_score, no stemming).
  METEOR: mean per-example NLTK METEOR (default alpha 0.9, beta 3, gamma 0.5).
"""

import argparse
import math
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

# Numbers reported in the README, used to check that this script reproduces them.
EXPECTED = {
    "full": {"BLEU-4": 16.40, "ROUGE-L": 39.89, "METEOR": 37.54},
    "lora": {"BLEU-4": 13.32, "ROUGE-L": 37.79, "METEOR": 35.29},
    "small": {"BLEU-4": 13.44, "ROUGE-L": 37.07, "METEOR": 34.71},
}
LABELS = {"full": "Full fine-tune", "lora": "LoRA", "small": "CodeT5-small"}
METRICS = ["BLEU-4", "ROUGE-L", "METEOR"]


# ---------------------------------------------------------------- metrics

def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu_stats(preds, refs):
    """Per-example sufficient statistics: clipped matches 1-4, totals 1-4, pred length, ref length."""
    from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

    tok = Tokenizer13a()
    out = np.zeros((len(preds), 10), dtype=np.float64)
    for i, (p, r) in enumerate(zip(preds, refs)):
        pt, rt = tok(p).split(), tok(r).split()
        for n in range(1, 5):
            pn, rn = _ngrams(pt, n), _ngrams(rt, n)
            out[i, n - 1] = sum(min(c, rn[g]) for g, c in pn.items())
            out[i, 3 + n] = max(len(pt) - n + 1, 0)
        out[i, 8], out[i, 9] = len(pt), len(rt)
    return out


def bleu_from_stats(s):
    m, t, pl, rl = s[:4], s[4:8], s[8], s[9]
    if pl == 0 or np.any(m == 0):
        return 0.0
    bp = 1.0 if pl > rl else math.exp(1 - rl / pl)
    return 100 * bp * math.exp(sum(math.log(m[k] / t[k]) for k in range(4)) / 4)


def rouge_l_per_example(preds, refs):
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return np.array([100 * sc.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)])


def meteor_per_example(preds, refs):
    import nltk
    from nltk import word_tokenize
    from nltk.translate import meteor_score

    for pkg in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)
    return np.array([
        100 * meteor_score.meteor_score([word_tokenize(r)], word_tokenize(p), alpha=0.9, beta=3, gamma=0.5)
        for p, r in zip(preds, refs)
    ])


# ---------------------------------------------------------------- bootstrap

def bootstrap(per, groups_idx, boot, seed):
    """Paired bootstrap. per[model] = dict(bleu=(N,10) array, rouge=(N,), meteor=(N,) or None).
    Returns {model: {subset: {metric: array(boot)}}} using the same resampled indices for every model."""
    rng = np.random.default_rng(seed)
    out = {m: {} for m in per}
    for subset, idx in groups_idx.items():
        n = len(idx)
        draws = rng.integers(0, n, size=(boot, n))
        for m, d in per.items():
            b = np.array([bleu_from_stats(d["bleu"][idx][row].sum(0)) for row in draws])
            r = d["rouge"][idx][draws].mean(1)
            me = d["meteor"][idx][draws].mean(1) if d["meteor"] is not None else None
            out[m].setdefault(subset, {})
            out[m][subset] = {"BLEU-4": b, "ROUGE-L": r, **({"METEOR": me} if me is not None else {})}
    return out


def ci(a):
    return np.percentile(a, 2.5), np.percentile(a, 97.5)


# ---------------------------------------------------------------- error analysis

_TAG = re.compile(r"^\s*(?:@\w+|:param|:returns?:|Args:|Returns:)", re.M)
_NUMPY = re.compile(r"^\s*[A-Za-z][A-Za-z ]*\n\s*-{3,}", re.M)


def shape_flags(pred):
    p = pred if isinstance(pred, str) else ""
    stripped = p.strip()
    lines = [l.strip() for l in stripped.split("\n") if l.strip()]
    headers = [m.group(0).strip().split("\n")[0].strip().lower() for m in _NUMPY.finditer(p)]
    first_block = stripped.split("\n\n")[0].strip()
    return {
        "empty": stripped == "",
        "tag_only": bool(stripped) and bool(_TAG.match(stripped)),
        "has_tags_or_sections": bool(_TAG.search(p)) or bool(_NUMPY.search(p)),
        "repeated_section": len(headers) != len(set(headers)),
        "repeated_line": len(lines) != len(set(lines)) and any(
            lines[i] == lines[i + 1] for i in range(len(lines) - 1)),
        "multi_line": "\n" in stripped,
        "words": len(stripped.split()),
        "first_block_words": len(first_block.split()),
    }


def shape_table(df, name):
    rows = []
    for g, sub in [("ALL", df)] + list(df.groupby("group")):
        f = pd.DataFrame([shape_flags(p) for p in sub["prediction"]])
        ref_words = sub["reference"].map(lambda s: len(str(s).split()))
        rows.append({
            "model": name, "group": g, "n": len(sub),
            "empty_%": 100 * f["empty"].mean(),
            "tag_only_%": 100 * f["tag_only"].mean(),
            "has_tags_or_sections_%": 100 * f["has_tags_or_sections"].mean(),
            "repeated_section_%": 100 * f["repeated_section"].mean(),
            "repeated_line_%": 100 * f["repeated_line"].mean(),
            "multi_line_%": 100 * f["multi_line"].mean(),
            "median_pred_words": float(f["words"].median()),
            "median_ref_words": float(ref_words.median()),
        })
    return pd.DataFrame(rows).round(2)


# ---------------------------------------------------------------- main

def load(path, name):
    df = pd.read_csv(path)
    need = {"reference", "prediction", "group"}
    if not need <= set(df.columns):
        sys.exit(f"{name}: {path} must have columns {sorted(need)} (found {list(df.columns)})")
    df["prediction"] = df["prediction"].fillna("").astype(str)
    df["reference"] = df["reference"].fillna("").astype(str)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--small", required=True)
    ap.add_argument("--out", default="results/final_comparison")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (default 2000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-meteor", action="store_true", help="skip METEOR (needs NLTK data downloads)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    dfs = {"full": load(args.full, "full"), "lora": load(args.lora, "lora"), "small": load(args.small, "small")}
    base = dfs["full"]
    for k in ("lora", "small"):
        if len(dfs[k]) != len(base) or not (dfs[k]["reference"].values == base["reference"].values).all():
            sys.exit(f"{k}: test examples do not line up with the full fine-tune file (different order or set). "
                     "All three prediction files must come from the same test set in the same order.")
    print(f"Loaded {len(base)} aligned test examples; groups: {sorted(base['group'].unique())}")

    # ---- per-example scores (the slow part: METEOR takes a few minutes per model)
    per = {}
    for k, df in dfs.items():
        preds = [p if p.strip() else " " for p in df["prediction"]]
        refs = [r if r.strip() else " " for r in df["reference"]]
        print(f"Scoring {LABELS[k]} ...", flush=True)
        per[k] = {
            "bleu": bleu_stats(preds, refs),
            "rouge": rouge_l_per_example(preds, refs),
            "meteor": None if args.no_meteor else meteor_per_example(preds, refs),
        }
    metrics = [m for m in METRICS if not (m == "METEOR" and args.no_meteor)]

    groups_idx = {"ALL": np.arange(len(base))}
    for g in sorted(base["group"].unique()):
        groups_idx[g] = np.where(base["group"].values == g)[0]

    # ---- point scores and the check against the README
    def point(k, idx):
        d = per[k]
        out = {"BLEU-4": bleu_from_stats(d["bleu"][idx].sum(0)), "ROUGE-L": d["rouge"][idx].mean()}
        if d["meteor"] is not None:
            out["METEOR"] = d["meteor"][idx].mean()
        return out

    rows, check = [], []
    for k in dfs:
        for g, idx in groups_idx.items():
            rows.append({"model": LABELS[k], "group": g, "n": len(idx), **point(k, idx)})
        got = point(k, groups_idx["ALL"])
        for m in metrics:
            ok = abs(got[m] - EXPECTED[k][m]) <= 0.10
            check.append({"model": LABELS[k], "metric": m, "this_script": round(got[m], 2),
                          "readme": EXPECTED[k][m], "status": "OK" if ok else "MISMATCH"})
    pd.DataFrame(rows).round(2).to_csv(f"{args.out}/point_scores.csv", index=False)
    chk = pd.DataFrame(check)
    chk.to_csv(f"{args.out}/check_against_readme.csv", index=False)
    print("\nCheck against README numbers (tolerance 0.10):\n", chk.to_string(index=False))
    if (chk["status"] == "MISMATCH").any():
        print("\nWARNING: some scores differ from the README. Do not use the intervals below until this is "
              "explained (most often: different library versions, or prediction files from a different run).")

    # ---- bootstrap
    print(f"\nBootstrapping ({args.boot} resamples) ...", flush=True)
    bs = bootstrap(per, groups_idx, args.boot, args.seed)
    ci_rows, diff_rows = [], []
    pairs = [("full", "lora"), ("full", "small"), ("lora", "small")]
    for g in groups_idx:
        for k in dfs:
            for m in metrics:
                lo, hi = ci(bs[k][g][m])
                ci_rows.append({"model": LABELS[k], "group": g, "n": len(groups_idx[g]), "metric": m,
                                "score": point(k, groups_idx[g])[m], "ci_low": lo, "ci_high": hi})
        for a, b in pairs:
            for m in metrics:
                d = bs[a][g][m] - bs[b][g][m]
                lo, hi = ci(d)
                diff_rows.append({
                    "comparison": f"{LABELS[a]} - {LABELS[b]}", "group": g, "n": len(groups_idx[g]), "metric": m,
                    "difference": point(a, groups_idx[g])[m] - point(b, groups_idx[g])[m],
                    "ci_low": lo, "ci_high": hi,
                    "interval_excludes_0": bool(lo > 0 or hi < 0)})
    cis = pd.DataFrame(ci_rows).round(2)
    diffs = pd.DataFrame(diff_rows).round(2)
    cis.to_csv(f"{args.out}/confidence_intervals.csv", index=False)
    diffs.to_csv(f"{args.out}/pairwise_differences.csv", index=False)
    print("\nOverall 95% CIs:\n", cis[cis.group == "ALL"].drop(columns="group").to_string(index=False))
    print("\nOverall paired differences:\n",
          diffs[diffs.group == "ALL"].drop(columns="group").to_string(index=False))

    # ---- error analysis
    shape = pd.concat([shape_table(df, LABELS[k]) for k, df in dfs.items()])
    shape.to_csv(f"{args.out}/output_shape.csv", index=False)

    n_src = base["source"].astype(str).str.len() if "source" in base else None
    if n_src is not None:
        bucket = pd.qcut(n_src, 4, labels=["shortest 25%", "25-50%", "50-75%", "longest 25%"])
        rows = []
        for k in dfs:
            s = pd.Series(per[k]["rouge"])
            for b in bucket.cat.categories:
                rows.append({"model": LABELS[k], "input_length": b, "n": int((bucket == b).sum()),
                             "ROUGE-L": s[(bucket == b).values].mean()})
        pd.DataFrame(rows).round(2).to_csv(f"{args.out}/rouge_by_input_length.csv", index=False)

    non_ascii = base["reference"].map(lambda s: sum(ord(c) > 127 for c in s) > 0.3 * max(len(s), 1)).values
    rows = []
    for k in dfs:
        for label, mask in (("mostly non-ASCII reference", non_ascii), ("ASCII reference", ~non_ascii)):
            rows.append({"model": LABELS[k], "subset": label, "n": int(mask.sum()),
                         "ROUGE-L": per[k]["rouge"][mask].mean() if mask.any() else float("nan")})
    pd.DataFrame(rows).round(2).to_csv(f"{args.out}/rouge_non_english_references.csv", index=False)

    # example files: biggest full-vs-LoRA and full-vs-small gaps in per-example ROUGE-L, and zero-score cases
    def clip(s, n=500):
        s = str(s)
        return s if len(s) <= n else s[:n] + " ..."

    ex = pd.DataFrame({
        "group": base["group"].values,
        "source": base.get("source", pd.Series([""] * len(base))).map(clip).values,
        "reference": base["reference"].map(clip).values,
        "full": dfs["full"]["prediction"].map(clip).values,
        "lora": dfs["lora"]["prediction"].map(clip).values,
        "small": dfs["small"]["prediction"].map(clip).values,
        "rouge_full": per["full"]["rouge"].round(1),
        "rouge_lora": per["lora"]["rouge"].round(1),
        "rouge_small": per["small"]["rouge"].round(1),
    })
    ex["gap_full_minus_lora"] = ex["rouge_full"] - ex["rouge_lora"]
    ex["gap_full_minus_small"] = ex["rouge_full"] - ex["rouge_small"]
    ex.sort_values("gap_full_minus_lora", ascending=False).head(30).to_csv(
        f"{args.out}/examples_full_beats_lora.csv", index=False)
    ex.sort_values("gap_full_minus_lora").head(30).to_csv(f"{args.out}/examples_lora_beats_full.csv", index=False)
    ex[ex["rouge_full"] == 0].sample(min(40, int((ex["rouge_full"] == 0).sum())), random_state=args.seed).to_csv(
        f"{args.out}/examples_full_zero_rouge.csv", index=False)
    bug = ex[ex["group"].str.contains("bug", case=False)]
    if len(bug):
        bug.sample(min(40, len(bug)), random_state=args.seed).to_csv(f"{args.out}/examples_bug_reports.csv",
                                                                   index=False)
    print(f"\nShare of examples where full fine-tune ROUGE-L > LoRA: "
          f"{100 * (ex['rouge_full'] > ex['rouge_lora']).mean():.1f}%  "
          f"(tie {100 * (ex['rouge_full'] == ex['rouge_lora']).mean():.1f}%)")

    # ---- figure: ROUGE-L by group with 95% CIs
    if not args.no_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            sub = cis[cis.metric == "ROUGE-L"]
            order = [g for g in groups_idx]
            fig, ax = plt.subplots(figsize=(10, 4.5))
            w = 0.26
            for j, k in enumerate(dfs):
                s = sub[sub.model == LABELS[k]].set_index("group").loc[order]
                x = np.arange(len(order)) + (j - 1) * w
                ax.bar(x, s["score"], w, label=LABELS[k],
                       yerr=[s["score"] - s["ci_low"], s["ci_high"] - s["score"]], capsize=2)
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels(order, rotation=30, ha="right")
            ax.set_ylabel("ROUGE-L (higher is better)")
            ax.set_title("ROUGE-L by group with 95% bootstrap intervals")
            ax.legend()
            fig.tight_layout()
            fig.savefig(f"{args.out}/rouge_l_by_group.png", dpi=200)
        except ImportError:
            print("matplotlib not installed; skipped the figure.")

    print(f"\nDone. Files written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
