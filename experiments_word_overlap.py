import csv
import itertools
import os
import warnings
from collections import Counter
from math import exp, log, sqrt

import nltk
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from factchecking_main import read_passages, read_fact_examples
from factcheck import WordRecallThresholdFactChecker
from overlap_metrics import METRICS
from preprocessing import tokenize

warnings.filterwarnings("ignore")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

NORMS = ["none", "stem", "lemma", "lemma+stem"]
STOPS = [False, True]                      # remove_stopwords
POOLS = ["max", "union"]

_stem, _lem = PorterStemmer(), WordNetLemmatizer()
_stem_cache, _lem_cache = {}, {}
_smooth = SmoothingFunction().method1


def stem(t):
    if t not in _stem_cache:
        _stem_cache[t] = _stem.stem(t)
    return _stem_cache[t]


def lemma(t):
    # no POS tagger, so try the verb form first ("won" -> "win"), then the noun form ("awards" -> "award")
    if t not in _lem_cache:
        _lem_cache[t] = _lem.lemmatize(_lem.lemmatize(t, "v"), "n")
    return _lem_cache[t]


def prep(text, norm, remove_stop):
    toks = tokenize(text, stem=False, remove_stopwords=remove_stop)     # lowercase + regex + optional stop words
    if norm in ("lemma", "lemma+stem"):
        toks = [lemma(t) for t in toks]
    if norm in ("stem", "lemma+stem"):
        toks = [stem(t) for t in toks]
    return toks


# ------------------------------------------------------------------ metrics (one pass computes all of them)
def ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def f_score(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def bleu_no_bp(f, p, n_max):
    """BLEU without the brevity penalty: geometric mean of the clipped 1..n_max-gram precisions of the fact
    (zero counts smoothed to 0.1 like nltk's method1). The penalty punishes 'hypothesis shorter than reference',
    which for a short fact vs a long passage is always true, so it mostly measures passage length."""
    total_log = 0.0
    for k in range(1, n_max + 1):
        rf, cp = ngrams(f, k), ngrams(p, k)
        match = sum(min(c, cp[g]) for g, c in rf.items())
        pk = (match if match else 0.1) / max(1, sum(rf.values()))
        total_log += log(pk) / n_max
    return exp(total_log)


def all_metrics(f, p, idf, idf_default):
    """f = fact tokens (reference), p = passage tokens (candidate). Returns {metric name: score}."""
    out = {}
    fs, ps = set(f), set(p)
    inter = len(fs & ps)
    prec = inter / len(ps) if ps else 0.0
    rec = inter / len(fs) if fs else 0.0
    out["precision"], out["recall"], out["f1"] = prec, rec, f_score(prec, rec)
    union = len(fs | ps)
    out["jaccard"] = inter / union if union else 0.0
    out["dice"] = 2 * inter / (len(fs) + len(ps)) if (fs or ps) else 0.0
    # TF-IDF (IDF computed over the fact's own retrieved passages)
    w = lambda t: idf.get(t, idf_default)
    tf_f, tf_p = Counter(f), Counter(p)
    num = sum(tf_f[t] * tf_p[t] * w(t) ** 2 for t in tf_f if t in tf_p)
    nf = sqrt(sum((c * w(t)) ** 2 for t, c in tf_f.items()))
    np_ = sqrt(sum((c * w(t)) ** 2 for t, c in tf_p.items()))
    out["tfidf_cosine"] = num / (nf * np_) if nf and np_ else 0.0
    tot = sum(w(t) for t in fs)
    out["tfidf_recall"] = sum(w(t) for t in fs & ps) / tot if tot else 0.0
    # ROUGE (reference = fact, candidate = passage; clipped n-gram overlap)
    for n in (1, 2):
        rf, cp = ngrams(f, n), ngrams(p, n)
        ov = sum(min(c, cp[g]) for g, c in rf.items())
        r = ov / sum(rf.values()) if rf else 0.0
        pr = ov / sum(cp.values()) if cp else 0.0
        out["rouge%d_r" % n] = r
        out["rouge%d_f" % n] = f_score(pr, r)
        if n == 1:
            out["rouge1_p"] = pr
    l = lcs_len(f, p)
    r, pr = (l / len(f) if f else 0.0), (l / len(p) if p else 0.0)
    out["rougeL_r"], out["rougeL_f"] = r, f_score(pr, r)
    # BLEU (hypothesis = fact, reference = passage; smoothed; includes the brevity penalty)
    for name, wts in (("bleu1", (1.0,)), ("bleu2", (0.5, 0.5)), ("bleu4", (0.25,) * 4)):
        out[name] = sentence_bleu([p], f, weights=wts, smoothing_function=_smooth) if f and p else 0.0
    for name, n in (("bleu1_nobp", 1), ("bleu2_nobp", 2), ("bleu4_nobp", 4)):
        out[name] = bleu_no_bp(f, p, n) if f and p else 0.0
    return out


METRIC_NAMES = ["precision", "recall", "f1", "jaccard", "dice", "tfidf_cosine", "tfidf_recall", "rouge1_p",
                "rouge1_r", "rouge1_f", "rouge2_r", "rouge2_f", "rougeL_r", "rougeL_f", "bleu1", "bleu2", "bleu4", "bleu1_nobp", "bleu2_nobp", "bleu4_nobp"]
FAMILY = {"precision": "precision/recall", "recall": "precision/recall", "f1": "precision/recall",
          "jaccard": "Jaccard", "dice": "Dice", "tfidf_cosine": "TF-IDF", "tfidf_recall": "TF-IDF",
          "rouge1_p": "ROUGE", "rouge1_r": "ROUGE", "rouge1_f": "ROUGE", "rouge2_r": "ROUGE", "rouge2_f": "ROUGE",
          "rougeL_r": "ROUGE", "rougeL_f": "ROUGE", "bleu1": "BLEU", "bleu2": "BLEU", "bleu4": "BLEU",
          "bleu1_nobp": "BLEU", "bleu2_nobp": "BLEU", "bleu4_nobp": "BLEU"}


def example_scores(fact, passages, norm, stop):
    """Score one example under one preprocessing variant, for both pooling modes."""
    f = prep(fact, norm, stop)
    ptoks = [prep(p["text"], norm, stop) for p in passages]
    if not f or not any(ptoks):
        zero = {m: 0.0 for m in METRIC_NAMES}
        return {"max": zero, "union": zero}
    df = Counter(t for toks in ptoks for t in set(toks))
    n = len(ptoks)
    idf = {t: log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
    idf_default = log(1 + n) + 1
    per_passage = [all_metrics(f, toks, idf, idf_default) for toks in ptoks]
    mx = {m: max(d[m] for d in per_passage) for m in METRIC_NAMES}
    merged = [t for toks in ptoks for t in toks]
    return {"max": mx, "union": all_metrics(f, merged, idf, idf_default)}


# ------------------------------------------------------------------ threshold search + cross-validation
def best_cut(scores, gold):
    """Best accuracy when predicting S for score >= cut; among equally good cuts take the median one."""
    d = np.unique(scores)
    accs = ((scores[:, None] >= d[None, :]) == gold[:, None]).mean(axis=0)
    best = accs.max()
    idx = np.flatnonzero(accs == best)
    return float(best), float(d[idx[len(idx) // 2]])


def cv_accuracy(scores, gold, repeats=3, folds=5):
    accs = []
    for seed in range(repeats):
        order = np.random.RandomState(seed).permutation(len(gold))
        correct = 0
        for k in range(folds):
            test = order[k::folds]
            train = np.setdiff1d(order, test)
            _, cut = best_cut(scores[train], gold[train])
            correct += int(((scores[test] >= cut) == gold[test]).sum())
        accs.append(correct / len(gold))
    return float(np.mean(accs))


def accuracy_at(scores, gold, threshold):
    return float(((np.asarray(scores) >= threshold) == gold).mean())


def run_grid(examples, gold):
    rows = []
    for norm in NORMS:
        for stop in STOPS:
            per_ex = [example_scores(ex.fact, ex.passages, norm, stop) for ex in examples]
            for pool in POOLS:
                for m in METRIC_NAMES:
                    scores = np.array([e[pool][m] for e in per_ex])
                    acc, cut = best_cut(scores, gold)
                    rows.append({"metric": m, "family": FAMILY[m], "normalisation": norm,
                                 "remove_stopwords": stop, "pool": pool, "dev_acc": round(acc, 4),
                                 "threshold": round(cut, 4), "cv_acc": round(cv_accuracy(scores, gold), 4)})
            print("done:", norm, "stopwords removed" if stop else "stopwords kept")
    return rows


def print_tables(rows):
    def mean(xs):
        return sum(xs) / len(xs)

    print("\n=== Top 15 configurations by dev accuracy ===")
    for r in sorted(rows, key=lambda r: -r["dev_acc"])[:15]:
        print("%.4f (cv %.4f) thr=%.3f  %-13s %-10s stop_removed=%-5s %s" % (
            r["dev_acc"], r["cv_acc"], r["threshold"], r["metric"], r["normalisation"], r["remove_stopwords"], r["pool"]))

    print("\n=== Best configuration per metric ===")
    for m in METRIC_NAMES:
        r = max((r for r in rows if r["metric"] == m), key=lambda r: (r["dev_acc"], r["cv_acc"]))
        mean_all = mean([x["dev_acc"] for x in rows if x["metric"] == m])
        print("%-13s best dev %.4f (cv %.4f) | %s, stop_removed=%s, %s, thr=%.3f | mean over all 16 configs %.4f" % (
            m, r["dev_acc"], r["cv_acc"], r["normalisation"], r["remove_stopwords"], r["pool"], r["threshold"], mean_all))

    print("\n=== Preprocessing effect (mean dev accuracy over all metrics and the other settings) ===")
    for key, vals in (("normalisation", NORMS), ("remove_stopwords", STOPS), ("pool", POOLS)):
        for v in vals:
            print("%-17s %-11s %.4f" % (key, v, mean([r["dev_acc"] for r in rows if r[key] == v])))

    print("\n=== Same, restricted to the strongest metrics (recall, rouge1_r, tfidf_recall) ===")
    strong = [r for r in rows if r["metric"] in ("recall", "rouge1_r", "tfidf_recall")]
    for key, vals in (("normalisation", NORMS), ("remove_stopwords", STOPS), ("pool", POOLS)):
        for v in vals:
            print("%-17s %-11s %.4f" % (key, v, mean([r["dev_acc"] for r in strong if r[key] == v])))


def check_real_class(rows, examples, gold):
    """The study above uses its own metric code; the submitted factcheck.py uses overlap_metrics.py. They must agree."""
    grid = {(r["metric"], r["normalisation"], r["remove_stopwords"], r["pool"]): r["dev_acc"] for r in rows}
    ok = total = 0
    for metric, pool, stem, stop in itertools.product(METRICS, POOLS, [False, True], STOPS):
        fc = WordRecallThresholdFactChecker(metric=metric, pool=pool, stem=stem, remove_stopwords=stop)
        scores = np.array([fc.score(ex.fact, ex.passages) for ex in examples])
        acc, _ = best_cut(scores, gold)
        expected = grid[(metric, "stem" if stem else "none", stop, pool)]
        total += 1
        if round(acc, 4) == expected:
            ok += 1
        else:
            print("  MISMATCH %s %s stem=%s stop=%s: real class %.4f vs grid %.4f" % (metric, pool, stem, stop, acc, expected))
    print("\n=== Consistency check: real WordRecallThresholdFactChecker vs grid: %d/%d configurations match ===" % (ok, total))


def deployed_config(examples, gold):
    fc = WordRecallThresholdFactChecker()                  # the defaults currently set in factcheck.py
    name = "metric=%s, pool=%s, stem=%s, remove_stopwords=%s" % (fc.metric, fc.pool, fc.stem, fc.remove_stopwords)
    scores = np.array([fc.score(ex.fact, ex.passages) for ex in examples])
    best_acc, _ = best_cut(scores, gold)
    print("\n=== Deployed defaults in factcheck.py: %s, threshold %s ===" % (name, fc.threshold))
    print("Dev accuracy at the deployed threshold: %.4f (best possible for this setting: %.4f)" % (
        accuracy_at(scores, gold, fc.threshold), best_acc))
    print("Accuracy around the threshold:")
    for k in range(-5, 6):
        t = round(fc.threshold + k / 100, 4)
        print("  threshold %s -> %.4f" % (t, accuracy_at(scores, gold, t)))
    ts = [i / 100 for i in range(0, 101)]
    plt.figure()
    plt.plot(ts, [accuracy_at(scores, gold, t) for t in ts])
    plt.axvline(fc.threshold, color="black", linestyle="--", label="deployed threshold")
    plt.axhline(0.75, color="red", linestyle=":", label="CW target (75%)")
    plt.xlabel("threshold")
    plt.ylabel("accuracy on dev set")
    plt.title(name)
    plt.legend()
    plt.savefig("experiments/part1_threshold.png", dpi=150)
    print("Saved experiments/part1_threshold.png")


def main():
    passages = read_passages("data/passages_bm25_ChatGPT_humfacts.jsonl")
    examples = read_fact_examples("data/dev_labeled_ChatGPT.jsonl", passages)
    gold = np.array([ex.label == "S" for ex in examples])               # IR counts as NS, like factchecking_main.py
    print("%d dev examples, %.1f%% supported (always-S baseline)" % (len(gold), 100 * gold.mean()))

    os.makedirs("experiments", exist_ok=True)
    rows = run_grid(examples, gold)
    with open("experiments/part1_all_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print_tables(rows)
    print("\nSaved experiments/part1_all_results.csv (%d rows)" % len(rows))

    check_real_class(rows, examples, gold)
    deployed_config(examples, gold)


if __name__ == "__main__":
    main()
