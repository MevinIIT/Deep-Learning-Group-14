import csv
import itertools
import os

import matplotlib.pyplot as plt

from factchecking_main import read_passages, read_fact_examples
from factcheck import WordRecallThresholdFactChecker
from overlap_metrics import METRICS

passages = read_passages("data/passages_bm25_ChatGPT_humfacts.jsonl")
examples = read_fact_examples("data/dev_labeled_ChatGPT.jsonl", passages)
gold = [ex.label == "S" for ex in examples]          # IR counts as NS, same as factchecking_main.py


def accuracy_at(scores, threshold):
    # share of facts where "score >= threshold" matches the gold label
    correct = 0
    for score, g in zip(scores, gold):
        if (score >= threshold) == g:
            correct += 1
    return correct / len(gold)


def best_threshold(scores):
    """Try every distinct score as a cut-off (predict S when score >= cut-off) and return
    (best_accuracy, threshold). The threshold is the midpoint between the best cut-off and the next lower
    distinct score, so it sits in the middle of a plateau instead of exactly on a data point."""
    distinct = sorted(set(scores))
    best_acc = -1
    best_i = 0
    for i, cutoff in enumerate(distinct):
        acc = accuracy_at(scores, cutoff)
        if acc > best_acc:  # only a strictly better accuracy replaces it, so ties keep the lowest cut-off
            best_acc = acc
            best_i = i
    if best_i == 0:
        threshold = distinct[0]  # there is no lower score to take a midpoint with
    else:
        threshold = (distinct[best_i] + distinct[best_i - 1]) / 2
    # round the threshold so it is easy to type, but only if the accuracy stays the same
    for digits in (2, 3, 4):
        rounded = round(threshold, digits)
        if accuracy_at(scores, rounded) == best_acc:
            return best_acc, rounded
    return best_acc, threshold


def run_experiments():
    """Loop over metric x pool x stem x remove_stopwords, find the best threshold for each setting and print
    the rows sorted by accuracy (best first). Also saves the table, checks how stable the winner is around its
    threshold and saves a plot, all in experiments/."""
    rows = []
    for metric, pool, stem, stop in itertools.product(METRICS, ["max", "union"], [False, True], [False, True]):
        fc = WordRecallThresholdFactChecker(metric=metric, pool=pool, stem=stem, remove_stopwords=stop)
        scores = [fc.score(ex.fact, ex.passages) for ex in examples]
        acc, threshold = best_threshold(scores)
        rows.append((acc, threshold, metric, pool, stem, stop, scores))

    rows.sort(key=lambda r: r[0], reverse=True)

    print("accuracy  threshold  metric   pool   stem   stop-words")
    for acc, threshold, metric, pool, stem, stop, _ in rows:
        print("%.4f    %-9s  %-8s %-6s %-6s %s" % (acc, threshold, metric, pool, stem, stop))

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/part1_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accuracy", "threshold", "metric", "pool", "stem", "remove_stopwords"])
        for acc, threshold, metric, pool, stem, stop, _ in rows:
            writer.writerow([round(acc, 4), threshold, metric, pool, stem, stop])

    # is the best row stable? check +-0.05 around its threshold in steps of 0.01
    acc, threshold, metric, pool, stem, stop, scores = rows[0]
    print("\nBest row: %s, %s, stem=%s, remove_stopwords=%s, threshold %s" % (metric, pool, stem, stop, threshold))
    print("Accuracy around the threshold:")
    for k in range(-5, 6):
        t = round(threshold + k / 100, 4)
        print("  threshold %s -> %.4f" % (t, accuracy_at(scores, t)))

    # accuracy against threshold for the best row
    ts = [i / 100 for i in range(0, 101)]
    plt.plot(ts, [accuracy_at(scores, t) for t in ts])
    plt.axvline(threshold, color="black", linestyle="--")
    plt.axhline(0.75, color="red", linestyle=":")
    plt.xlabel("threshold")
    plt.ylabel("accuracy on dev set")
    plt.title("%s, %s, stem=%s, remove_stopwords=%s" % (metric, pool, stem, stop))
    plt.savefig("experiments/part1_threshold.png", dpi=150)
    print("\nSaved experiments/part1_results.csv and experiments/part1_threshold.png")
    return rows


if __name__ == "__main__":
    run_experiments()