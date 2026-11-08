# Tries different settings for the word overlap checker (Part 1) and saves the results in experiments/
# Run from the folder that has factchecking_main.py and data/:   python tune_word_overlap.py

import csv
import itertools
import os

import matplotlib.pyplot as plt

import factchecking_main as fm
from factcheck import WordRecallThresholdFactChecker

LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"

METRICS = ["recall", "jaccard", "dice"]
POOLS = ["union", "max"]
STEM = [False, True]
STOPWORDS = [False, True]

# Recall scores are big (close to 1 when a fact is supported), Jaccard and Dice scores are small because they also
# divide by the size of the passage. So each metric gets its own range of thresholds to try, in steps of 0.01.
RANGES = {"recall": (30, 95), "jaccard": (1, 30), "dice": (1, 30)}


def get_thresholds(metric):
    low, high = RANGES[metric]
    return [round(i / 100, 2) for i in range(low, high + 1)]


def get_accuracy(checker, examples):
    correct = 0
    for ex in examples:
        gold = ex.label
        if gold == "IR":  # irrelevant counts as not supported
            gold = "NS"
        if checker.predict(ex.fact, ex.passages) == gold:
            correct += 1
    return correct / len(examples)


def best_threshold(accuracies, thresholds):
    # accuracies has one number for each threshold
    best_acc = max(accuracies)
    tied = [t for t, a in zip(thresholds, accuracies) if a == best_acc]
    # if a few thresholds tie, take the one in the middle so we are not sitting on the edge
    return best_acc, tied[len(tied) // 2]


examples = fm.read_fact_examples(LABELS_PATH, fm.read_passages(PASSAGES_PATH))
os.makedirs("experiments", exist_ok=True)

rows = []
curves = {}
for metric, pool, stem, stop in itertools.product(METRICS, POOLS, STEM, STOPWORDS):
    thresholds = get_thresholds(metric)
    accuracies = []
    for t in thresholds:
        checker = WordRecallThresholdFactChecker(threshold=t, metric=metric, pool=pool, stem=stem,
                                                 remove_stopwords=stop)
        accuracies.append(get_accuracy(checker, examples))
    best_acc, best_t = best_threshold(accuracies, thresholds)
    rows.append({"metric": metric, "pool": pool, "stem": stem, "remove_stopwords": stop,
                 "threshold": best_t, "accuracy": round(best_acc, 4)})
    curves[(metric, pool, stem, stop)] = (thresholds, accuracies)
    print(metric, pool, "stem=%s" % stem, "stop=%s" % stop, "-> accuracy", round(best_acc, 4), "at", best_t, flush=True)

# best setting first
rows.sort(key=lambda r: r["accuracy"], reverse=True)
with open("experiments/part1_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["metric", "pool", "stem", "remove_stopwords", "threshold", "accuracy"])
    writer.writeheader()
    writer.writerows(rows)

winner = rows[0]
key = (winner["metric"], winner["pool"], winner["stem"], winner["remove_stopwords"])
thresholds, accuracies = curves[key]

# is the winner stable? look at +-0.05 around its threshold
print("\nAccuracy around the chosen threshold", winner["threshold"])
for t, a in zip(thresholds, accuracies):
    if abs(t - winner["threshold"]) <= 0.05 + 1e-9:
        print("  threshold", t, "->", round(a, 4))
if winner["threshold"] in (thresholds[0], thresholds[-1]):
    print("WARNING: the best threshold is at the edge of the range, widen RANGES for", winner["metric"])

plt.plot(thresholds, accuracies)
plt.axvline(winner["threshold"], color="black", linestyle="--")
plt.axhline(0.75, color="red", linestyle=":")
plt.xlabel("threshold")
plt.ylabel("accuracy on dev set")
plt.title("Part 1: %s, %s pooling, stem=%s, remove stopwords=%s" % key)
plt.savefig("experiments/part1_threshold.png", dpi=150)

print("\nWINNER:", winner)