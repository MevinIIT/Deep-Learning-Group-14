# error_tally.py - helper for error analysis part

import csv
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ALL_ERRORS = os.path.join(RES, "errors_for_part3.csv")
TO_LABEL = os.path.join(RES, "errors_to_label.csv")
N_PER_TYPE = 10   # 10 FP + 10 FN
SEED = 42


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def pick():
    fields, rows = read_rows(ALL_ERRORS)
    random.seed(SEED)
    chosen = []
    for etype in ("FP", "FN"):
        group = [r for r in rows if r["error_type"] == etype]
        if len(group) > N_PER_TYPE:
            group = random.sample(group, N_PER_TYPE)
        print(etype + ": " + str(len(group)) + " picked")
        chosen += group
    with open(TO_LABEL, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(chosen)
    print("written " + TO_LABEL)
    print("now fill the 'category' and 'notes' columns by hand")


def tally():
    if not os.path.exists(TO_LABEL):
        sys.exit("run 'pick' first")
    _, rows = read_rows(TO_LABEL)
    counts = defaultdict(Counter)
    unlabelled = 0
    for r in rows:
        cat = r["category"].strip()
        if not cat:
            unlabelled += 1
            continue
        counts[r["error_type"]][cat] += 1
    if unlabelled:
        print("warning: " + str(unlabelled) + " rows have no category yet")

    lines = ["| error type | category | count | % of that type |", "|---|---|---|---|"]
    for etype in ("FP", "FN"):
        total = sum(counts[etype].values())
        for cat, n in counts[etype].most_common():
            lines.append("| " + etype + " | " + cat + " | " + str(n) + " | " + format(100 * n / total, ".0f") + "% |")
    table = "\n".join(lines)
    print(table)
    out_md = os.path.join(RES, "error_tally.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("written " + out_md)

    # bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping the chart")
        return
    labels, values = [], []
    for etype in ("FP", "FN"):
        for cat, n in counts[etype].most_common():
            labels.append(etype + ": " + cat)
            values.append(n)
    if not values:
        return
    plt.figure(figsize=(8, 0.5 * len(values) + 1.5))
    plt.barh(labels[::-1], values[::-1])
    plt.xlabel("number of errors")
    plt.title("Error categories (labelled sample)")
    plt.tight_layout()
    out_png = os.path.join(RES, "error_tally.png")
    plt.savefig(out_png, dpi=150)
    print("written " + out_png)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "pick":
        pick()
    elif cmd == "tally":
        tally()
    else:
        print("usage: python experiments/error_tally.py [pick|tally]")
