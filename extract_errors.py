# extract_errors.py -- Part 3 (error analysis), step 1: find the entailment model's mistakes   [Kovindu]
#
# What it does
#   1. Runs the finished entailment fact checker over the whole dev set (same model, same settings as
#      `python factchecking_main.py --mode entailment`).
#   2. Marks every example as TP / TN / FP / FN  (IR counts as NS, exactly like factchecking_main.py).
#   3. Randomly picks 10 false positives and 10 false negatives (fixed seed -> same picks every run).
#   4. Writes CSV review sheets with the fact, labels, model score, best evidence and the full passages,
#      plus empty "category" and "notes" columns for the teammates who label the errors.
#
# Run from the repo root (next to factchecking_main.py and the data/ folder):
#   python extract_errors.py            # add --cuda if you have a GPU
# Output folder: error_analysis/
#   all_predictions.csv     every example with its outcome (for counting and sanity checks)
#   false_positives.csv     the 10 sampled false positives  -> Member 2 labels these
#   false_negatives.csv     the 10 sampled false negatives  -> Member 3 labels these
import argparse
import csv
import os
import random
from collections import Counter

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from factchecking_main import read_passages, read_fact_examples
from factcheck import EntailmentModel, EntailmentFactChecker

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"   # same model as factchecking_main.py


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_path", default="data/dev_labeled_ChatGPT.jsonl")
    parser.add_argument("--passages_path", default="data/passages_bm25_ChatGPT_humfacts.jsonl")
    parser.add_argument("--out", default="error_analysis", help="output folder")
    parser.add_argument("--n", type=int, default=10, help="errors to sample per type (FP and FN)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for the sampling")
    parser.add_argument("--cuda", action="store_true", help="run the model on the GPU")
    return parser.parse_args(argv)


def load_checker(cuda: bool) -> EntailmentFactChecker:
    # Same construction as the entailment branch of factchecking_main.py
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    if cuda:
        model.to("cuda")
    return EntailmentFactChecker(EntailmentModel(model, tokenizer, cuda))


def outcome_of(gold: str, pred: str) -> str:
    if gold == "S":
        return "TP" if pred == "S" else "FN"     # supported fact: found it (TP) or missed it (FN)
    return "FP" if pred == "S" else "TN"         # unsupported fact: wrongly accepted (FP) or rejected (TN)


def format_evidence(evidence, top_k=3) -> str:
    """The top_k kept premises with the highest entailment probability, one per line."""
    best = sorted(evidence, key=lambda e: e[2]["entailment"], reverse=True)[:top_k]
    lines = []
    for premise, overlap, probs in best:
        lines.append("overlap=%.2f  E=%.2f N=%.2f C=%.2f | %s" % (
            overlap, probs["entailment"], probs["neutral"], probs["contradiction"], premise))
    return "\n".join(lines) if lines else "(no premise survived pruning)"


def write_csv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig so Excel shows accents correctly
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    fact_to_passage = read_passages(args.passages_path)
    examples = read_fact_examples(args.labels_path, fact_to_passage)
    checker = load_checker(args.cuda)

    rows = []
    for i, ex in enumerate(tqdm(examples)):
        result = checker.score_fact(ex.fact, ex.passages)          # score + evidence in ONE model pass
        # same decision rule as EntailmentFactChecker.predict()
        pred = "S" if result["evidence"] and result["score"] >= checker.threshold else "NS"
        gold = "NS" if ex.label == "IR" else ex.label
        rows.append({
            "id": i,
            "fact": ex.fact,
            "gold_raw": ex.label,                                   # S / NS / IR (IR is scored as NS)
            "gold": gold,
            "predicted": pred,
            "outcome": outcome_of(gold, pred),
            "entail_score": round(result["score"], 4),
            "premises_total": result["n_premises"],
            "premises_kept": len(result["evidence"]),
            "top_evidence": format_evidence(result["evidence"]),
            "passages": "\n\n".join("[%s] %s" % (p.get("title", ""), p["text"]) for p in ex.passages),
            "category": "",                                         # <- reviewers fill these two in
            "notes": "",
        })

    # sanity check: this confusion matrix and accuracy must match `--mode entailment`
    counts = Counter(r["outcome"] for r in rows)
    acc = (counts["TP"] + counts["TN"]) / len(rows)
    print("\nTP=%d  FN=%d  FP=%d  TN=%d   accuracy=%d/%d = %.4f" % (
        counts["TP"], counts["FN"], counts["FP"], counts["TN"], counts["TP"] + counts["TN"], len(rows), acc))

    rng = random.Random(args.seed)
    columns = ["id", "fact", "gold_raw", "predicted", "entail_score", "premises_total", "premises_kept",
               "top_evidence", "passages", "category", "notes"]
    for outcome, filename in (("FP", "false_positives.csv"), ("FN", "false_negatives.csv")):
        pool = [r for r in rows if r["outcome"] == outcome]
        picked = sorted(rng.sample(pool, min(args.n, len(pool))), key=lambda r: r["id"])
        write_csv(os.path.join(args.out, filename), picked, columns)
        extra = ""
        if outcome == "FP":
            extra = "  (%d of the %d have raw label IR)" % (sum(r["gold_raw"] == "IR" for r in pool), len(pool))
        print("%s: %d errors in total, wrote %d to %s%s" % (outcome, len(pool), len(picked), filename, extra))

    write_csv(os.path.join(args.out, "all_predictions.csv"), rows,
              ["id", "fact", "gold_raw", "gold", "predicted", "outcome", "entail_score",
               "premises_total", "premises_kept", "top_evidence"])
    print("Saved everything in %s/" % args.out)


if __name__ == "__main__":
    main()
