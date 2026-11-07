# Part 2 experiments: run the NLI model once, cache the probabilities, then compare settings offline.
# Run from the repo root:
#   python experiments/part2_experiments.py cache     (slow, runs the model, add --cuda for GPU)
#   python experiments/part2_experiments.py analyse   (tables + plots -> experiments/results/)
#   python experiments/part2_experiments.py errors    (FP/FN list for Part 3)
#   python experiments/part2_experiments.py timing    (real runtime with/without pruning)
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import factcheck as fc  # noqa: E402
from factchecking_main import read_passages, read_fact_examples  # noqa: E402

LABELS = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES = "data/passages_bm25_ChatGPT_humfacts.jsonl"
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
CACHE_DIR = os.path.join(ROOT, "experiments", "cache")
RESULTS_DIR = os.path.join(ROOT, "experiments", "results")

# name -> (context, clean?)
VARIANTS = {
    "raw_sentence":   ("sentence", False),  # no cleaning
    "clean_sentence": ("sentence", True),   # cleaned
    "clean_title":    ("title", True),      # "Name: sentence"
    "clean_window":   ("window", True),     # prev + current sentence
    "clean_passage":  ("passage", True),    # whole passage
}

# should match the defaults in factcheck.py
FINAL = dict(variant="clean_window", strategy="max_entail", min_overlap=0.1, max_candidates=10, threshold=None)

THRESHOLDS = np.round(np.arange(0.0, 1.0001, 0.01), 2)


# ---- data ----
def load_examples():
    examples = read_fact_examples(LABELS, read_passages(PASSAGES))
    for ex in examples:
        ex.gold = "NS" if ex.label == "IR" else ex.label
        ex.entity = ex.passages[0]["title"] if ex.passages else ""
    return examples


def load_model(cuda):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    if cuda:
        model.to("cuda")
    return fc.EntailmentModel(model, tok, cuda)


# ---- 1. cache probabilities (no pruning here, pruning is done offline later) ----
def cmd_cache(args):
    os.makedirs(CACHE_DIR, exist_ok=True)
    examples = load_examples()
    ent = load_model(args.cuda)
    for name in (args.variants or VARIANTS):
        context, clean = VARIANTS[name]
        path = os.path.join(CACHE_DIR, name + ".json")
        if os.path.exists(path):
            print(f"[cache] {name}: exists, skipping")
            continue
        rows, n_pairs, t0 = [], 0, time.time()
        for i, ex in enumerate(examples):
            premises = fc.build_premises(ex.passages, context=context, clean=clean)
            q = fc.fact_query_tokens(ex.fact, ex.entity)
            probs = ent.check_entailment_batch(premises, ex.fact) if premises else []
            rows.append([[p, fc.overlap_recall(q, p), pr["entailment"], pr["neutral"], pr["contradiction"]]
                         for p, pr in zip(premises, probs)])
            n_pairs += len(premises)
            if (i + 1) % 20 == 0:
                print(f"[cache] {name}: {i + 1}/{len(examples)} facts, {n_pairs} pairs, {time.time() - t0:.0f}s")
        secs = time.time() - t0
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"variant": name, "seconds": secs, "pairs": n_pairs, "rows": rows}, f)
        print(f"[cache] {name}: done, {n_pairs} pairs in {secs:.0f}s ({n_pairs / max(secs, 1e-9):.1f} pairs/s)")


def load_cache(name):
    with open(os.path.join(CACHE_DIR, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


# ---- offline evaluation ----
def fact_scores(rows, strategy, min_overlap, max_candidates):
    # score per fact (nan = everything pruned -> NS) and number of model calls
    scores, calls = [], 0
    for premises in rows:
        kept = [r for r in premises if r[1] >= min_overlap]
        kept.sort(key=lambda r: r[1], reverse=True)
        if max_candidates is not None:
            kept = kept[:max_candidates]
        calls += len(kept)
        if not kept:
            scores.append(np.nan)
            continue
        probs = [{"entailment": r[2], "neutral": r[3], "contradiction": r[4]} for r in kept]
        scores.append(fc.aggregate_score(probs, strategy))
    return np.array(scores, dtype=float), calls


def metrics(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred)
    out = {"acc": float(np.mean(gold == pred)), "n_correct": int(np.sum(gold == pred))}
    f1s = []
    for c in ("S", "NS"):
        tp = np.sum((pred == c) & (gold == c))
        p = tp / max(np.sum(pred == c), 1)
        r = tp / max(np.sum(gold == c), 1)
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out[f"P_{c}"], out[f"R_{c}"], out[f"F1_{c}"] = float(p), float(r), float(f1)
        f1s.append(f1)
    out["macro_F1"] = float(np.mean(f1s))
    out["FP"] = int(np.sum((pred == "S") & (gold == "NS")))
    out["FN"] = int(np.sum((pred == "NS") & (gold == "S")))
    return out


def predict(scores, threshold):
    return np.where(~np.isnan(scores) & (np.nan_to_num(scores, nan=-1) >= threshold), "S", "NS")


def sweep(scores, gold, strategy):
    if strategy == "discrete":
        return [(0.5, metrics(gold, predict(scores, 0.5)))]
    return [(t, metrics(gold, predict(scores, t))) for t in THRESHOLDS]


SMOOTH = 3  # smoothing window (+/- 0.03)


def best_of(sw):
    # pick threshold by smoothed accuracy so a single spike (e.g. 0.01) is not chosen
    if len(sw) == 1:
        return sw[0][0], sw[0][1], (sw[0][0], sw[0][0])
    accs = np.array([m["acc"] for _, m in sw])
    smooth = np.array([accs[max(0, i - SMOOTH):i + SMOOTH + 1].mean() for i in range(len(accs))])
    best_idx = np.flatnonzero(np.isclose(smooth, smooth.max()))
    i = int(best_idx[len(best_idx) // 2])
    lo = hi = i
    while lo > 0 and accs[lo - 1] == accs[i]:
        lo -= 1
    while hi < len(accs) - 1 and accs[hi + 1] == accs[i]:
        hi += 1
    return sw[i][0], sw[i][1], (sw[lo][0], sw[hi][0])


def raw_best(sw):
    # plain best threshold, just for comparison
    t, m = max(sw, key=lambda x: x[1]["acc"])
    return t, m


def grouped_cv(scores, gold, entities, strategy, folds=5, seed=0):
    # 5-fold CV grouped by entity: tune threshold on 4 folds, test on the 5th
    ents = sorted(set(entities))
    random.Random(seed).shuffle(ents)
    fold_of = {e: i % folds for i, e in enumerate(ents)}
    fid = np.array([fold_of[e] for e in entities])
    correct, chosen = 0, []
    for k in range(folds):
        tr, te = fid != k, fid == k
        if strategy == "discrete":
            t = 0.5
        else:
            t, _, _ = best_of(sweep(scores[tr], gold[tr], strategy))
        chosen.append(t)
        correct += int(np.sum(predict(scores[te], t) == gold[te]))
    return correct / len(gold), chosen


# ---- 2. analysis ----
def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def cmd_analyse(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    examples = load_examples()
    gold = np.array([ex.gold for ex in examples])
    entities = [ex.entity for ex in examples]
    available = [v for v in VARIANTS if os.path.exists(os.path.join(CACHE_DIR, v + ".json"))]
    caches = {v: load_cache(v) for v in available}
    eps0, k0 = FINAL["min_overlap"], FINAL["max_candidates"]
    report = [f"# Part 2 experiment results\n\nDev set: {len(gold)} facts "
              f"({int(np.sum(gold == 'S'))} S / {int(np.sum(gold == 'NS'))} NS incl. IR). "
              f"Majority-class baseline = {np.mean(gold == 'NS') if np.mean(gold == 'NS') > 0.5 else np.mean(gold == 'S'):.3f}.\n"]

    # exp 1: decision strategies
    v = FINAL["variant"]
    rows, sweeps = [], {}
    for strat in fc.STRATEGIES:
        sc, calls = fact_scores(caches[v]["rows"], strat, eps0, k0)
        sw = sweep(sc, gold, strat)
        sweeps[strat] = sw
        t, m, plateau = best_of(sw)
        cv, _ = grouped_cv(sc, gold, entities, strat)
        m05 = metrics(gold, predict(sc, 0.5))
        rt, rm = raw_best(sw)
        rows.append({"strategy": strat, "acc@0.5": f"{m05['acc']:.3f}",
                     "raw_argmax": f"{rm['acc']:.3f}@{rt}", "best_threshold": t,
                     "best_plateau": f"{plateau[0]}-{plateau[1]}", "best_acc": f"{m['acc']:.3f}",
                     "macro_F1": f"{m['macro_F1']:.3f}", "FP": m["FP"], "FN": m["FN"], "cv_acc": f"{cv:.3f}"})
    write_csv(os.path.join(RESULTS_DIR, "exp1_strategies.csv"), rows)
    report.append(f"## 1. Decision strategy (variant={v}, pruning min_overlap={eps0}, top-k={k0})\n")
    report.append(md_table(rows, list(rows[0].keys())))
    report.append("\n`raw_argmax` = highest accuracy at any single threshold (can be a one-point spike). "
                  f"`best_threshold` = chosen by accuracy smoothed over +/-{SMOOTH} steps; `best_acc` is the raw "
                  "accuracy there. `cv_acc` = 5-fold entity-grouped cross-validation: the threshold is chosen the "
                  "same way on 4 folds and applied to the held-out fold, estimating accuracy on unseen entities.\n")

    sweep_rows = []
    for strat, sw in sweeps.items():
        for t, m in sw:
            sweep_rows.append({"strategy": strat, "threshold": t, "acc": round(m["acc"], 4),
                               "F1_S": round(m["F1_S"], 4), "F1_NS": round(m["F1_NS"], 4)})
    write_csv(os.path.join(RESULTS_DIR, "exp2_threshold_sweep.csv"), sweep_rows)

    # exp 3: pruning grid
    sec_per_pair = caches[v]["seconds"] / max(caches[v]["pairs"], 1)
    rows = []
    for eps in (0.0, 0.01, 0.1, 0.2, 0.34, 0.5):
        for k in (1, 3, 5, 10, 20, None):
            sc, calls = fact_scores(caches[v]["rows"], FINAL["strategy"], eps, k)
            t, m, _ = best_of(sweep(sc, gold, FINAL["strategy"]))
            m05 = metrics(gold, predict(sc, 0.5))
            pruned = np.isnan(sc)
            rows.append({"min_overlap": eps, "top_k": k if k else "all", "model_calls": calls,
                         "calls_%": f"{100 * calls / caches[v]['pairs']:.1f}",
                         "est_runtime_s": f"{calls * sec_per_pair:.0f}",
                         "facts_pruned": int(pruned.sum()),
                         "pruned_NS_precision": f"{np.mean(gold[pruned] == 'NS'):.3f}" if pruned.any() else "-",
                         "acc@0.5": f"{m05['acc']:.3f}", "best_threshold": t, "best_acc": f"{m['acc']:.3f}"})
    write_csv(os.path.join(RESULTS_DIR, "exp3_pruning.csv"), rows)
    report.append(f"## 3. Pruning (variant={v}, strategy={FINAL['strategy']})\n")
    report.append(f"Measured cost without pruning: {caches[v]['pairs']} model calls, "
                  f"{caches[v]['seconds']:.0f}s ({sec_per_pair * 1000:.0f} ms/pair on this machine). "
                  "`est_runtime_s` = calls x measured time per pair; `timing` gives real end-to-end numbers.\n")
    report.append(md_table(rows, list(rows[0].keys())))

    # exp 4: preprocessing variants
    rows = []
    for name in available:
        for eps, k in ((0.0, None), (eps0, k0)):
            sc, calls = fact_scores(caches[name]["rows"], FINAL["strategy"], eps, k)
            t, m, _ = best_of(sweep(sc, gold, FINAL["strategy"]))
            cv, _ = grouped_cv(sc, gold, entities, FINAL["strategy"])
            m05 = metrics(gold, predict(sc, 0.5))
            rows.append({"variant": name, "pruning": "none" if k is None else f"eps={eps},k={k}",
                         "premises": caches[name]["pairs"], "model_calls": calls, "acc@0.5": f"{m05['acc']:.3f}",
                         "best_threshold": t, "best_acc": f"{m['acc']:.3f}", "cv_acc": f"{cv:.3f}",
                         "FP": m["FP"], "FN": m["FN"]})
    write_csv(os.path.join(RESULTS_DIR, "exp4_preprocessing.csv"), rows)
    report.append(f"\n## 4. Preprocessing / premise construction (strategy={FINAL['strategy']})\n")
    report.append(md_table(rows, list(rows[0].keys())))

    timing_csv = os.path.join(RESULTS_DIR, "exp5_timing.csv")
    if os.path.exists(timing_csv):
        with open(timing_csv, encoding="utf-8") as f:
            trows = list(csv.DictReader(f))
        report.append("\n## 5. Real end-to-end runtime of EntailmentFactChecker (from `timing`, final defaults)\n")
        report.append(md_table(trows, list(trows[0].keys())))

    # final config
    sc, calls = fact_scores(caches[v]["rows"], FINAL["strategy"], eps0, k0)
    t, m, plateau = best_of(sweep(sc, gold, FINAL["strategy"]))
    report.append(f"\n## Final configuration\n\nvariant={v}, strategy={FINAL['strategy']}, min_overlap={eps0}, "
                  f"top_k={k0}, threshold={t} (accuracy is maximal for thresholds {plateau[0]}-{plateau[1]})\n\n"
                  f"Accuracy {m['n_correct']}/{len(gold)} = {m['acc']:.4f}, macro-F1 {m['macro_F1']:.3f}, "
                  f"FP={m['FP']}, FN={m['FN']}, model calls={calls}\n")

    with open(os.path.join(RESULTS_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("\n".join(report))
    make_plots(sweeps, sc, gold)


def make_plots(sweeps, final_scores, gold):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyse] matplotlib not installed; skipping plots (pip install matplotlib)")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for strat, sw in sweeps.items():
        if strat == "discrete":
            ax.axhline(sw[0][1]["acc"], ls=":", color="grey", label="discrete (argmax)")
            continue
        ax.plot([t for t, _ in sw], [m["acc"] for _, m in sw], label=strat)
    ax.axhline(0.83, ls="--", color="red", lw=0.8, label="83% target")
    ax.set_xlabel("threshold")
    ax.set_ylabel("dev accuracy")
    ax.set_ylim(0.4, 0.95)
    ax.legend(fontsize=8)
    ax.set_title("Accuracy vs decision threshold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "threshold_sweep.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    s = np.nan_to_num(final_scores, nan=0.0)
    bins = np.linspace(0, 1, 21)
    ax.hist(s[gold == "S"], bins=bins, alpha=0.6, label="gold S")
    ax.hist(s[gold == "NS"], bins=bins, alpha=0.6, label="gold NS (incl. IR)")
    ax.set_xlabel("fact score (max entailment probability; pruned facts = 0)")
    ax.set_ylabel("facts")
    ax.legend()
    ax.set_title("Score distribution by gold label")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "score_hist.png"), dpi=150)
    print(f"[analyse] plots written to {RESULTS_DIR}")


# ---- 3. errors for Part 3 ----
def cmd_errors(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    examples = load_examples()
    gold = np.array([ex.gold for ex in examples])
    cache = load_cache(args.variant)
    sc, _ = fact_scores(cache["rows"], args.strategy, args.min_overlap, args.max_candidates)
    thr = args.threshold if args.threshold is not None else best_of(sweep(sc, gold, args.strategy))[0]
    pred = predict(sc, thr)
    out = []
    for i, ex in enumerate(examples):
        if pred[i] == gold[i]:
            continue
        kept = sorted([r for r in cache["rows"][i] if r[1] >= args.min_overlap], key=lambda r: r[1], reverse=True)
        kept = kept[:args.max_candidates] if args.max_candidates else kept
        top = sorted(kept, key=lambda r: r[2], reverse=True)[:3]
        row = {"error_type": "FP" if pred[i] == "S" else "FN", "entity": ex.entity, "fact": ex.fact,
               "gold_raw": ex.label, "gold": gold[i], "pred": pred[i],
               "score": "pruned" if np.isnan(sc[i]) else f"{sc[i]:.3f}", "category": "", "notes": ""}
        for j in range(3):
            r = top[j] if j < len(top) else None
            row[f"premise{j + 1}"] = r[0] if r else ""
            row[f"premise{j + 1}_E/N/C"] = f"{r[2]:.2f}/{r[3]:.2f}/{r[4]:.2f}" if r else ""
        out.append(row)
    out.sort(key=lambda r: r["error_type"])
    path = os.path.join(RESULTS_DIR, "errors_for_part3.csv")
    write_csv(path, out)
    print(f"threshold={thr}: {sum(r['error_type'] == 'FP' for r in out)} FPs, "
          f"{sum(r['error_type'] == 'FN' for r in out)} FNs -> {path}")


# ---- 4. real timing ----
def cmd_timing(args):
    examples = load_examples()
    ent = load_model(args.cuda)
    rows = []
    for eps, k in ((0.0, None), (0.1, 10), (0.1, 5), (0.2, 3)):
        checker = fc.EntailmentFactChecker(ent, min_overlap=eps, max_candidates=k)
        t0 = time.time()
        preds = [checker.predict(ex.fact, ex.passages) for ex in examples]
        secs = time.time() - t0
        acc = np.mean(np.array(preds) == np.array([ex.gold for ex in examples]))
        rows.append({"min_overlap": eps, "top_k": k if k else "all", "seconds": f"{secs:.1f}",
                     "acc@default_threshold": f"{acc:.3f}"})
        print(rows[-1])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_csv(os.path.join(RESULTS_DIR, "exp5_timing.csv"), rows)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--cuda", action="store_true")
    c.add_argument("--variants", nargs="*", choices=list(VARIANTS))
    sub.add_parser("analyse")
    e = sub.add_parser("errors")
    e.add_argument("--variant", default=FINAL["variant"])
    e.add_argument("--strategy", default=FINAL["strategy"])
    e.add_argument("--min-overlap", type=float, default=FINAL["min_overlap"])
    e.add_argument("--max-candidates", type=int, default=FINAL["max_candidates"])
    e.add_argument("--threshold", type=float, default=FINAL["threshold"])
    t = sub.add_parser("timing")
    t.add_argument("--cuda", action="store_true")
    args = ap.parse_args()
    {"cache": cmd_cache, "analyse": cmd_analyse, "errors": cmd_errors, "timing": cmd_timing}[args.cmd](args)


if __name__ == "__main__":
    main()
