# Part 2 handoff – Isum

Start a new Claude chat, attach this file + `part2_package.zip` + the CW brief, and paste the prompt Desindu sent you.

## Where we are

- Part 2 (Desindu + Isum), Part 1 word overlap (Kovindu + Mevin), Part 3 error analysis (everyone, after Parts 1 and 2 are final).
- Repo: github.com/MevinIIT/Deep-Learning-Group-14
  - `main` – framework
  - `feature/part2-entailment` – final `factcheck.py` (Desindu, commit `25c94f9`)
  - `feature/part2-experiments` – experiments folder (Isum, to create)
- Don't touch `factchecking_main.py`. Submission = `factcheck.py` + one PDF report.
- GenAI categories 3 and 4 apply, so we need the GenAI acknowledgement in the report.

## Results (dev set, 221 facts)

| | Accuracy |
|---|---|
| Baseline v0 (`7ff4eb3`) | 183/221 = 82.8% |
| Final (`25c94f9`) | **191/221 = 86.4%** |
| Final, 5-fold CV by entity | 84.6% |
| Final without pruning | 84.2% (4.5x slower) |

Full tables: `experiments/results/summary.md`. Plot: `threshold_sweep.png`.

## How the checker works

1. Clean passages: remove `<s>` tags, section headings, repeated names; split into sentences (NLTK, regex fallback).
2. Premise = previous sentence + current sentence (window).
3. Pruning: word overlap without stopwords and the person's name; keep overlap >= 0.1, top 10. No match -> NS.
4. DeBERTa NLI in batches of 16, memory freed after each fact.
5. Score = max entailment probability; S if >= 0.31.

## Your tasks

1. Branch and commit:
   ```
   git fetch origin
   git checkout feature/part2-entailment
   git pull
   git checkout -b feature/part2-experiments
   ```
   Unzip `experiments/` into the repo (don't overwrite `factcheck.py`, the committed one is the same), then:
   ```
   git add experiments
   git commit -m "feat(part2): experiment harness, cached probabilities and results"
   git push -u origin feature/part2-experiments
   ```
2. Reproduce (`pip install matplotlib` first):
   ```
   python experiments/part2_experiments.py analyse
   python experiments/part2_experiments.py errors
   python experiments/part2_experiments.py timing   (optional, ~15 min)
   ```
   Numbers should match `summary.md`. The cache is included, so there's no need to run `cache` unless you delete it.
3. Write the Part 2 report section: strategy comparison, threshold choice (smoothed + CV), pruning speed vs accuracy, preprocessing comparison, memory/efficiency, limitations.
4. `results/errors_for_part3.csv` (7 FP, 23 FN) is the starting point for Part 3.

## Notes

- Only Desindu edits `factcheck.py`. If the defaults ever change, update `FINAL` in the script and re-run `analyse`.
- If Part 1 changes the shared overlap functions, re-run `cache` and `analyse`.
