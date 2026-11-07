# Part 2 experiment results

Dev set: 221 facts (112 S / 109 NS incl. IR). Majority-class baseline = 0.507.

## 1. Decision strategy (variant=clean_window, pruning min_overlap=0.1, top-k=10)

| strategy | acc@0.5 | raw_argmax | best_threshold | best_plateau | best_acc | macro_F1 | FP | FN | cv_acc |
|---|---|---|---|---|---|---|---|---|---|
| max_entail | 0.855 | 0.864@0.19 | 0.31 | 0.29-0.33 | 0.864 | 0.864 | 7 | 23 | 0.846 |
| entail_minus_contra | 0.860 | 0.864@0.18 | 0.28 | 0.24-0.32 | 0.864 | 0.864 | 7 | 23 | 0.842 |
| entail_vs_contra | 0.814 | 0.855@0.89 | 0.91 | 0.89-0.93 | 0.855 | 0.855 | 12 | 20 | 0.810 |
| discrete | 0.855 | 0.855@0.5 | 0.5 | 0.5-0.5 | 0.855 | 0.855 | 7 | 25 | 0.855 |

`raw_argmax` = highest accuracy at any single threshold (can be a one-point spike). `best_threshold` = chosen by accuracy smoothed over +/-3 steps; `best_acc` is the raw accuracy there. `cv_acc` = 5-fold entity-grouped cross-validation: the threshold is chosen the same way on 4 folds and applied to the held-out fold, estimating accuracy on unseen entities.

## 3. Pruning (variant=clean_window, strategy=max_entail)

Measured cost without pruning: 5215 model calls, 565s (108 ms/pair on this machine). `est_runtime_s` = calls x measured time per pair; `timing` gives real end-to-end numbers.

| min_overlap | top_k | model_calls | calls_% | est_runtime_s | facts_pruned | pruned_NS_precision | acc@0.5 | best_threshold | best_acc |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1 | 221 | 4.2 | 24 | 0 | - | 0.792 | 0.28 | 0.801 |
| 0.0 | 3 | 663 | 12.7 | 72 | 0 | - | 0.819 | 0.73 | 0.828 |
| 0.0 | 5 | 1105 | 21.2 | 120 | 0 | - | 0.837 | 0.25 | 0.846 |
| 0.0 | 10 | 2200 | 42.2 | 238 | 0 | - | 0.837 | 0.7 | 0.846 |
| 0.0 | 20 | 3842 | 73.7 | 416 | 0 | - | 0.828 | 0.7 | 0.837 |
| 0.0 | all | 5215 | 100.0 | 565 | 0 | - | 0.824 | 0.23 | 0.842 |
| 0.01 | 1 | 176 | 3.4 | 19 | 45 | 0.889 | 0.796 | 0.28 | 0.810 |
| 0.01 | 3 | 476 | 9.1 | 52 | 45 | 0.889 | 0.833 | 0.38 | 0.842 |
| 0.01 | 5 | 693 | 13.3 | 75 | 45 | 0.889 | 0.851 | 0.31 | 0.860 |
| 0.01 | 10 | 1034 | 19.8 | 112 | 45 | 0.889 | 0.855 | 0.31 | 0.864 |
| 0.01 | 20 | 1170 | 22.4 | 127 | 45 | 0.889 | 0.855 | 0.31 | 0.864 |
| 0.01 | all | 1181 | 22.6 | 128 | 45 | 0.889 | 0.855 | 0.31 | 0.864 |
| 0.1 | 1 | 176 | 3.4 | 19 | 45 | 0.889 | 0.796 | 0.28 | 0.810 |
| 0.1 | 3 | 476 | 9.1 | 52 | 45 | 0.889 | 0.833 | 0.38 | 0.842 |
| 0.1 | 5 | 693 | 13.3 | 75 | 45 | 0.889 | 0.851 | 0.31 | 0.860 |
| 0.1 | 10 | 1034 | 19.8 | 112 | 45 | 0.889 | 0.855 | 0.31 | 0.864 |
| 0.1 | 20 | 1170 | 22.4 | 127 | 45 | 0.889 | 0.855 | 0.31 | 0.864 |
| 0.1 | all | 1181 | 22.6 | 128 | 45 | 0.889 | 0.855 | 0.31 | 0.864 |
| 0.2 | 1 | 173 | 3.3 | 19 | 48 | 0.896 | 0.796 | 0.28 | 0.810 |
| 0.2 | 3 | 470 | 9.0 | 51 | 48 | 0.896 | 0.833 | 0.38 | 0.842 |
| 0.2 | 5 | 681 | 13.1 | 74 | 48 | 0.896 | 0.851 | 0.31 | 0.860 |
| 0.2 | 10 | 997 | 19.1 | 108 | 48 | 0.896 | 0.855 | 0.31 | 0.864 |
| 0.2 | 20 | 1127 | 21.6 | 122 | 48 | 0.896 | 0.855 | 0.31 | 0.864 |
| 0.2 | all | 1138 | 21.8 | 123 | 48 | 0.896 | 0.855 | 0.31 | 0.864 |
| 0.34 | 1 | 135 | 2.6 | 15 | 86 | 0.884 | 0.801 | 0.31 | 0.814 |
| 0.34 | 3 | 333 | 6.4 | 36 | 86 | 0.884 | 0.837 | 0.38 | 0.842 |
| 0.34 | 5 | 438 | 8.4 | 47 | 86 | 0.884 | 0.846 | 0.31 | 0.851 |
| 0.34 | 10 | 542 | 10.4 | 59 | 86 | 0.884 | 0.851 | 0.31 | 0.855 |
| 0.34 | 20 | 574 | 11.0 | 62 | 86 | 0.884 | 0.851 | 0.31 | 0.855 |
| 0.34 | all | 575 | 11.0 | 62 | 86 | 0.884 | 0.851 | 0.31 | 0.855 |
| 0.5 | 1 | 127 | 2.4 | 14 | 94 | 0.872 | 0.792 | 0.31 | 0.805 |
| 0.5 | 3 | 310 | 5.9 | 34 | 94 | 0.872 | 0.828 | 0.38 | 0.833 |
| 0.5 | 5 | 408 | 7.8 | 44 | 94 | 0.872 | 0.837 | 0.31 | 0.842 |
| 0.5 | 10 | 509 | 9.8 | 55 | 94 | 0.872 | 0.842 | 0.31 | 0.846 |
| 0.5 | 20 | 541 | 10.4 | 59 | 94 | 0.872 | 0.842 | 0.31 | 0.846 |
| 0.5 | all | 542 | 10.4 | 59 | 94 | 0.872 | 0.842 | 0.31 | 0.846 |

## 4. Preprocessing / premise construction (strategy=max_entail)

| variant | pruning | premises | model_calls | acc@0.5 | best_threshold | best_acc | cv_acc | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| raw_sentence | none | 5228 | 5228 | 0.837 | 0.36 | 0.855 | 0.833 | 14 | 18 |
| raw_sentence | eps=0.1,k=10 | 5228 | 774 | 0.828 | 0.35 | 0.851 | 0.819 | 5 | 28 |
| clean_sentence | none | 5210 | 5210 | 0.819 | 0.7 | 0.828 | 0.796 | 8 | 30 |
| clean_sentence | eps=0.1,k=10 | 5210 | 758 | 0.824 | 0.2 | 0.842 | 0.814 | 7 | 28 |
| clean_title | none | 5210 | 5210 | 0.810 | 0.4 | 0.824 | 0.787 | 18 | 21 |
| clean_title | eps=0.1,k=10 | 5210 | 770 | 0.824 | 0.41 | 0.837 | 0.805 | 8 | 28 |
| clean_window | none | 5215 | 5215 | 0.824 | 0.23 | 0.842 | 0.814 | 18 | 17 |
| clean_window | eps=0.1,k=10 | 5215 | 1034 | 0.855 | 0.31 | 0.864 | 0.846 | 7 | 23 |
| clean_passage | none | 686 | 686 | 0.783 | 0.2 | 0.792 | 0.747 | 18 | 28 |
| clean_passage | eps=0.1,k=10 | 686 | 370 | 0.796 | 0.04 | 0.824 | 0.796 | 15 | 24 |

## 5. Real end-to-end runtime of EntailmentFactChecker (from `timing`, final defaults)

| min_overlap | top_k | seconds | acc@default_threshold |
|---|---|---|---|
| 0.0 | all | 598.2 | 0.837 |
| 0.1 | 10 | 134.0 | 0.864 |
| 0.1 | 5 | 99.4 | 0.860 |
| 0.2 | 3 | 77.8 | 0.837 |

## Final configuration

variant=clean_window, strategy=max_entail, min_overlap=0.1, top_k=10, threshold=0.31 (accuracy is maximal for thresholds 0.29-0.33)

Accuracy 191/221 = 0.8643, macro-F1 0.864, FP=7, FN=23, model calls=1034
