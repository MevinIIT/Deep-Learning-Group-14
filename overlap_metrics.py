# CONTRACT (do not change): every metric is  f(fact_set: Set[str], passage_set: Set[str]) -> float  in [0, 1],
# higher = fact better covered by the passage.  Register each metric in METRICS at the bottom; factcheck.py looks
# metrics up by name, so a new entry in METRICS is automatically usable via WordRecallThresholdFactChecker(metric="name").
from typing import Set


def recall(fact_set: Set[str], passage_set: Set[str]) -> float:
    # Share of the fact's words that also appear in the passage: |F & P| / |F|
    if not fact_set:
        return 0.0
    return len(fact_set & passage_set) / len(fact_set)


def jaccard(fact_set: Set[str], passage_set: Set[str]) -> float:
    # Shared words divided by all distinct words in the fact and the passage: |F & P| / |F | P|
    union = fact_set | passage_set
    if len(union) == 0:
        return 0.0
    return len(fact_set & passage_set) / len(union)


def dice(fact_set: Set[str], passage_set: Set[str]) -> float:
    # Twice the shared words divided by the total size of both sets: 2|F & P| / (|F| + |P|)
    total = len(fact_set) + len(passage_set)
    if total == 0:
        return 0.0
    return 2 * len(fact_set & passage_set) / total


METRICS = {
    "recall": recall,
    "jaccard": jaccard,
    "dice": dice,
}
