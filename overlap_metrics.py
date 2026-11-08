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
    
    # [Mevin - TODO] |F & P| / |F | P|.  Return 0.0 if the union is empty
    raise NotImplementedError("TODO(Mevin): implement Jaccard overlap")


def dice(fact_set: Set[str], passage_set: Set[str]) -> float:
    # [Mevin - TODO] 2|F & P| / (|F| + |P|).  Return 0.0 if both sets are empty
    raise NotImplementedError("TODO(Mevin): implement Dice overlap")


METRICS = {
    "recall": recall,
    "jaccard": jaccard,
    "dice": dice,
}
