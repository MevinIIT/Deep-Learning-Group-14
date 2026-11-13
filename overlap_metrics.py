from typing import Set

def recall(fact_set: Set[str], passage_set: Set[str]) -> float:
    # fraction of the fact's words found in the passage
    if not fact_set:
        return 0.0
    return len(fact_set & passage_set) / len(fact_set)


def jaccard(fact_set: Set[str], passage_set: Set[str]) -> float:
   
    union = fact_set | passage_set
    if len(union) == 0:
        return 0.0
    return len(fact_set & passage_set) / len(union)


def dice(fact_set: Set[str], passage_set: Set[str]) -> float:
    
    total = len(fact_set) + len(passage_set)
    if total == 0:
        return 0.0
    return 2 * len(fact_set & passage_set) / total


METRICS = {
    "recall": recall,
    "jaccard": jaccard,
    "dice": dice,
}
