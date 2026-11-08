import torch
from typing import List
import numpy as np
import spacy
import gc

# Part 1 (word overlap) helpers live in their own files; make_submission.py inlines them for submission.
from preprocessing import tokenize
from overlap_metrics import METRICS


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        raise Exception("Not implemented")

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        # return something


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    """
    Part 1: word-overlap fact checker.  [Kovindu: class + preprocessing + recall metric; Mevin: tuning + extra metrics]
    Pipeline: tokenize fact and passages (preprocessing.tokenize) -> score with a metric (overlap_metrics.METRICS)
    -> predict "S" if score >= threshold, else "NS".

    :param threshold: decision threshold. Provisional 0.65; Mevin sets the final value using tune_word_overlap.py
    :param metric: key of overlap_metrics.METRICS ("recall" done; "jaccard", "dice" are Mevin's)
    :param pool: "max" = score each passage separately, keep the best; "union" = merge all passages into one word set
    :param stem: apply Porter stemming
    :param remove_stopwords: drop stop words before comparing
    """
    def __init__(self, threshold: float = 0.65, metric: str = "recall", pool: str = "max",
                 stem: bool = True, remove_stopwords: bool = True):
        if metric not in METRICS:
            raise ValueError("unknown metric " + repr(metric) + "; choose from " + repr(list(METRICS)))
        if pool not in ("max", "union"):
            raise ValueError("pool must be 'max' or 'union'")
        self.threshold = threshold
        self.metric = metric
        self.pool = pool
        self.stem = stem
        self.remove_stopwords = remove_stopwords

    def score(self, fact: str, passages: List[dict]) -> float:
        fact_set = set(tokenize(fact, self.stem, self.remove_stopwords))
        if not fact_set or not passages:
            return 0.0
        passage_sets = [set(tokenize(p["text"], self.stem, self.remove_stopwords)) for p in passages]
        if self.pool == "union":
            passage_sets = [set().union(*passage_sets)]
        metric_fn = METRICS[self.metric]
        return max(metric_fn(fact_set, pset) for pset in passage_sets)

    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S" if self.score(fact, passages) >= self.threshold else "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
