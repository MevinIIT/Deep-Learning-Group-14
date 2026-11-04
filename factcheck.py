# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
import re
from nltk.tokenize import sent_tokenize


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text".
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


def extract_clean_sentences(passages: List[dict]) -> List[str]:
    """
    Extracts, cleans, and deduplicates individual sentences from Wikipedia passages.
    Prevents sequence truncation and attention dilution caused by long text chunks.
    """
    sentences = []
    seen = set()
    for passage in passages:
        text = passage.get("text", "")
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            continue
        for s in sent_tokenize(text):
            cleaned = s.strip()
            # Exclude very short noisy fragments
            if len(cleaned.split()) >= 4 and cleaned not in seen:
                seen.add(cleaned)
                sentences.append(cleaned)
    return sentences


def compute_token_overlap(sentence: str, fact: str) -> float:
    """
    Computes token-level recall of the fact in the candidate sentence.
    """
    sent_tokens = set(re.findall(r'\w+', sentence.lower()))
    fact_tokens = set(re.findall(r'\w+', fact.lower()))
    if not fact_tokens:
        return 0.0
    return len(sent_tokens.intersection(fact_tokens)) / len(fact_tokens)


def prune_candidate_sentences(sentences: List[str], fact: str, min_overlap: float = 0.08, max_candidates: int = 5) -> List[str]:
    """
    Prunes passages/sentences that fail to meet a lexical overlap threshold.
    Selects top-k candidates to optimize computational efficiency and avoid OOM.
    """
    scored = []
    for s in sentences:
        overlap = compute_token_overlap(s, fact)
        if overlap >= min_overlap:
            scored.append((overlap, s))

    # Sort descending by token overlap
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_candidates]]


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

        # Dynamically determine the entailment index from model config
        self.entailment_idx = 0
        label2id = getattr(self.model.config, "label2id", None)
        if label2id:
            for label, idx in label2id.items():
                if "entail" in label.lower():
                    self.entailment_idx = idx
                    break

    def check_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Infers logical entailment probability of hypothesis given the premise.
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors='pt',
                truncation=True,
                max_length=256,
                padding=True
            )
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}

            outputs = self.model(**inputs)
            logits = outputs.logits  # shape: [1, 3]

            # Convert logits to class probabilities via softmax
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)
            entailment_prob = probabilities[self.entailment_idx].item()

        # Prevent GPU memory accumulation
        del inputs, outputs, logits, probabilities
        if self.cuda:
            torch.cuda.empty_cache()

        return entailment_prob


class FactChecker(object):
    """
    Fact checker base type
    """
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return np.random.choice(["S", "NS"])


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me (Assigned to Kovindu & Mevin)")


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model, threshold: float = 0.50):
        self.ent_model = ent_model
        self.threshold = threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        # Step 1: Preprocess passages into distinct sentences
        all_sentences = extract_clean_sentences(passages)
        if not all_sentences:
            return "NS"

        # Step 2: Prune candidates based on word overlap to accelerate inference
        candidates = prune_candidate_sentences(all_sentences, fact, min_overlap=0.08, max_candidates=5)

        # Fall back to first few sentences if pruning leaves no candidates
        if not candidates:
            candidates = all_sentences[:3]

        # Step 3: Run entailment inference and compute max probability
        max_entailment_prob = 0.0
        for sent in candidates:
            prob = self.ent_model.check_entailment(premise=sent, hypothesis=fact)
            if prob > max_entailment_prob:
                max_entailment_prob = prob
                # Early termination if strong evidence is found
                if max_entailment_prob >= 0.92:
                    break

        # Step 4: Map real-valued probability to binary S / NS decision
        return "S" if max_entailment_prob >= self.threshold else "NS"


class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me (Optional)")

    def get_dependencies(self, sent: str):
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations