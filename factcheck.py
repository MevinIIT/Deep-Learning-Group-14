# factcheck.py

import torch
from typing import List, Dict, Optional
import numpy as np
import spacy
import gc
import re
import nltk
from nltk.stem import PorterStemmer


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


# ---------------- text helpers ----------------

# own stopword list, so no nltk download needed
STOP_WORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or
other our ours ourselves out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where which while who whom why
will with would you your yours yourself yourselves s t
""".split())

_STEMMER = PorterStemmer()
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def content_tokens(text: str) -> set:
    # lowercase, remove stopwords, stem
    return {_STEMMER.stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS}


def overlap_recall(fact_tokens: set, text: str) -> float:
    # fraction of fact words found in the text
    if not fact_tokens:
        return 0.0
    return len(fact_tokens & content_tokens(text)) / len(fact_tokens)


def fact_query_tokens(fact: str, title: Optional[str]) -> set:
    # drop the person's name, it's in almost every passage anyway
    toks = content_tokens(fact)
    if title:
        reduced = toks - content_tokens(title)
        if reduced:
            return reduced
    return toks


# sentence splitting (regex fallback if punkt isn't installed)
_PUNKT_OK = None
_FALLBACK_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=["\'(\[]?[A-Z0-9])')


def _punkt_available() -> bool:
    global _PUNKT_OK
    if _PUNKT_OK is None:
        try:
            nltk.data.find('tokenizers/punkt_tab')
            _PUNKT_OK = True
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
                nltk.data.find('tokenizers/punkt_tab')
                _PUNKT_OK = True
            except Exception:
                _PUNKT_OK = False
    return _PUNKT_OK


def split_sentences(text: str) -> List[str]:
    if _punkt_available():
        try:
            return nltk.tokenize.sent_tokenize(text)
        except Exception:
            pass
    return _FALLBACK_SPLIT_RE.split(text)


# passage cleaning
# e.g. '<s>Name Name (born ...) ...</s><s>Career.:Club. He joined ...</s>'
_SEGMENT_RE = re.compile(r'(<s>)?(.*?)(?:</s>|(?=<s>)|$)', re.S)
_HEADING_RE = re.compile(r'^((?:[^.]{1,60}\.:)*[^.]{1,60}\.)(?:\s+|$)')
_MIN_WORDS = 4


def _strip_heading(seg: str) -> str:
    m = _HEADING_RE.match(seg)
    if m:
        heading = m.group(1)
        words = heading.replace('.:', ' ').split()
        if len(words) <= 8 and not any(ch.isdigit() for ch in heading):
            return seg[m.end():]
    return seg


def clean_passage(text: str, title: Optional[str]) -> List[List[str]]:
    # returns paragraphs as lists of sentences
    paragraphs = []
    for m in _SEGMENT_RE.finditer(text):
        opened, seg = m.group(1), m.group(2)
        seg = re.sub(r'\s+', ' ', seg).strip()
        if not seg:
            continue
        if opened:  # remove repeated name / section heading
            if title and seg.startswith(title + ' ') and len(seg) > len(title) + 1:
                seg = seg[len(title) + 1:]
            else:
                seg = _strip_heading(seg)
        sents = [s.strip() for s in split_sentences(seg)]
        sents = [s for s in sents if len(s.split()) >= _MIN_WORDS]
        if sents:
            paragraphs.append(sents)
    return paragraphs


def build_premises(passages: List[dict], context: str = "title", clean: bool = True) -> List[str]:
    # context: sentence / title ("Name: sent") / window (prev + sent) / passage
    premises, seen = [], set()

    def add(p):
        if p not in seen:
            seen.add(p)
            premises.append(p)

    for passage in passages:
        title = passage.get("title")
        text = passage.get("text", "")
        if clean:
            paragraphs = clean_passage(text, title)
        else:
            raw = re.sub(r'\s+', ' ', text).strip()
            paragraphs = [[s for s in split_sentences(raw) if len(s.split()) >= _MIN_WORDS]]
        if context == "passage":
            joined = " ".join(s for para in paragraphs for s in para)
            if joined:
                add(joined)
            continue
        for para in paragraphs:
            for i, sent in enumerate(para):
                if context == "title" and title:
                    add(sent if sent.startswith(title) else title + ": " + sent)
                elif context == "window" and i > 0:
                    add(para[i - 1] + " " + sent)
                else:
                    add(sent)
    return premises


def prune_premises(premises: List[str], fact_tokens: set, min_overlap: float, max_candidates: Optional[int]):
    # keep premises above min_overlap, top k only
    scored = [(p, overlap_recall(fact_tokens, p)) for p in premises]
    kept = [x for x in scored if x[1] >= min_overlap]
    kept.sort(key=lambda x: x[1], reverse=True)
    if max_candidates is not None:
        kept = kept[:max_candidates]
    return kept


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False, batch_size: int = 16, max_length: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda
        self.batch_size = batch_size
        self.max_length = max_length
        self.model.eval()
        # label order from the model config
        self.idx = {"entailment": 0, "neutral": 1, "contradiction": 2}
        id2label = getattr(self.model.config, "id2label", None) or {}
        for i, name in id2label.items():
            name = str(name).lower()
            for key in self.idx:
                if name.startswith(key[:6]):
                    self.idx[key] = int(i)

    def check_entailment(self, premise: str, hypothesis: str) -> Dict[str, float]:
        return self.check_entailment_batch([premise], hypothesis)[0]

    def check_entailment_batch(self, premises: List[str], hypothesis: str) -> List[Dict[str, float]]:
        # batched version
        results = []
        for start in range(0, len(premises), self.batch_size):
            chunk = premises[start:start + self.batch_size]
            with torch.no_grad():
                # only cut the premise if too long, not the fact
                inputs = self.tokenizer(chunk, [hypothesis] * len(chunk), return_tensors='pt',
                                        truncation='only_first', max_length=self.max_length, padding=True)
                if self.cuda:
                    inputs = {key: value.to('cuda') for key, value in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                results.append({k: float(row[i]) for k, i in self.idx.items()})
            # free memory to avoid OOM
            del inputs, outputs, logits, probs
        gc.collect()
        if self.cuda:
            torch.cuda.empty_cache()
        return results


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
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")


# ways to combine premise probabilities into one score
STRATEGIES = ("max_entail", "entail_minus_contra", "entail_vs_contra", "discrete")


def aggregate_score(probs: List[Dict[str, float]], strategy: str) -> float:
    if not probs:
        return 0.0
    if strategy == "max_entail":
        return max(p["entailment"] for p in probs)
    if strategy == "entail_minus_contra":
        return max(p["entailment"] - p["contradiction"] for p in probs)
    if strategy == "entail_vs_contra":
        return max(p["entailment"] / (p["entailment"] + p["contradiction"] + 1e-9) for p in probs)
    if strategy == "discrete":
        # 1 if any sentence's top label is entailment
        return 1.0 if any(max(p, key=p.get) == "entailment" for p in probs) else 0.0
    raise ValueError("Unknown strategy: " + strategy)


class EntailmentFactChecker(FactChecker):
    # clean -> prune -> NLI -> max entailment vs threshold
    # defaults from our experiments
    def __init__(self, ent_model, threshold: float = 0.31, strategy: str = "max_entail", context: str = "window",
                 min_overlap: float = 0.1, max_candidates: Optional[int] = 10, clean: bool = True):
        self.ent_model = ent_model
        self.threshold = threshold
        self.strategy = strategy
        self.context = context
        self.min_overlap = min_overlap
        self.max_candidates = max_candidates
        self.clean = clean

    def score_fact(self, fact: str, passages: List[dict]) -> dict:
        # also returns the evidence (for error analysis)
        title = passages[0].get("title") if passages else None
        premises = build_premises(passages, context=self.context, clean=self.clean)
        kept = prune_premises(premises, fact_query_tokens(fact, title), self.min_overlap, self.max_candidates)
        probs = self.ent_model.check_entailment_batch([p for p, _ in kept], fact) if kept else []
        return {
            "score": aggregate_score(probs, self.strategy),
            "n_premises": len(premises),
            "evidence": [(p, ov, pr) for (p, ov), pr in zip(kept, probs)],
        }

    def predict(self, fact: str, passages: List[dict]) -> str:
        result = self.score_fact(fact, passages)
        if not result["evidence"]:
            return "NS"  # nothing left after pruning
        return "S" if result["score"] >= self.threshold else "NS"


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