import re
from typing import List
from nltk.stem import PorterStemmer

# Built-in stop-word list, so no nltk.download() (and no network) is needed at grading time.
STOP_WORDS = frozenset("""
a an the and or but if of at by for with about against between into through during before after above below
to from up down in out on off over under again further then once here there when where why how all any both each
few more most other some such no nor not only own same so than too very s t can will just don should now is are
was were be been being have has had having do does did doing i me my we our you your he him his she her it its
they them their what which who whom this that these those am as until while also
""".split())

_STEMMER = PorterStemmer()
_STEM_CACHE = {}                       # word -> stem, avoids re-stemming the same word thousands of times
_TOKEN_RE = re.compile(r"[a-z0-9]+")   # letters/digits only -> punctuation is dropped


def tokenize(text: str, stem: bool = True, remove_stopwords: bool = True) -> List[str]:
    """Lowercase -> regex word tokens -> optional stop-word removal -> optional Porter stemming.

    Returns a LIST (order and repeats kept) so count-based metrics such as TF-IDF cosine can use it;
    set-based metrics simply wrap it in set().
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if stem:
        out = []
        for t in tokens:
            if t not in _STEM_CACHE:
                _STEM_CACHE[t] = _STEMMER.stem(t)
            out.append(_STEM_CACHE[t])
        tokens = out
    return tokens
