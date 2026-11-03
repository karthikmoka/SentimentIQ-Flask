"""
SentimentIQ Flask — Text Preprocessing Pipeline

Steps:
  1. Lowercase
  2. Strip HTML tags
  3. Remove URLs
  4. Remove special characters (preserve apostrophes in contractions)
  5. Tokenize
  6. Remove stop-words (but keep negation words)
  7. Lemmatize

Negation words preserved: not, no, never, neither, nor, nobody,
                           nothing, nowhere, none, hardly, barely,
                           scarcely, doesn't, isn't, wasn't, aren't,
                           won't, wouldn't, don't, didn't, haven't,
                           hasn't, hadn't, shouldn't, couldn't, mustn't
"""

import re
import string
import logging

import nltk
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── NLTK downloads (idempotent) ───────────────────────────────────────────────
_NLTK_RESOURCES = [
    ("tokenizers/punkt",       "punkt"),
    ("tokenizers/punkt_tab",   "punkt_tab"),
    ("corpora/stopwords",      "stopwords"),
    ("corpora/wordnet",        "wordnet"),
    ("corpora/omw-1.4",        "omw-1.4"),
]

def _ensure_nltk():
    for path, pkg in _NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception as exc:
                logger.warning("Could not download NLTK resource %s: %s", pkg, exc)

_ensure_nltk()

from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer

# ── Negation words to always keep ─────────────────────────────────────────────
_NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "nobody",
    "nothing", "nowhere", "none", "hardly", "barely",
    "scarcely", "doesn't", "isn't", "wasn't", "aren't",
    "won't", "wouldn't", "don't", "didn't", "haven't",
    "hasn't", "hadn't", "shouldn't", "couldn't", "mustn't",
    "cannot", "can't", "needn't",
}

# ── English stop-words minus negation ─────────────────────────────────────────
_STOP_WORDS: set[str] = set(stopwords.words("english")) - _NEGATION_WORDS

_LEMMATIZER = WordNetLemmatizer()

# ── Compiled regex patterns ────────────────────────────────────────────────────
_URL_RE        = re.compile(r"https?://\S+|www\.\S+")
_SPECIAL_RE    = re.compile(r"[^a-z\s']")   # keep letters, spaces, apostrophes
_MULTI_SPACE   = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Return a cleaned, lemmatized string ready for TF-IDF vectorisation.
    Safe to call with None or empty strings.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Strip HTML
    try:
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except Exception:
        pass

    # 3. Remove URLs
    text = _URL_RE.sub(" ", text)

    # 4. Remove special characters (keep apostrophes for "don't", "can't" etc.)
    text = _SPECIAL_RE.sub(" ", text)

    # 5. Collapse whitespace
    text = _MULTI_SPACE.sub(" ", text).strip()

    # 6. Tokenise
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # 7. Remove stop-words (keep negation)  +  8. Lemmatize
    tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok and (tok in _NEGATION_WORDS or tok not in _STOP_WORDS)
    ]

    return " ".join(tokens)


def preprocess_series(series):
    """Vectorised preprocessing of a pandas Series. Returns a new Series."""
    return series.fillna("").astype(str).map(clean_text)
