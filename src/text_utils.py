"""Text preprocessing helpers shared by training and inference."""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^a-z0-9'\s]")
_QUOTED_RE = re.compile(r"[\"“”‘’]")

# Very common English words that carry no signal about credibility.
_STOPWORDS = {
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "but", "by", "can", "did", "do", "does", "for",
    "from", "had", "has", "have", "he", "her", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "our",
    "out", "she", "so", "some", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "to", "up", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "why", "will", "with",
    "would", "you", "your", "said", "says", "say",
}


def clean_text(text: str) -> str:
    """Normalise raw article text for vectorisation."""
    if not text:
        return ""
    text = str(text)
    text = _URL_RE.sub(" ", text)
    text = _QUOTED_RE.sub(" ", text)
    text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def remove_stopwords(text: str) -> str:
    """Drop low-signal English stopwords after cleaning."""
    tokens = [t for t in text.split() if t not in _STOPWORDS]
    return " ".join(tokens)


def preprocess(text: str) -> str:
    """Full cleaning pipeline used before TF-IDF vectorisation."""
    return remove_stopwords(clean_text(text))


def word_count(text: str) -> int:
    return len(text.split()) if text else 0