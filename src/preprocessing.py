"""Text preprocessing utilities for the polarization detection demo."""

from __future__ import annotations

import re
from typing import Final

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

NEGATION_WORDS: Final[set[str]] = {
    "no",
    "not",
    "nor",
    "never",
    "cannot",
    "none",
    "nobody",
    "nothing",
    "neither",
    "nowhere",
    "hardly",
    "barely",
    "rarely",
    "without",
}

STOP_WORDS: Final[set[str]] = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS

CONTRACTIONS: Final[dict[str, str]] = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "cant": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "might've": "might have",
    "must've": "must have",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what're": "what are",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")
_URL_RE: Final[re.Pattern[str]] = re.compile(r"https?://\S+|www\.\S+|\burl\b|@url\b", flags=re.IGNORECASE)
_MENTION_RE: Final[re.Pattern[str]] = re.compile(r"@\w+")
_HASHTAG_RE: Final[re.Pattern[str]] = re.compile(r"#(\w+)")
_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_CONTRACTION_RE: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(key) for key in sorted(CONTRACTIONS, key=len, reverse=True)),
    flags=re.IGNORECASE,
)


def expand_contractions(text: str) -> str:
    """Expand common English contractions without requiring external NLP data."""

    def replace(match: re.Match[str]) -> str:
        return CONTRACTIONS.get(match.group(0).lower(), match.group(0))

    return _CONTRACTION_RE.sub(replace, text)


def light_lemmatize(token: str) -> str:
    """Apply small rule-based normalization for English social-media text."""

    irregular: dict[str, str] = {
        "children": "child",
        "men": "man",
        "women": "woman",
        "people": "people",
        "stories": "story",
        "countries": "country",
        "parties": "party",
        "republicans": "republican",
        "democrats": "democrat",
        "authoritarians": "authoritarian",
    }

    if token in irregular:
        return irregular[token]

    if len(token) <= 3 or token in NEGATION_WORDS:
        return token

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"

    if token.endswith("ing") and len(token) > 5:
        base = token[:-3]
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base

    if token.endswith("ed") and len(token) > 4:
        base = token[:-2]
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base

    if token.endswith("es") and len(token) > 4:
        return token[:-2]

    if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        return token[:-1]

    return token


def clean_text(text: str) -> str:
    """Clean a raw English social-media text before TF-IDF vectorization."""

    if not isinstance(text, str):
        text = "" if text is None else str(text)

    normalized = text.lower()
    normalized = expand_contractions(normalized)
    normalized = _URL_RE.sub(" ", normalized)
    normalized = _MENTION_RE.sub(" ", normalized)
    normalized = _HASHTAG_RE.sub(r" \1 ", normalized)
    normalized = normalized.replace("'", " ")
    normalized = normalized.replace("’", " ")
    normalized = _SPACE_RE.sub(" ", normalized).strip()

    tokens = _WORD_RE.findall(normalized)
    cleaned_tokens = [
        light_lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) >= 2
    ]

    return " ".join(cleaned_tokens)


def is_too_short(cleaned_text: str, min_tokens: int = 3) -> bool:
    """Return True when the cleaned text is too short for a reliable prediction."""

    return len(cleaned_text.split()) < min_tokens
