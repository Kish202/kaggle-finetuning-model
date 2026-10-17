"""
Feature pipeline for stylometric authorship attribution.

Three feature groups, each a sparse matrix, concatenated:
  1. char n-grams  (raw `char` analyzer so punctuation + spacing patterns that
                    cross word boundaries are kept, e.g. " — ", "'s ", ": ")
  2. word n-grams  (vocabulary / phrasing, punctuation kept as tokens)
  3. hand-crafted stylometry (dense, scaled): punctuation rates, sentence /
     word length statistics, lexical diversity, unicode usage.
"""

import re
import string

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z']+")

PUNCT_CHARS = {
    "em_dash": "—",
    "en_dash": "–",
    "hyphen": "-",
    "curly_apos": "’",
    "straight_apos": "'",
    "curly_open": "“",
    "curly_close": "”",
    "straight_quote": '"',
    "semicolon": ";",
    "colon": ":",
    "comma": ",",
    "period": ".",
    "question": "?",
    "exclaim": "!",
    "paren_open": "(",
    "ellipsis": "…",
    "asterisk": "*",
    "slash": "/",
}

# very common function words: topic-independent style markers
FUNCTION_WORDS = (
    "the a an and or but of to in on at by for with from as that this these those "
    "it its is are was were be been being not no nor so yet if then than because "
    "while when where which who whom whose what how why can could may might must "
    "shall should will would do does did have has had very just only also even "
    "rather quite perhaps indeed however therefore thus hence moreover instead "
    "yet still already often sometimes always never both either neither each every "
    "we our us you your they their them he she his her i my me"
).split()


def _stylometry_row(text: str) -> dict:
    n_chars = max(len(text), 1)
    words = _WORD.findall(text)
    n_words = max(len(words), 1)
    sents = [s for s in _SENT_SPLIT.split(text.strip()) if s]
    sent_lens = np.array([len(_WORD.findall(s)) for s in sents]) if sents else np.array([n_words])
    word_lens = np.array([len(w) for w in words]) if words else np.array([0])
    lower = text.lower()
    lower_words = [w.lower() for w in words]

    f = {}
    # punctuation per 100 words
    for name, ch in PUNCT_CHARS.items():
        f[f"p_{name}"] = 100.0 * text.count(ch) / n_words
    f["p_any_punct"] = 100.0 * sum(c in string.punctuation for c in text) / n_words
    f["p_non_ascii"] = 100.0 * sum(ord(c) > 127 for c in text) / n_words
    f["p_digits"] = 100.0 * sum(c.isdigit() for c in text) / n_words
    f["p_upper"] = 100.0 * sum(c.isupper() for c in text) / n_chars

    # length / rhythm
    f["n_words"] = n_words
    f["n_sents"] = len(sents)
    f["sent_len_mean"] = sent_lens.mean()
    f["sent_len_std"] = sent_lens.std()
    f["sent_len_max"] = sent_lens.max()
    f["sent_len_min"] = sent_lens.min()
    f["word_len_mean"] = word_lens.mean()
    f["word_len_std"] = word_lens.std()
    f["long_word_rate"] = (word_lens >= 8).mean()
    f["short_word_rate"] = (word_lens <= 3).mean()

    # lexical diversity
    f["ttr"] = len(set(lower_words)) / n_words
    f["hapax_rate"] = sum(1 for w, c in pd.Series(lower_words).value_counts().items() if c == 1) / n_words

    # starting / ending habits
    f["starts_with_quote"] = float(text[:1] in "\"“'")
    f["ends_with_period"] = float(text.rstrip()[-1:] == ".")
    f["ends_with_question"] = float(text.rstrip()[-1:] == "?")
    f["first_word_the"] = float(lower_words[:1] == ["the"])
    f["first_word_i"] = float(lower_words[:1] == ["i"])

    # contractions and dialogue
    f["contraction_rate"] = 100.0 * len(re.findall(r"\w[’']\w", text)) / n_words
    f["dash_spaced"] = text.count(" — ") + text.count(" - ")
    f["dash_tight"] = text.count("—") - text.count(" — ")

    # function word rates
    counts = pd.Series(lower_words).value_counts()
    for w in FUNCTION_WORDS:
        f[f"fw_{w}"] = 100.0 * counts.get(w, 0) / n_words
    return f


def stylometry(texts) -> pd.DataFrame:
    return pd.DataFrame([_stylometry_row(t) for t in texts])


def prompt_relative(style: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Express each passage's style relative to the other answers to the same
    prompt (group). Cancels prompt-level effects (e.g. a creative prompt makes
    every model use more dashes) so what's left is the model's own habit.

    Returns [raw | raw - group_mean | within-group percentile rank]."""
    df = pd.DataFrame(style)
    g = df.groupby(groups)
    rel = (df - g.transform("mean")).values
    rank = g.rank(pct=True).values
    return np.hstack([style, rel, rank]).astype(np.float32)


class StyleFeaturizer:
    """fit/transform wrapper that returns one combined sparse matrix."""

    def __init__(
        self,
        char_ngram=(1, 6),
        word_ngram=(1, 2),
        char_max_features=400_000,
        word_max_features=200_000,
        use_char=True,
        use_word=True,
        use_style=True,
        style_weight=1.0,
    ):
        self.use_char, self.use_word, self.use_style = use_char, use_word, use_style
        self.style_weight = style_weight
        self.char_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=char_ngram,
            min_df=2,
            sublinear_tf=True,
            lowercase=False,
            max_features=char_max_features,
            dtype=np.float32,
        )
        self.word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=word_ngram,
            min_df=2,
            sublinear_tf=True,
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",
            max_features=word_max_features,
            dtype=np.float32,
        )
        self.scaler = StandardScaler()

    def _blocks(self, texts, fit):
        blocks = []
        if self.use_char:
            blocks.append(self.char_vec.fit_transform(texts) if fit else self.char_vec.transform(texts))
        if self.use_word:
            blocks.append(self.word_vec.fit_transform(texts) if fit else self.word_vec.transform(texts))
        if self.use_style:
            s = stylometry(texts).values.astype(np.float32)
            s = self.scaler.fit_transform(s) if fit else self.scaler.transform(s)
            s = np.clip(s, -5, 5) * self.style_weight
            blocks.append(csr_matrix(s))
        return blocks

    def fit_transform(self, texts):
        return hstack(self._blocks(list(texts), fit=True)).tocsr()

    def transform(self, texts):
        return hstack(self._blocks(list(texts), fit=False)).tocsr()
