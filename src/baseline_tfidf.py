"""
Baseline: TF-IDF (char + word n-grams) -> Logistic Regression.

- Cross-validation is GroupKFold on prompt_id, because the hidden test set uses
  prompts that never appear in train. Row-level CV would leak prompt/topic info
  and overestimate the score.
- Metric is balanced accuracy (the competition metric).
- Writes OOF probabilities (for later ensembling) and a submission.parquet.

Usage:
    python src/baseline_tfidf.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

N_FOLDS = 5
SEED = 42


def build_vectorizers():
    """Two views of the text: character-level (punctuation, morphology) and
    word-level (vocabulary, phrasing). Both are classic stylometry features."""
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 5),
        min_df=2,
        sublinear_tf=True,
        lowercase=False,  # capitalisation is a stylistic signal
        max_features=300_000,
    )
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b|[^\w\s]",  # keep punctuation as tokens
        max_features=200_000,
    )
    return char_vec, word_vec


def featurize(char_vec, word_vec, texts, fit=False):
    if fit:
        xc = char_vec.fit_transform(texts)
        xw = word_vec.fit_transform(texts)
    else:
        xc = char_vec.transform(texts)
        xw = word_vec.transform(texts)
    return hstack([xc, xw]).tocsr()


def make_model():
    return LogisticRegression(
        C=10.0,
        max_iter=3000,
        class_weight="balanced",
    )


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    le = LabelEncoder()
    y = le.fit_transform(train["model"])
    classes = le.classes_
    n_classes = len(classes)

    oof = np.zeros((len(train), n_classes), dtype=np.float32)
    gkf = GroupKFold(n_splits=N_FOLDS)

    print(f"train={len(train)}  prompts={train.prompt_id.nunique()}  classes={n_classes}")
    for fold, (tr_idx, va_idx) in enumerate(
        gkf.split(train, y, groups=train["prompt_id"])
    ):
        char_vec, word_vec = build_vectorizers()
        x_tr = featurize(char_vec, word_vec, train.text.iloc[tr_idx], fit=True)
        x_va = featurize(char_vec, word_vec, train.text.iloc[va_idx])

        clf = make_model()
        clf.fit(x_tr, y[tr_idx])
        oof[va_idx] = clf.predict_proba(x_va)

        fold_ba = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        print(f"fold {fold}: balanced_acc={fold_ba:.4f}  (n_feat={x_tr.shape[1]})")

    pred = oof.argmax(1)
    ba = balanced_accuracy_score(y, pred)
    print(f"\nOOF balanced accuracy: {ba:.4f}")

    # per-class recall: shows which models are hard to separate
    cm = confusion_matrix(y, pred, labels=range(n_classes))
    recall = cm.diagonal() / cm.sum(1)
    report = pd.DataFrame({"model": classes, "recall": recall, "n": cm.sum(1)})
    report["top_confusion"] = [
        classes[np.argsort(cm[i])[-2]] if cm[i].argmax() == i else classes[cm[i].argmax()]
        for i in range(n_classes)
    ]
    print("\nper-class recall:")
    print(report.sort_values("recall").to_string(index=False))

    # save OOF probs for ensembling later
    oof_df = pd.DataFrame(oof, columns=classes)
    oof_df.insert(0, "id", train["id"])
    oof_df.to_parquet(OUT / "oof_tfidf_lr.parquet", index=False)

    # refit on all data and predict test
    char_vec, word_vec = build_vectorizers()
    x_all = featurize(char_vec, word_vec, train.text, fit=True)
    x_te = featurize(char_vec, word_vec, test.text)
    clf = make_model()
    clf.fit(x_all, y)
    test_proba = clf.predict_proba(x_te)

    test_df = pd.DataFrame(test_proba, columns=classes)
    test_df.insert(0, "id", test["id"])
    test_df.to_parquet(OUT / "test_proba_tfidf_lr.parquet", index=False)

    sub = pd.DataFrame({"id": test["id"], "model": classes[test_proba.argmax(1)]})
    sub.to_parquet(OUT / "submission.parquet", index=False)
    print(f"\nwrote {OUT / 'submission.parquet'}  ({len(sub)} rows)")
    print(sub.head())


if __name__ == "__main__":
    main()
