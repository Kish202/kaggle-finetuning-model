"""
Compare feature / model variants under identical prompt-grouped CV.

    python src/experiment.py            # run all variants
    python src/experiment.py base style # run selected variants
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import StyleFeaturizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
N_FOLDS = 5


def lr(C=10.0):
    return lambda: LogisticRegression(C=C, max_iter=5000, class_weight="balanced")


def svc(C=0.5):
    return lambda: LinearSVC(C=C, class_weight="balanced", max_iter=20000)


VARIANTS = {
    # name: (featurizer kwargs, model factory)
    "base_charwb":  (dict(char_ngram=(1, 5), use_style=False), lr(10)),
    "char16":       (dict(char_ngram=(1, 6), use_style=False), lr(10)),
    "char16_style": (dict(char_ngram=(1, 6), use_style=True, style_weight=1.0), lr(10)),
    "char16_style3": (dict(char_ngram=(1, 6), use_style=True, style_weight=3.0), lr(10)),
    "style_only":   (dict(use_char=False, use_word=False, use_style=True), lr(1)),
    "char16_style_svc": (dict(char_ngram=(1, 6), use_style=True, style_weight=1.0), svc(0.3)),
    "char26_style_svc": (dict(char_ngram=(2, 6), use_style=True, style_weight=1.0), svc(0.3)),
}


def run(name, feat_kwargs, model_factory, train, y, groups, classes):
    t0 = time.time()
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_pred = np.zeros(len(train), dtype=int)
    oof_score = None
    for tr_idx, va_idx in gkf.split(train, y, groups=groups):
        fz = StyleFeaturizer(**feat_kwargs)
        x_tr = fz.fit_transform(train.text.iloc[tr_idx])
        x_va = fz.transform(train.text.iloc[va_idx])
        clf = model_factory()
        clf.fit(x_tr, y[tr_idx])
        if hasattr(clf, "predict_proba"):
            s = clf.predict_proba(x_va)
        else:
            s = clf.decision_function(x_va)
        if oof_score is None:
            oof_score = np.zeros((len(train), s.shape[1]), dtype=np.float32)
        oof_score[va_idx] = s
        oof_pred[va_idx] = s.argmax(1)
    ba = balanced_accuracy_score(y, oof_pred)
    lab = lambda m: "openai" if m.startswith("gpt") else m.split("-")[0]
    lab_ba = balanced_accuracy_score([lab(classes[i]) for i in y], [lab(classes[i]) for i in oof_pred])
    print(f"{name:20s} bal_acc={ba:.4f}  lab_acc={lab_ba:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(oof_score, columns=classes)
    df.insert(0, "id", train["id"].values)
    df.to_parquet(OUT / f"oof_{name}.parquet", index=False)
    return ba


def main():
    train = pd.read_csv(DATA / "train.csv")
    le = LabelEncoder()
    y = le.fit_transform(train["model"])
    classes = le.classes_
    groups = train["prompt_id"].values

    names = sys.argv[1:] or list(VARIANTS)
    results = {}
    for n in names:
        kw, mf = VARIANTS[n]
        results[n] = run(n, kw, mf, train, y, groups, classes)
    print("\nsummary:")
    for n, s in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {n:20s} {s:.4f}")


if __name__ == "__main__":
    main()
