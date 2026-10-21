"""
Two-model ensemble under prompt-grouped CV:
  A) TF-IDF (char_wb 1-5 + word 1-2)  -> LogisticRegression   (sparse, linear)
  B) hand-crafted stylometry (~130 dense) -> HistGradientBoosting (non-linear)
Probabilities are averaged with a weight tuned on OOF.

    python src/ensemble_style.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import StyleFeaturizer, stylometry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
N_FOLDS = 5
LAB = lambda m: "openai" if m.startswith("gpt") else m.split("-")[0]  # noqa: E731


def report(name, y, proba, classes):
    pred = proba.argmax(1)
    ba = balanced_accuracy_score(y, pred)
    lab_ba = balanced_accuracy_score([LAB(classes[i]) for i in y], [LAB(classes[i]) for i in pred])
    print(f"{name:28s} bal_acc={ba:.4f}  lab_acc={lab_ba:.4f}", flush=True)
    return ba


def main():
    train = pd.read_csv(DATA / "train.csv")
    le = LabelEncoder()
    y = le.fit_transform(train["model"])
    classes = le.classes_
    k = len(classes)
    groups = train["prompt_id"].values

    # precompute dense style features once (they don't depend on the fold)
    t0 = time.time()
    style_df = stylometry(train.text)
    style_all = style_df.values.astype(np.float32)
    print(f"style features: {style_all.shape}  ({time.time()-t0:.0f}s)")

    oof_tfidf = np.zeros((len(train), k), dtype=np.float32)
    oof_style = np.zeros((len(train), k), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=N_FOLDS).split(train, y, groups=groups)):
        t0 = time.time()
        # A) tfidf + LR
        fz = StyleFeaturizer(char_ngram=(1, 5), use_style=False)
        fz.char_vec.set_params(analyzer="char_wb")
        x_tr = fz.fit_transform(train.text.iloc[tr_idx])
        x_va = fz.transform(train.text.iloc[va_idx])
        lr = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")
        lr.fit(x_tr, y[tr_idx])
        oof_tfidf[va_idx] = lr.predict_proba(x_va)

        # B) stylometry + HGB
        hgb = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42,
        )
        hgb.fit(style_all[tr_idx], y[tr_idx])
        oof_style[va_idx] = hgb.predict_proba(style_all[va_idx])

        print(
            f"fold {fold}: tfidf={balanced_accuracy_score(y[va_idx], oof_tfidf[va_idx].argmax(1)):.4f} "
            f"style={balanced_accuracy_score(y[va_idx], oof_style[va_idx].argmax(1)):.4f}  ({time.time()-t0:.0f}s)",
            flush=True,
        )

    print()
    report("tfidf_lr", y, oof_tfidf, classes)
    report("style_hgb", y, oof_style, classes)

    # weight sweep for the blend (geometric mean of probabilities works well for LR+trees)
    best = (0, None)
    for w in np.linspace(0, 1, 21):
        blend = w * np.log(oof_tfidf + 1e-6) + (1 - w) * np.log(oof_style + 1e-6)
        ba = balanced_accuracy_score(y, blend.argmax(1))
        if ba > best[0]:
            best = (ba, w)
    w = best[1]
    blend = w * np.log(oof_tfidf + 1e-6) + (1 - w) * np.log(oof_style + 1e-6)
    print(f"\nbest blend weight tfidf={w:.2f} style={1-w:.2f}")
    report("blend", y, blend, classes)

    # feature importance proxy: which style features matter (permutation on last fold is slow; use HGB on all data)
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=10,
        l2_regularization=1.0, class_weight="balanced", random_state=42,
    ).fit(style_all, y)
    from sklearn.inspection import permutation_importance
    names = style_df.columns
    sub = np.random.RandomState(0).choice(len(train), 600, replace=False)
    pi = permutation_importance(hgb, style_all[sub], y[sub], n_repeats=3, random_state=0, scoring="balanced_accuracy")
    imp = pd.Series(pi.importances_mean, index=names).sort_values(ascending=False)
    print("\ntop stylometric features (permutation importance):")
    print(imp.head(20).round(4).to_string())

    for name, arr in [("tfidf_lr", oof_tfidf), ("style_hgb", oof_style), ("blend", np.exp(blend) / np.exp(blend).sum(1, keepdims=True))]:
        df = pd.DataFrame(arr, columns=classes)
        df.insert(0, "id", train["id"].values)
        df.to_parquet(OUT / f"oof_{name}.parquet", index=False)


if __name__ == "__main__":
    main()
