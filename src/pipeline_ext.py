"""
Step 1: add the public external passage pool to the CPU models.

- Official train is still the only thing scored (GroupKFold on prompt_id).
- External rows are added to every training fold, except rows generated from a
  prompt that sits in the current validation fold (avoids topic leakage).
- Models: TF-IDF+LR (sparse) and stylometry+prompt-relative+HGB (dense),
  log-linear blend, then cluster-and-assign post-processing.

    python src/pipeline_ext.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assignment import assign_within_groups, cluster_by_topic  # noqa: E402
from features import StyleFeaturizer, prompt_relative, stylometry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
N_FOLDS = 5
CLUSTER_THR = 0.92
W_OFFICIAL = 3.0  # official rows outweigh external ones (external is 3x larger and from different routes)
LAB = lambda m: "openai" if m.startswith("gpt") else m.split("-")[0]  # noqa: E731


def load_external() -> pd.DataFrame:
    ext = pd.read_csv(DATA / "external" / "prvsiyan" / "passages.csv")
    ext = ext[ext["passes_funnel"] & ~ext["has_markdown"] & ext["in_word_range"]]
    ext = ext[["prompt_id", "source_prompt_id", "text", "model"]].reset_index(drop=True)
    ext["prompt_id"] = "ext_" + ext["prompt_id"].astype(str)
    return ext


def hgb():
    return HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, class_weight="balanced", random_state=42,
    )


def main():
    train = pd.read_csv(DATA / "train.csv")
    ext = load_external()
    classes = np.array(sorted(train["model"].unique()))
    c2i = {c: i for i, c in enumerate(classes)}
    y = train["model"].map(c2i).values
    y_ext = ext["model"].map(c2i).values
    print(f"official={len(train)}  external={len(ext)} ({ext.prompt_id.nunique()} prompts)")

    # dense style features. Official rows use blind clusters (as at test time);
    # external rows use their known prompt ids.
    t0 = time.time()
    cl_train = cluster_by_topic(train["text"].tolist(), distance_threshold=CLUSTER_THR)
    S_train = prompt_relative(stylometry(train["text"]).values.astype(np.float32), cl_train)
    S_ext = prompt_relative(stylometry(ext["text"]).values.astype(np.float32), ext["prompt_id"].values)
    print(f"style features {S_train.shape[1]} dims ({time.time()-t0:.0f}s)")

    oof_tfidf = np.zeros((len(train), len(classes)), dtype=np.float32)
    oof_style = np.zeros((len(train), len(classes)), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(GroupKFold(N_FOLDS).split(train, y, groups=train["prompt_id"])):
        t0 = time.time()
        va_prompts = set(train["prompt_id"].iloc[va_idx])
        ext_ok = ~ext["source_prompt_id"].isin(va_prompts).values  # drop regenerations of held-out prompts

        texts_tr = pd.concat([train["text"].iloc[tr_idx], ext["text"][ext_ok]])
        y_tr = np.concatenate([y[tr_idx], y_ext[ext_ok]])
        w_tr = np.concatenate([np.full(len(tr_idx), W_OFFICIAL), np.ones(ext_ok.sum())])

        fz = StyleFeaturizer(char_ngram=(1, 5), use_style=False)
        fz.char_vec.set_params(analyzer="char_wb")
        x_tr = fz.fit_transform(texts_tr)
        x_va = fz.transform(train["text"].iloc[va_idx])
        lr = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")
        lr.fit(x_tr, y_tr, sample_weight=w_tr)
        oof_tfidf[va_idx] = lr.predict_proba(x_va)

        X_tr = np.vstack([S_train[tr_idx], S_ext[ext_ok]])
        m = hgb().fit(X_tr, y_tr, sample_weight=w_tr)
        oof_style[va_idx] = m.predict_proba(S_train[va_idx])

        print(
            f"fold {fold}: n_train={len(y_tr)}  tfidf={balanced_accuracy_score(y[va_idx], oof_tfidf[va_idx].argmax(1)):.4f}  "
            f"style={balanced_accuracy_score(y[va_idx], oof_style[va_idx].argmax(1)):.4f}  ({time.time()-t0:.0f}s)",
            flush=True,
        )

    def rep(name, p):
        pr = p.argmax(1)
        print(f"{name:34s} bal_acc={balanced_accuracy_score(y, pr):.4f}  lab_acc={balanced_accuracy_score([LAB(classes[i]) for i in y], [LAB(classes[i]) for i in pr]):.4f}")

    print()
    rep("tfidf_lr (+ext)", oof_tfidf)
    rep("style_hgb (+ext)", oof_style)
    lt, ls = np.log(oof_tfidf + 1e-6), np.log(oof_style + 1e-6)
    best = max(((balanced_accuracy_score(y, (w * lt + (1 - w) * ls).argmax(1)), w) for w in np.linspace(0, 1, 21)))
    w = best[1]
    blend = np.exp(w * lt + (1 - w) * ls)
    blend /= blend.sum(1, keepdims=True)
    rep(f"blend w_tfidf={w:.2f}", blend)
    rep("blend + assign (clustered)", np.eye(len(classes))[assign_within_groups(blend, cl_train)])
    rep("blend + assign (true groups, bound)", np.eye(len(classes))[assign_within_groups(blend, train["prompt_id"].values)])

    for name, arr in [("tfidf_ext", oof_tfidf), ("style_ext", oof_style), ("blend_ext", blend)]:
        df = pd.DataFrame(arr, columns=classes)
        df.insert(0, "id", train["id"].values)
        df.to_parquet(OUT / f"oof_{name}.parquet", index=False)


if __name__ == "__main__":
    main()
