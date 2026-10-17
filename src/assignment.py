"""
Exploit the dataset structure: each prompt is answered by every model at most
once. Given per-passage class probabilities and a grouping of passages into
prompts, solve a linear assignment per group so no model is used twice.

Step 1 (oracle): use true prompt_id groups on OOF probs -> upper bound.
Step 2 (realistic): cluster passages by topic to recover prompt groups, then
assign. Cluster quality is measured against the true prompt_id.

    python src/assignment.py outputs/oof_blend.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def assign_within_groups(proba: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """For each group, Hungarian assignment of passages -> classes maximising
    sum of log-probabilities, each class used at most once per group.
    Groups larger than n_classes fall back to argmax for the overflow."""
    k = proba.shape[1]
    logp = np.log(proba + 1e-9)
    pred = proba.argmax(1).copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) <= 1:
            continue
        cost = -logp[idx]  # rows=passages, cols=classes
        if len(idx) > k:
            # more passages than classes (cluster merged two prompts):
            # allow each class up to ceil(n/k) times by tiling columns
            reps = int(np.ceil(len(idx) / k))
            cost = np.tile(cost, (1, reps))
        r, c = linear_sum_assignment(cost)
        pred[idx[r]] = c % k
    return pred


def cluster_by_topic(texts, n_clusters=None, distance_threshold=None) -> np.ndarray:
    """Group passages that answer the same prompt. Same-prompt passages share
    content words heavily, so word TF-IDF cosine + average linkage works.
    Use `distance_threshold` at test time, where the prompt count is unknown."""
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.5,
        sublinear_tf=True, stop_words="english",
    )
    x = vec.fit_transform(texts)
    sim = cosine_similarity(x)
    dist = np.clip(1.0 - sim, 0, 2)
    cl = AgglomerativeClustering(
        n_clusters=n_clusters, distance_threshold=distance_threshold,
        metric="precomputed", linkage="average",
    )
    return cl.fit_predict(dist)


def main():
    oof_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/oof_blend.parquet"
    train = pd.read_csv(DATA / "train.csv")
    oof = pd.read_parquet(ROOT / oof_path)
    classes = np.array([c for c in oof.columns if c != "id"])
    proba = oof[classes].values.astype(np.float64)
    proba = proba / proba.sum(1, keepdims=True)
    y = pd.Categorical(train["model"], categories=classes).codes

    base = balanced_accuracy_score(y, proba.argmax(1))
    print(f"source: {oof_path}")
    print(f"per-passage argmax            bal_acc={base:.4f}")

    # --- oracle grouping: true prompt ids
    pred_oracle = assign_within_groups(proba, train["prompt_id"].values)
    print(f"assignment, TRUE prompt groups bal_acc={balanced_accuracy_score(y, pred_oracle):.4f}   (upper bound)")

    # --- realistic grouping: cluster by topic with a distance threshold
    # (prompt count is unknown at test time, so no fixed n_clusters)
    for thr in [0.80, 0.85, 0.88, 0.90, 0.92, 0.94]:
        cl = cluster_by_topic(train["text"].tolist(), distance_threshold=thr)
        ari = adjusted_rand_score(train["prompt_id"], cl)
        sizes = pd.Series(cl).value_counts()
        pred_cl = assign_within_groups(proba, cl)
        print(
            f"assignment, thr={thr:.2f} -> {len(sizes):3d} clusters  bal_acc={balanced_accuracy_score(y, pred_cl):.4f}   "
            f"ARI={ari:.3f}  sizes: min={sizes.min()} med={int(sizes.median())} max={sizes.max()}"
        )


if __name__ == "__main__":
    main()
