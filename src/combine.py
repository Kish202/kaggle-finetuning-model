"""
Final OOF evaluation: blend any number of OOF probability files (log-linear,
weights tuned by coordinate search on balanced accuracy), then apply
cluster-and-assign post-processing.

    python src/combine.py outputs/oof_deberta.parquet outputs/oof_tfidf_lr.parquet outputs/oof_blend_rel.parquet
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assignment import assign_within_groups, cluster_by_topic  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LAB = lambda m: "openai" if m.startswith("gpt") else m.split("-")[0]  # noqa: E731


def main():
    paths = sys.argv[1:]
    train = pd.read_csv(DATA / "train.csv")
    classes = None
    logps = []
    for p in paths:
        df = pd.read_parquet(ROOT / p)
        assert (df["id"].values == train["id"].values).all(), f"id order mismatch in {p}"
        cls = [c for c in df.columns if c != "id"]
        classes = classes or cls
        pr = df[cls].values.astype(np.float64)
        pr /= pr.sum(1, keepdims=True)
        logps.append(np.log(pr + 1e-6))
        print(f"{p:40s} bal_acc={balanced_accuracy_score(train.model.map({c:i for i,c in enumerate(cls)}), pr.argmax(1)):.4f}")
    classes = np.array(classes)
    y = pd.Categorical(train["model"], categories=classes).codes

    def score(w):
        bl = sum(wi * lp for wi, lp in zip(w, logps))
        return balanced_accuracy_score(y, bl.argmax(1))

    grid = np.linspace(0, 1, 11)
    best = max(((score(w), w) for w in product(grid, repeat=len(logps)) if abs(sum(w) - 1) < 1e-9), key=lambda t: t[0])
    w = np.array(best[1]) / sum(best[1])
    print(f"\nweights: {dict(zip([Path(p).stem for p in paths], w.round(2)))}")
    bl = sum(wi * lp for wi, lp in zip(w, logps))
    proba = np.exp(bl)
    proba /= proba.sum(1, keepdims=True)
    pred = proba.argmax(1)
    print(f"blend                       bal_acc={balanced_accuracy_score(y, pred):.4f}  lab_acc={balanced_accuracy_score([LAB(classes[i]) for i in y], [LAB(classes[i]) for i in pred]):.4f}")

    oracle = assign_within_groups(proba, train["prompt_id"].values)
    print(f"blend + assign (true groups) bal_acc={balanced_accuracy_score(y, oracle):.4f}  (upper bound)")
    cl = cluster_by_topic(train["text"].tolist(), distance_threshold=0.92)
    pa = assign_within_groups(proba, cl)
    print(f"blend + assign (clustered)  bal_acc={balanced_accuracy_score(y, pa):.4f}  <- realistic")

    out = pd.DataFrame(proba, columns=classes)
    out.insert(0, "id", train["id"].values)
    out.to_parquet(ROOT / "outputs" / "oof_final_blend.parquet", index=False)


if __name__ == "__main__":
    main()
