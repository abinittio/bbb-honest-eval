"""Baselines: ECFP fingerprint + classical ML, under the identical scaffold split.

A custom GNN's value should be judged against the cheap, strong baseline the
cheminformatics literature keeps finding competitive on small datasets: Morgan
(ECFP4) fingerprints with logistic regression and random forest. Evaluated under
the same scaffold split as the GNN so the comparison is like-for-like.

Usage:  python src/exp_baselines.py [n_seeds]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import study

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SEEDS = list(range(N_SEEDS))


def main():
    _graphs, labels, smiles = study.load_bbbp()
    fps = study._fingerprints(smiles)
    print(f"BBBP: {len(smiles)} compounds, ECFP4 1024-bit. Seeds={SEEDS}\n")

    rows = []
    for seed in SEEDS:
        tr, va, te = study.get_split("scaffold", len(smiles), smiles, seed)
        tr = np.concatenate([tr, va])  # classical models need no val set
        Xtr, ytr, Xte, yte = fps[tr], labels[tr], fps[te], labels[te]
        for name, clf in [
            ("ECFP+LogReg", LogisticRegression(max_iter=1000)),
            ("ECFP+RF", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
        ]:
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:, 1]
            rows.append(dict(model=name, seed=seed,
                             test_auc=roc_auc_score(yte, p),
                             test_pr=average_precision_score(yte, p)))
        print(f"  seed{seed} done")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(study.REPO, "results"), exist_ok=True)
    df.to_csv(os.path.join(study.REPO, "results", "baselines_scaffold.csv"), index=False)
    print("\n" + "=" * 48)
    print(f"{'baseline (scaffold split)':26s}{'ROC-AUC':>11s}{'PR-AUC':>9s}")
    print("=" * 48)
    for name in ["ECFP+LogReg", "ECFP+RF"]:
        m = df[df.model == name]
        print(f"{name:26s}{m.test_auc.mean():>7.4f}+-{m.test_auc.std():.3f}"
              f"{m.test_pr.mean():>7.3f}")
    print("\nSaved -> results/baselines_scaffold.csv")


if __name__ == "__main__":
    main()
