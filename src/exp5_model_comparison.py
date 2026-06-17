"""Experiment 5: iterated robustness benchmark across a model panel.

Runs the identical honest-evaluation protocol (random / scaffold / cluster splits,
N seeds, matched splits) over every model in models.REGISTRY and reports a
model x split ROC-AUC matrix plus each model's random-minus-scaffold gap. This is
the multi-model generalisation of Exp 1: adding a model is one REGISTRY entry.

Usage:  python src/exp5_model_comparison.py [n_seeds] [max_epochs]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import study
import models

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MAX_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 120
SEEDS = list(range(N_SEEDS))
SPLITS = ["random", "scaffold", "cluster"]


def main():
    graphs, labels, smiles = study.load_bbbp()
    print(f"BBBP: {len(graphs)} compounds. Models={list(models.REGISTRY)} "
          f"Splits={SPLITS} Seeds={SEEDS}\n")

    rows = []
    t_start = time.time()
    for name, make in models.REGISTRY.items():
        bench = make()
        for split in SPLITS:
            for seed in SEEDS:
                tr, va, te = study.get_split(split, len(graphs), smiles, seed)
                t0 = time.time()
                bench.fit(graphs, labels, smiles, tr, va, seed, max_epochs=MAX_EPOCHS)
                p = bench.predict_proba(graphs, smiles, te)
                rows.append(dict(model=name, split=split, seed=seed,
                                 test_auc=roc_auc_score(labels[te], p),
                                 test_pr=average_precision_score(labels[te], p),
                                 secs=round(time.time() - t0, 1)))
                print(f"  {name:12s} {split:9s} seed{seed}: "
                      f"AUC={rows[-1]['test_auc']:.4f} ({rows[-1]['secs']}s)")
                sys.stdout.flush()

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(study.REPO, "results"), exist_ok=True)
    df.to_csv(os.path.join(study.REPO, "results", "exp5_model_comparison_raw.csv"), index=False)

    # model x split AUC matrix (mean +/- std); groupby+unstack keeps split
    # columns even when std is all-NaN (single-seed quick looks).
    g = df.groupby(["model", "split"])["test_auc"]
    mean = g.mean().unstack("split").reindex(list(models.REGISTRY))[SPLITS]
    std = g.std().unstack("split").reindex(list(models.REGISTRY))[SPLITS].fillna(0.0)
    matrix = mean.copy()
    for split in SPLITS:
        matrix[split] = [f"{m:.3f}+-{s:.3f}" for m, s in zip(mean[split], std[split])]
    matrix["random_minus_scaffold"] = (mean["random"] - mean["scaffold"]).round(4)
    matrix.to_csv(os.path.join(study.REPO, "results", "exp5_model_comparison_matrix.csv"))

    print("\n" + "=" * 72)
    print("MODEL x SPLIT  ROC-AUC (mean +/- std over seeds)")
    print("=" * 72)
    print(f"{'model':12s}" + "".join(f"{s:>16s}" for s in SPLITS) + f"{'rnd-scaf':>11s}")
    for name in models.REGISTRY:
        line = f"{name:12s}"
        for split in SPLITS:
            line += f"{mean.loc[name, split]:>9.3f}+-{std.loc[name, split]:>4.3f}"
        line += f"{mean.loc[name,'random']-mean.loc[name,'scaffold']:>+11.4f}"
        print(line)
    print(f"\nWall time: {(time.time()-t_start)/60:.1f} min")
    print("Saved -> results/exp5_model_comparison_{raw,matrix}.csv")


if __name__ == "__main__":
    main()
