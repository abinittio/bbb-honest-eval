"""Experiment 1: split-strategy study.

Train the identical architecture from the same random init under random, scaffold
(Bemis-Murcko), and cluster splits, across N seeds, and report test ROC-AUC and
PR-AUC (mean +/- std). The random-minus-scaffold gap is the headline quantity:
it measures how much reported performance depends on the evaluation split rather
than the model.

Usage:  python src/exp1_split_study.py [n_seeds] [max_epochs]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import study

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MAX_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 120
SEEDS = list(range(N_SEEDS))
SPLITS = ["random", "scaffold", "cluster"]


def make_split(kind, n, smiles, seed):
    if kind == "random":
        return study.random_split(n, seed)
    if kind == "scaffold":
        return study.scaffold_split(smiles, seed)
    if kind == "cluster":
        return study.cluster_split(smiles, seed)
    raise ValueError(kind)


def main():
    graphs, labels, smiles = study.load_bbbp()
    print(f"BBBP featurised: {len(graphs)} compounds "
          f"({int(labels.sum())} BBB+, {int(len(labels)-labels.sum())} BBB-)")
    print(f"Seeds={SEEDS}  max_epochs={MAX_EPOCHS}\n")

    rows = []
    t_start = time.time()
    for kind in SPLITS:
        for seed in SEEDS:
            tr, va, te = make_split(kind, len(graphs), smiles, seed)
            t0 = time.time()
            r = study.train_eval(graphs, labels, tr, va, te,
                                 seed=seed, max_epochs=MAX_EPOCHS)
            r.update(split=kind, seed=seed, secs=round(time.time() - t0, 1))
            rows.append(r)
            print(f"  {kind:9s} seed{seed}: test AUC={r['test_auc']:.4f}  "
                  f"PR={r['test_pr']:.4f}  (n_test={r['n_test']}, {r['secs']}s)")
            sys.stdout.flush()

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(study.REPO, "results"), exist_ok=True)
    df.to_csv(os.path.join(study.REPO, "results", "exp1_split_study_raw.csv"), index=False)

    summ = (df.groupby("split")
              .agg(auc_mean=("test_auc", "mean"), auc_std=("test_auc", "std"),
                   pr_mean=("test_pr", "mean"), pr_std=("test_pr", "std"))
              .reindex(SPLITS))
    summ.to_csv(os.path.join(study.REPO, "results", "exp1_split_study_summary.csv"))

    print("\n" + "=" * 60)
    print("SUMMARY (mean +/- std over seeds)")
    print("=" * 60)
    print(f"{'split':10s}{'ROC-AUC':>18s}{'PR-AUC':>18s}")
    for k in SPLITS:
        s = summ.loc[k]
        print(f"{k:10s}{s.auc_mean:>10.4f} +/-{s.auc_std:>5.4f}"
              f"{s.pr_mean:>10.4f} +/-{s.pr_std:>5.4f}")
    gap = summ.loc["random", "auc_mean"] - summ.loc["scaffold", "auc_mean"]
    print("-" * 46)
    print(f"random - scaffold AUC gap: {gap:+.4f}")
    print(f"\nTotal wall time: {(time.time()-t_start)/60:.1f} min")
    print("Saved -> results/exp1_split_study_{raw,summary}.csv")


if __name__ == "__main__":
    main()
