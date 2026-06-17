"""Experiment 3: ablation study.

Two questions, both pre-specified by the project's framing:

  A. Do the stereochemistry features earn their place, and does their value
     depend on the split? We compare the full 21-feature model against a
     no-stereo variant (the 6 stereo dims zeroed) under BOTH random and scaffold
     splits. If stereo helps under random but not scaffold, the feature inflates
     the optimistic metric without improving out-of-scaffold generalisation.

  B. Is the architecture over-parameterised for ~2,000 compounds? We sweep the
     encoder depth (4, 2, 1 layers) under the scaffold split.

Usage:  python src/exp3_ablation.py [n_seeds] [max_epochs]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import study

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MAX_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 120
SEEDS = list(range(N_SEEDS))


def run(graphs, labels, smiles, split, seed, **kw):
    tr, va, te = study.get_split(split, len(graphs), smiles, seed)
    r = study.train_eval(graphs, labels, tr, va, te, seed=seed,
                         max_epochs=MAX_EPOCHS, **kw)
    return r


def main():
    graphs, labels, smiles = study.load_bbbp()
    print(f"BBBP: {len(graphs)} compounds. Seeds={SEEDS}\n")
    rows = []
    t0 = time.time()

    # Block A: stereo features x split
    print("Block A: stereo features under random vs scaffold")
    for split in ["random", "scaffold"]:
        for feats, cols in [("full", None), ("no_stereo", study.STEREO_COLS)]:
            for seed in SEEDS:
                r = run(graphs, labels, smiles, split, seed, feat_zero_cols=cols)
                r.update(block="A", split=split, feats=feats, num_layers=4, seed=seed)
                rows.append(r)
                print(f"  {split:9s} {feats:9s} seed{seed}: AUC={r['test_auc']:.4f}")
                sys.stdout.flush()

    # Block B: encoder depth under scaffold (full features); reuse 4-layer from A
    print("\nBlock B: encoder depth under scaffold split")
    for nl in [2, 1]:
        for seed in SEEDS:
            r = run(graphs, labels, smiles, "scaffold", seed, num_layers=nl)
            r.update(block="B", split="scaffold", feats="full", num_layers=nl, seed=seed)
            rows.append(r)
            print(f"  layers={nl} seed{seed}: AUC={r['test_auc']:.4f}")
            sys.stdout.flush()

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(study.REPO, "results"), exist_ok=True)
    df.to_csv(os.path.join(study.REPO, "results", "exp3_ablation_raw.csv"), index=False)

    print("\n" + "=" * 56)
    print("A. STEREO EFFECT (full minus no_stereo AUC, mean over seeds)")
    print("=" * 56)
    a = df[df.block == "A"]
    for split in ["random", "scaffold"]:
        full = a[(a.split == split) & (a.feats == "full")].test_auc.mean()
        nost = a[(a.split == split) & (a.feats == "no_stereo")].test_auc.mean()
        print(f"  {split:9s}: full={full:.4f}  no_stereo={nost:.4f}  "
              f"stereo gain={full-nost:+.4f}")

    print("\nB. CAPACITY (scaffold split, full features, mean over seeds)")
    print("=" * 56)
    cap = pd.concat([
        a[(a.split == "scaffold") & (a.feats == "full")].assign(num_layers=4),
        df[df.block == "B"],
    ])
    for nl in [4, 2, 1]:
        m = cap[cap.num_layers == nl].test_auc
        print(f"  layers={nl}: AUC={m.mean():.4f} +/- {m.std():.4f}")
    print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
    print("Saved -> results/exp3_ablation_raw.csv")


if __name__ == "__main__":
    main()
