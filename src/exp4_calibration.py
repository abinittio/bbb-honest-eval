"""Experiment 4: calibration.

Exp 2 showed the model ranks well (high AUC) but over-predicts BBB+ (specificity
~0.66). AUC is blind to that; calibration is not. We train one model under the
scaffold split, then ask whether its probabilities are trustworthy and whether
post-hoc scaling fixes them. We report expected calibration error (ECE) and Brier
score before scaling, after temperature scaling, and after Platt scaling, with
reliability-diagram data saved for plotting in the notebook.

Usage:  python src/exp4_calibration.py [seed] [max_epochs]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import Batch
import study

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 120


def logits_for(model, graphs, idx, bs=64):
    model.eval(); out = []
    for s in range(0, len(idx), bs):
        b = Batch.from_data_list([graphs[i] for i in idx[s:s + bs]])
        with torch.no_grad():
            _logbb, prob = model(b.x, b.edge_index, b.batch)
        out.extend(prob.cpu().numpy().flatten().tolist())
    return np.array(out)


def ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    e, N = 0.0, len(labels)
    rows = []
    for i in range(n_bins):
        m = (probs > bins[i]) & (probs <= bins[i + 1])
        if m.sum() == 0:
            rows.append((0.5 * (bins[i] + bins[i + 1]), np.nan, np.nan, 0)); continue
        conf, acc, w = probs[m].mean(), labels[m].mean(), m.sum()
        e += w / N * abs(conf - acc)
        rows.append((0.5 * (bins[i] + bins[i + 1]), conf, acc, int(w)))
    return e, rows


def brier(probs, labels):
    return float(np.mean((probs - labels) ** 2))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    graphs, labels, smiles = study.load_bbbp()
    tr, va, te = study.get_split("scaffold", len(graphs), smiles, SEED)
    print(f"Scaffold split (seed {SEED}): train {len(tr)}, val {len(va)}, test {len(te)}")

    t0 = time.time()
    out, _p, model = study.train_eval(graphs, labels, tr, va, te, seed=SEED,
                                      max_epochs=MAX_EPOCHS, return_model=True)
    print(f"Trained. test AUC={out['test_auc']:.4f}  ({(time.time()-t0)/60:.1f} min)")

    val_logits, test_logits = logits_for(model, graphs, va), logits_for(model, graphs, te)
    y_test = labels[te].astype(float); y_val = labels[va].astype(float)

    # uncalibrated
    p_raw = sigmoid(test_logits)
    # temperature scaling: fit T on val NLL
    def nll(T):
        p = np.clip(sigmoid(val_logits / T), 1e-6, 1 - 1e-6)
        return -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
    T = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x
    p_temp = sigmoid(test_logits / T)
    # Platt scaling: 1D logistic on val logits
    platt = LogisticRegression().fit(val_logits.reshape(-1, 1), y_val)
    p_platt = platt.predict_proba(test_logits.reshape(-1, 1))[:, 1]

    print("\n" + "=" * 52)
    print(f"{'method':16s}{'ECE':>10s}{'Brier':>10s}")
    print("=" * 52)
    rows_out = {}
    for name, p in [("uncalibrated", p_raw), ("temperature", p_temp), ("platt", p_platt)]:
        e, rel = ece(p, y_test)
        b = brier(p, y_test)
        rows_out[name] = rel
        print(f"{name:16s}{e:>10.4f}{b:>10.4f}")
    print("-" * 52)
    print(f"fitted temperature T = {T:.3f}  (T>1 => model was over-confident)")

    os.makedirs(os.path.join(study.REPO, "results"), exist_ok=True)
    rel_df = []
    for name, rel in rows_out.items():
        for binc, conf, acc, w in rel:
            rel_df.append(dict(method=name, bin_center=binc, confidence=conf,
                               accuracy=acc, count=w))
    pd.DataFrame(rel_df).to_csv(
        os.path.join(study.REPO, "results", "exp4_reliability.csv"), index=False)
    print("Saved -> results/exp4_reliability.csv")


if __name__ == "__main__":
    main()
