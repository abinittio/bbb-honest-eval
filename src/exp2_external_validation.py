"""Experiment 2: external validation on B3DB, naive vs InChIKey-deduplicated.

The pretrained 5-fold ensemble was trained on BBBP. B3DB (Meng et al. 2021) is a
50-source aggregation that absorbs the BBBP data, so a naive "external" test on
B3DB scores partly on compounds the model already trained on. This script
quantifies that leakage: it reports the ensemble ROC-AUC on all of B3DB, then on
B3DB with every compound whose InChIKey appears in BBBP removed.

Run from the repo root:  python src/exp2_external_validation.py
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from torch_geometric.data import Batch

from zinc_stereo_pretraining import StereoAwareEncoder
from bbb_stereo_v2 import BBBStereoV2Model
from mol_to_graph_enhanced import mol_to_graph_enhanced

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
MODELS = os.path.join(REPO, "models")


def inchikey(smiles: str) -> str | None:
    """Full standard InChIKey; None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def load_fold(path: str) -> BBBStereoV2Model:
    enc = StereoAwareEncoder(node_features=21, hidden_dim=128, num_layers=4)
    model = BBBStereoV2Model(enc, hidden_dim=128)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    return model


def featurise(df: pd.DataFrame):
    """Return (graphs, labels, inchikeys) for rows that featurise to 21-dim graphs."""
    graphs, labels, keys = [], [], []
    failed = 0
    for i, row in enumerate(df.itertuples(index=False)):
        smi, lab = row.SMILES, row.label
        g = mol_to_graph_enhanced(smi, y=float(lab), include_quantum=False,
                                  include_stereo=True, use_dft=False)
        if g is not None and g.x.shape[1] == 21:
            graphs.append(g); labels.append(float(lab)); keys.append(inchikey(smi))
        else:
            failed += 1
        if (i + 1) % 1500 == 0:
            print(f"  featurised {i+1}/{len(df)} ({len(graphs)} ok, {failed} failed)")
            sys.stdout.flush()
    return graphs, np.array(labels), keys


def ensemble_probs(models, graphs, batch_size=64):
    """Mean P(BBB+) across folds for every graph."""
    fold_preds = []
    for m in models:
        preds = []
        for s in range(0, len(graphs), batch_size):
            b = Batch.from_data_list(graphs[s:s + batch_size])
            with torch.no_grad():
                _logbb, prob = m(b.x, b.edge_index, b.batch)
            preds.extend(torch.sigmoid(prob).cpu().numpy().flatten().tolist())
        fold_preds.append(np.array(preds))
    return np.mean(fold_preds, axis=0)


def metrics(y, p):
    yb = (p > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yb).ravel()
    return {
        "n": len(y), "pos": int(y.sum()), "neg": int(len(y) - y.sum()),
        "auc": roc_auc_score(y, p), "acc": accuracy_score(y, yb),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
    }


def main():
    print("=" * 64)
    print("EXPERIMENT 2: external validation, naive vs deduplicated")
    print("=" * 64)

    # BBBP training InChIKeys
    bbbp = pd.read_csv(os.path.join(DATA, "BBBP.csv"))
    smi_col = "smiles" if "smiles" in bbbp.columns else "SMILES"
    bbbp_keys = {k for k in (inchikey(s) for s in bbbp[smi_col]) if k}
    print(f"BBBP: {len(bbbp)} rows, {len(bbbp_keys)} unique parseable InChIKeys")

    # B3DB external set
    b3 = pd.read_csv(os.path.join(DATA, "B3DB_classification.tsv"), sep="\t")
    b3 = b3.rename(columns={"BBB+/BBB-": "cls"})
    b3["label"] = (b3["cls"] == "BBB+").astype(int)
    b3 = b3[["SMILES", "label"]].dropna()
    print(f"B3DB: {len(b3)} rows ({int(b3.label.sum())} BBB+, {int((1-b3.label).sum())} BBB-)")

    print("\nFeaturising B3DB (this takes a couple of minutes)...")
    graphs, labels, keys = featurise(b3)
    print(f"Featurised {len(graphs)} compounds.")

    print("\nLoading 5 folds + scoring ensemble...")
    models = [load_fold(os.path.join(MODELS, f"bbb_stereo_v2_fold{k}_best.pth")) for k in range(1, 6)]
    probs = ensemble_probs(models, graphs)

    # Overlap mask: B3DB compound whose InChIKey is in BBBP training
    in_bbbp = np.array([(k is not None and k in bbbp_keys) for k in keys])
    n_overlap = int(in_bbbp.sum())

    naive = metrics(labels, probs)
    dedup = metrics(labels[~in_bbbp], probs[~in_bbbp])

    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"B3DB featurised:            {naive['n']}")
    print(f"Overlap with BBBP (removed): {n_overlap} "
          f"({100*n_overlap/naive['n']:.1f}% of the 'external' set)")
    print()
    hdr = f"{'set':<26}{'N':>6}{'AUC':>9}{'Acc':>8}{'Sens':>8}{'Spec':>8}"
    print(hdr); print("-" * len(hdr))
    print(f"{'naive (all B3DB)':<26}{naive['n']:>6}{naive['auc']:>9.4f}"
          f"{naive['acc']:>8.3f}{naive['sensitivity']:>8.3f}{naive['specificity']:>8.3f}")
    print(f"{'deduplicated (no overlap)':<26}{dedup['n']:>6}{dedup['auc']:>9.4f}"
          f"{dedup['acc']:>8.3f}{dedup['sensitivity']:>8.3f}{dedup['specificity']:>8.3f}")
    print("-" * len(hdr))
    print(f"AUC drop from removing train/test overlap: "
          f"{naive['auc'] - dedup['auc']:+.4f}")

    os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
    pd.DataFrame([dict(set="naive", **naive), dict(set="dedup", **dedup)]).to_csv(
        os.path.join(REPO, "results", "exp2_external_validation.csv"), index=False)
    print("\nSaved -> results/exp2_external_validation.csv")


if __name__ == "__main__":
    main()
