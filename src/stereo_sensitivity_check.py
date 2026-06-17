"""Diagnostic: does the deployed BBB model actually respond to chirality?

For each chiral drug we predict both enantiomers (R/S) with the 5-fold ensemble
and report |P(R) - P(S)|. If the differences are ~0 the model is functionally
stereo-blind on BBB; if they are non-trivial it uses stereo, just not in a way
that improves BBB accuracy (per the ablation).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from torch_geometric.data import Batch
from zinc_stereo_pretraining import StereoAwareEncoder
from bbb_stereo_v2 import BBBStereoV2Model
from mol_to_graph_enhanced import mol_to_graph_enhanced

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(REPO, "models")

PAIRS = {
    "amphetamine":  ("C[C@H](N)Cc1ccccc1", "C[C@@H](N)Cc1ccccc1"),
    "DOPA":         ("N[C@@H](Cc1ccc(O)c(O)c1)C(O)=O", "N[C@H](Cc1ccc(O)c(O)c1)C(O)=O"),
    "ibuprofen":    ("CC(C)Cc1ccc(cc1)[C@H](C)C(O)=O", "CC(C)Cc1ccc(cc1)[C@@H](C)C(O)=O"),
    "propranolol":  ("CC(C)NC[C@H](O)COc1cccc2ccccc12", "CC(C)NC[C@@H](O)COc1cccc2ccccc12"),
    "methamphetamine": ("CN[C@H](C)Cc1ccccc1", "CN[C@@H](C)Cc1ccccc1"),
}


def load_folds():
    folds = []
    for k in range(1, 6):
        enc = StereoAwareEncoder(node_features=21, hidden_dim=128, num_layers=4)
        m = BBBStereoV2Model(enc, hidden_dim=128)
        sd = torch.load(os.path.join(MODELS, f"bbb_stereo_v2_fold{k}_best.pth"),
                        map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        m.load_state_dict(sd); m.eval(); folds.append(m)
    return folds


def predict(folds, smiles):
    g = mol_to_graph_enhanced(smiles, y=0.0, include_quantum=False,
                              include_stereo=True, use_dft=False)
    if g is None:
        return None
    b = Batch.from_data_list([g])
    ps = []
    for m in folds:
        with torch.no_grad():
            _logbb, prob = m(b.x, b.edge_index, b.batch)
        ps.append(float(torch.sigmoid(prob)))
    return float(np.mean(ps))


def main():
    folds = load_folds()
    print(f"{'compound':16s}{'P(R/S-1)':>10s}{'P(R/S-2)':>10s}{'|diff|':>9s}")
    print("-" * 45)
    diffs = []
    for name, (a, b) in PAIRS.items():
        pa, pb = predict(folds, a), predict(folds, b)
        if pa is None or pb is None:
            print(f"{name:16s}  unfeaturisable"); continue
        d = abs(pa - pb); diffs.append(d)
        print(f"{name:16s}{pa:>10.4f}{pb:>10.4f}{d:>9.4f}")
    print("-" * 45)
    print(f"mean |P(enantiomer-1) - P(enantiomer-2)| = {np.mean(diffs):.4f}")
    print("(~0 => functionally stereo-blind on BBB; >0 => responds to chirality)")


if __name__ == "__main__":
    main()
