"""Shared utilities for the BBB evaluation study: featurisation cache, splitters,
and a fixed train/evaluate routine. Imported by the per-experiment scripts and by
the submission notebook so every experiment uses the identical pipeline.
"""
from __future__ import annotations
import os, sys, random, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.cluster import MiniBatchKMeans
from torch_geometric.data import Batch

from zinc_stereo_pretraining import StereoAwareEncoder
from bbb_stereo_v2 import BBBStereoV2Model
from mol_to_graph_enhanced import mol_to_graph_enhanced

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
CACHE = os.path.join(DATA, "bbbp_graphs.pt")


# --------------------------------------------------------------------------- #
# Data + featurisation (cached once, reused by every run)
# --------------------------------------------------------------------------- #
def load_bbbp():
    """Return (graphs, labels[np], smiles[list]) for BBBP, featurised once and cached."""
    if os.path.exists(CACHE):
        blob = torch.load(CACHE, weights_only=False)
        return blob["graphs"], blob["labels"], blob["smiles"]

    df = pd.read_csv(os.path.join(DATA, "BBBP.csv"))
    smi_col = "smiles" if "smiles" in df.columns else "SMILES"
    graphs, labels, smiles = [], [], []
    for smi, lab in zip(df[smi_col], df["p_np"]):
        g = mol_to_graph_enhanced(smi, y=float(lab), include_quantum=False,
                                  include_stereo=True, use_dft=False)
        if g is not None and g.x.shape[1] == 21:
            graphs.append(g); labels.append(int(lab)); smiles.append(smi)
    labels = np.array(labels)
    torch.save({"graphs": graphs, "labels": labels, "smiles": smiles}, CACHE)
    return graphs, labels, smiles


# --------------------------------------------------------------------------- #
# Splitters: all return (train_idx, val_idx, test_idx) as numpy arrays
# --------------------------------------------------------------------------- #
def random_split(n, seed, fracs=(0.7, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a = int(fracs[0] * n); b = a + int(fracs[1] * n)
    return idx[:a], idx[a:b], idx[b:]


def _group_split(groups, seed, fracs=(0.7, 0.15, 0.15)):
    """Assign whole groups (scaffold or cluster) to splits, so no group spans them."""
    groups = np.asarray(groups)
    uniq = list(pd.unique(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(groups)
    n_train, n_val = int(fracs[0] * n), int(fracs[1] * n)
    train, val, test = [], [], []
    by_group = {g: np.where(groups == g)[0] for g in uniq}
    for g in uniq:
        members = by_group[g]
        if len(train) < n_train:
            train.extend(members)
        elif len(val) < n_val:
            val.extend(members)
        else:
            test.extend(members)
    return np.array(train), np.array(val), np.array(test)


def _murcko(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles  # its own group
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return smiles


def scaffold_split(smiles, seed, fracs=(0.7, 0.15, 0.15)):
    return _group_split([_murcko(s) for s in smiles], seed, fracs)


def _fingerprints(smiles):
    fps = np.zeros((len(smiles), 1024), dtype=np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fps[i] = np.frombuffer(bytes(bv.ToBitString(), "ascii"), "u1") - ord("0")
    return fps


def cluster_split(smiles, seed, fracs=(0.7, 0.15, 0.15), n_clusters=100):
    fps = _fingerprints(smiles)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=3)
    return _group_split(km.fit_predict(fps), seed, fracs)


def get_split(kind, n, smiles, seed):
    if kind == "random":
        return random_split(n, seed)
    if kind == "scaffold":
        return scaffold_split(smiles, seed)
    if kind == "cluster":
        return cluster_split(smiles, seed)
    raise ValueError(kind)


STEREO_COLS = list(range(15, 21))  # the 6 stereochemistry node-feature dimensions


# --------------------------------------------------------------------------- #
# Train + evaluate one configuration (fresh model, fixed protocol)
# --------------------------------------------------------------------------- #
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_model(seed=0, hidden_dim=128, num_layers=4):
    set_seed(seed)
    enc = StereoAwareEncoder(node_features=21, hidden_dim=hidden_dim, num_layers=num_layers)
    model = BBBStereoV2Model(enc, hidden_dim=hidden_dim)
    return model


def _apply_mask(batch, zero_cols):
    """Return batch with the given node-feature columns zeroed (clone, no mutation)."""
    if zero_cols is None:
        return batch
    batch = batch.clone()
    batch.x[:, zero_cols] = 0.0
    return batch


def stereo_forward(module, b):
    """forward_logit for the StereoGNN, whose forward returns (logBB, prob)."""
    _logbb, prob = module(b.x, b.edge_index, b.batch)
    return prob


def plain_forward(module, b):
    """forward_logit for a baseline GNN whose forward returns a single logit."""
    return module(b.x, b.edge_index, b.batch)


def _predict_gnn(module, forward_logit, graphs, idx, bs=64, zero_cols=None):
    """P(BBB+) for any module exposing forward_logit(module, batch) -> logit."""
    module.eval()
    probs = []
    for s in range(0, len(idx), bs):
        b = _apply_mask(Batch.from_data_list([graphs[i] for i in idx[s:s + bs]]), zero_cols)
        with torch.no_grad():
            logit = forward_logit(module, b)
        probs.extend(torch.sigmoid(logit).cpu().numpy().flatten().tolist())
    return np.array(probs)


def _eval_gnn(module, forward_logit, graphs, idx, labels, bs=64, zero_cols=None):
    p = _predict_gnn(module, forward_logit, graphs, idx, bs, zero_cols)
    y = labels[idx]
    return roc_auc_score(y, p), average_precision_score(y, p), p


def _fit_gnn(module, forward_logit, graphs, labels, tr, va, seed=0, max_epochs=120,
             patience=15, bs=32, lr=1e-3, feat_zero_cols=None):
    """Model-agnostic trainer: train any GNN module, early-stop on val AUC, return
    the best-val module. forward_logit abstracts each model's output shape."""
    set_seed(seed)
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=1e-2)
    lossfn = nn.BCEWithLogitsLoss()
    best_val, best_state, wait = -1.0, None, 0
    for _ep in range(max_epochs):
        module.train()
        order = np.random.permutation(tr)
        for s in range(0, len(order), bs):
            chunk = order[s:s + bs]
            if len(chunk) < 2:      # BatchNorm needs >1 sample in train mode
                continue
            b = _apply_mask(Batch.from_data_list([graphs[i] for i in chunk]), feat_zero_cols)
            opt.zero_grad()
            logit = forward_logit(module, b)
            loss = lossfn(logit, b.y.view(-1, 1).float())
            loss.backward(); opt.step()
        val_auc, _, _ = _eval_gnn(module, forward_logit, graphs, va, labels, zero_cols=feat_zero_cols)
        if val_auc > best_val:
            best_val, wait = val_auc, 0
            best_state = {k: v.clone() for k, v in module.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        module.load_state_dict(best_state)
    return module, best_val


def train_eval(graphs, labels, tr, va, te, seed=0, max_epochs=120, patience=15,
               bs=32, lr=1e-3, hidden_dim=128, num_layers=4, feat_zero_cols=None,
               return_probs=False, return_model=False):
    """Train from scratch on tr, early-stop on val AUC, evaluate on te.

    feat_zero_cols: node-feature columns to zero (e.g. range(15,21) ablates stereo).
    hidden_dim / num_layers: capacity knobs for the over-parameterisation ablation.
    """
    set_seed(seed)
    model = build_model(seed=seed, hidden_dim=hidden_dim, num_layers=num_layers)
    model, best_val = _fit_gnn(model, stereo_forward, graphs, labels, tr, va,
                               seed=seed, max_epochs=max_epochs, patience=patience,
                               bs=bs, lr=lr, feat_zero_cols=feat_zero_cols)
    test_auc, test_pr, p = _eval_gnn(model, stereo_forward, graphs, te, labels,
                                     zero_cols=feat_zero_cols)
    out = {"test_auc": test_auc, "test_pr": test_pr, "val_auc": best_val,
           "n_train": len(tr), "n_test": len(te)}
    if return_model:
        return out, p, model
    return (out, p) if return_probs else out
