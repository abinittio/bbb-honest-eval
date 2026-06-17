"""Model registry for the iterated robustness benchmark.

Every model exposes the same interface so the harness treats them identically:

    bench.fit(graphs, labels, smiles, train_idx, val_idx, seed)
    bench.predict_proba(graphs, smiles, idx) -> np.ndarray of P(BBB+)

Adding a model to the benchmark is one entry in REGISTRY. Two families are
supported: graph models (trained via study._fit_gnn) and fingerprint models
(ECFP + a scikit-learn classifier).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import study
from zinc_stereo_pretraining import StereoAwareEncoder
from bbb_stereo_v2 import BBBStereoV2Model


# --------------------------------------------------------------------------- #
# Baseline graph modules (return a single [B,1] logit)
# --------------------------------------------------------------------------- #
class GCNNet(nn.Module):
    def __init__(self, in_dim=21, hidden=128, layers=2):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_dim if i == 0 else hidden, hidden) for i in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
        return self.head(global_mean_pool(h, batch))


class GINNet(nn.Module):
    def __init__(self, in_dim=21, hidden=128, layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(layers):
            d = in_dim if i == 0 else hidden
            mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
        return self.head(global_mean_pool(h, batch))


# --------------------------------------------------------------------------- #
# Uniform benchmark wrappers
# --------------------------------------------------------------------------- #
class GNNBench:
    def __init__(self, name, factory, forward_logit):
        self.name = name
        self.factory = factory          # seed -> nn.Module
        self.fl = forward_logit
        self.model = None

    def fit(self, graphs, labels, smiles, tr, va, seed, max_epochs=120):
        study.set_seed(seed)
        module = self.factory(seed)
        self.model, _ = study._fit_gnn(module, self.fl, graphs, labels, tr, va,
                                       seed=seed, max_epochs=max_epochs)
        return self

    def predict_proba(self, graphs, smiles, idx):
        return study._predict_gnn(self.model, self.fl, graphs, idx)


class FPBench:
    def __init__(self, name, make_clf):
        self.name = name
        self.make_clf = make_clf        # seed -> sklearn classifier
        self.clf = None
        self._fps = None

    def _fp(self, smiles):
        if self._fps is None:           # compute ECFP once, reuse across runs
            self._fps = study._fingerprints(smiles)
        return self._fps

    def fit(self, graphs, labels, smiles, tr, va, seed, max_epochs=None):
        fps = self._fp(smiles)
        idx = np.concatenate([tr, va])  # classical models need no validation set
        self.clf = self.make_clf(seed)
        self.clf.fit(fps[idx], labels[idx])
        return self

    def predict_proba(self, graphs, smiles, idx):
        return self.clf.predict_proba(self._fp(smiles)[idx])[:, 1]


# --------------------------------------------------------------------------- #
# Registry: name -> zero-arg constructor of a bench model
# --------------------------------------------------------------------------- #
def _stereo_factory(seed):
    enc = StereoAwareEncoder(node_features=21, hidden_dim=128, num_layers=4)
    return BBBStereoV2Model(enc, hidden_dim=128)


REGISTRY = {
    "StereoGNN":   lambda: GNNBench("StereoGNN", _stereo_factory, study.stereo_forward),
    "GCN":         lambda: GNNBench("GCN", lambda s: GCNNet(), study.plain_forward),
    "GIN":         lambda: GNNBench("GIN", lambda s: GINNet(), study.plain_forward),
    "ECFP+LogReg": lambda: FPBench("ECFP+LogReg", lambda s: LogisticRegression(max_iter=1000)),
    "ECFP+RF":     lambda: FPBench("ECFP+RF",
                                   lambda s: RandomForestClassifier(n_estimators=300,
                                                                    random_state=s, n_jobs=-1)),
}
