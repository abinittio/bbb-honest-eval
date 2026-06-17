"""
ZINC Pretraining with Stereoisomer Enumeration

Key insight: Molecules with stereocenters should be treated as MULTIPLE molecules
during pretraining - one for each stereoisomer (E/Z, R/S configurations).

This teaches the model that stereochemistry matters for molecular properties.

For ZINC (unlabeled pretraining):
- Enumerate all stereoisomers of each molecule
- Each stereoisomer is a separate training example
- Self-supervised learning on molecular structure

For BBB (labeled fine-tuning):
- If ALL stereoisomers are BBB+ → BBB+
- If ANY stereoisomer is BBB- → needs careful handling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, TransformerConv, global_mean_pool, global_max_pool
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
import sys
import os
from typing import List, Tuple, Optional

# Import our graph conversion with stereo features
from mol_to_graph_enhanced import batch_smiles_to_graphs_enhanced, mol_to_graph_enhanced


def enumerate_stereoisomers(smiles: str, max_isomers: int = 8) -> List[str]:
    """
    Enumerate all stereoisomers of a molecule.

    Args:
        smiles: Input SMILES string
        max_isomers: Maximum number of isomers to generate (to avoid combinatorial explosion)

    Returns:
        List of SMILES strings for each stereoisomer
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]  # Return original if can't parse

    # Configure enumeration options
    opts = StereoEnumerationOptions(
        tryEmbedding=False,  # Don't require 3D embedding
        unique=True,         # Only unique isomers
        maxIsomers=max_isomers,
        onlyUnassigned=False  # Enumerate all, not just unassigned
    )

    try:
        isomers = list(EnumerateStereoisomers(mol, options=opts))
        if len(isomers) == 0:
            return [smiles]

        isomer_smiles = []
        for isomer in isomers:
            try:
                iso_smiles = Chem.MolToSmiles(isomer, isomericSmiles=True)
                isomer_smiles.append(iso_smiles)
            except:
                continue

        return isomer_smiles if isomer_smiles else [smiles]
    except:
        return [smiles]


def count_stereocenters(smiles: str) -> Tuple[int, int]:
    """
    Count chiral centers and E/Z double bonds in a molecule.

    Returns:
        (num_chiral_centers, num_ez_bonds)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (0, 0)

    # Count chiral centers
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    num_chiral = len(chiral_centers)

    # Count E/Z double bonds
    num_ez = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            stereo = bond.GetStereo()
            if stereo in [Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
                          Chem.BondStereo.STEREOANY]:
                num_ez += 1

    return (num_chiral, num_ez)


def expand_zinc_with_stereoisomers(zinc_smiles: List[str],
                                    max_isomers_per_mol: int = 4,
                                    verbose: bool = True) -> List[str]:
    """
    Expand ZINC dataset by enumerating stereoisomers.

    Args:
        zinc_smiles: List of SMILES from ZINC
        max_isomers_per_mol: Max stereoisomers per molecule
        verbose: Print progress

    Returns:
        Expanded list of SMILES with stereoisomers
    """
    expanded_smiles = []
    stereo_count = 0
    total_isomers = 0

    for i, smiles in enumerate(zinc_smiles):
        isomers = enumerate_stereoisomers(smiles, max_isomers=max_isomers_per_mol)
        expanded_smiles.extend(isomers)

        if len(isomers) > 1:
            stereo_count += 1
            total_isomers += len(isomers)

        if verbose and (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{len(zinc_smiles)} molecules, "
                  f"{stereo_count} with stereo ({total_isomers} isomers generated)")
            sys.stdout.flush()

    if verbose:
        print(f"\nStereoisomer expansion complete:")
        print(f"  Original molecules: {len(zinc_smiles)}")
        print(f"  Molecules with stereocenters: {stereo_count} ({100*stereo_count/len(zinc_smiles):.1f}%)")
        print(f"  Total after expansion: {len(expanded_smiles)}")
        print(f"  Expansion factor: {len(expanded_smiles)/len(zinc_smiles):.2f}x")
        sys.stdout.flush()

    return expanded_smiles


class StereoAwareEncoder(nn.Module):
    """
    GNN encoder that handles 21 features (15 atomic + 6 stereo).
    Used for self-supervised pretraining on ZINC with stereoisomers.
    Now with edge feature support (7 bond features).
    """
    def __init__(self, node_features=21, hidden_dim=128, num_layers=4, dropout=0.2, edge_features=7):
        super().__init__()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.edge_features = edge_features

        # Initial node embedding
        self.input_embed = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Edge embedding (bond features → hidden dim for attention)
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)  # Match GAT head dim
        )

        # GAT layers with edge_dim support
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout,
                          concat=True, edge_dim=hidden_dim // 4)
            )
            self.gat_norms.append(nn.BatchNorm1d(hidden_dim))

        # Transformer layer
        self.transformer = TransformerConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.transformer_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # Initial embedding
        x = self.input_embed(x)

        # Embed edge features if provided
        if edge_attr is not None and edge_attr.size(0) > 0:
            edge_embed = self.edge_embed(edge_attr)
        else:
            edge_embed = None

        # GAT layers with residuals and edge features
        for gat, norm in zip(self.gat_layers, self.gat_norms):
            x_new = gat(x, edge_index, edge_attr=edge_embed)
            x_new = norm(x_new)
            x_new = torch.nn.functional.relu(x_new)
            x = x + x_new

        # Transformer
        x_trans = self.transformer(x, edge_index)
        x_trans = self.transformer_norm(x_trans)
        x = x + x_trans

        # Graph-level representation
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        return torch.cat([x_mean, x_max], dim=1)  # [batch, hidden_dim * 2]


class StereoPretrainingModel(nn.Module):
    """
    Self-supervised pretraining model for ZINC with stereoisomers.

    Pretraining task: Predict molecular properties from structure
    - Predicts normalized molecular weight
    - Predicts number of atoms
    - Predicts has_stereocenters (binary)
    """
    def __init__(self, node_features=21, hidden_dim=128, num_layers=4, dropout=0.2):
        super().__init__()

        self.encoder = StereoAwareEncoder(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # Prediction heads
        self.mol_weight_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.atom_count_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.stereo_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, batch):
        # Get graph embedding
        graph_embed = self.encoder(x, edge_index, batch)

        # Predict properties
        mol_weight = self.mol_weight_head(graph_embed)
        atom_count = self.atom_count_head(graph_embed)
        has_stereo = self.stereo_head(graph_embed)

        return mol_weight, atom_count, has_stereo

    def get_encoder(self):
        return self.encoder


class StereoAwareBBBNet(nn.Module):
    """
    BBB prediction model with stereo-aware encoder (21 features).
    Can load pretrained weights from StereoPretrainingModel.
    """
    def __init__(self, node_features=21, hidden_dim=128, num_layers=4, dropout=0.2):
        super().__init__()

        self.encoder = StereoAwareEncoder(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # BBB prediction head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, batch):
        graph_embed = self.encoder(x, edge_index, batch)
        return self.classifier(graph_embed)

    def load_pretrained_encoder(self, pretrained_model_path: str, device='cpu'):
        """Load encoder weights from pretrained model."""
        checkpoint = torch.load(pretrained_model_path, map_location=device)

        # Extract encoder weights
        encoder_state = {}
        for key, value in checkpoint.items():
            if key.startswith('encoder.'):
                encoder_state[key.replace('encoder.', '')] = value

        # Load into our encoder
        self.encoder.load_state_dict(encoder_state)
        print(f"Loaded {len(encoder_state)} encoder layers from {pretrained_model_path}")


def pretrain_on_zinc_stereo(zinc_smiles: List[str],
                            epochs: int = 30,
                            batch_size: int = 64,
                            lr: float = 0.001,
                            device: str = 'cpu',
                            save_path: str = 'models/pretrained_stereo_encoder.pth'):
    """
    Pretrain encoder on ZINC with stereoisomer expansion.
    """
    print("=" * 70)
    print("ZINC PRETRAINING WITH STEREOISOMER EXPANSION")
    print("=" * 70)
    sys.stdout.flush()

    # Expand ZINC with stereoisomers
    print("\nStep 1: Expanding ZINC with stereoisomers...")
    expanded_smiles = expand_zinc_with_stereoisomers(zinc_smiles, max_isomers_per_mol=4)

    # Convert to graphs with stereo features (21 features)
    print("\nStep 2: Converting to graphs with stereo features...")
    sys.stdout.flush()

    graphs = batch_smiles_to_graphs_enhanced(
        expanded_smiles,
        y_list=None,  # No labels for pretraining
        include_quantum=False,  # No quantum for now
        include_stereo=True,    # Yes stereo
        use_dft=False,
        verbose=True
    )

    # Add self-supervised labels
    print("\nStep 3: Computing self-supervised targets...")
    for i, (smiles, graph) in enumerate(zip(expanded_smiles, graphs)):
        if graph is None:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Target 1: Normalized molecular weight [0, 1]
        mol_weight = Chem.Descriptors.MolWt(mol)
        graph.mol_weight = torch.tensor([mol_weight / 500.0], dtype=torch.float)

        # Target 2: Normalized atom count [0, 1]
        atom_count = mol.GetNumAtoms()
        graph.atom_count = torch.tensor([atom_count / 50.0], dtype=torch.float)

        # Target 3: Has stereocenters (binary)
        chiral, ez = count_stereocenters(smiles)
        has_stereo = 1.0 if (chiral > 0 or ez > 0) else 0.0
        graph.has_stereo = torch.tensor([has_stereo], dtype=torch.float)

    # Filter out None graphs
    graphs = [g for g in graphs if g is not None and hasattr(g, 'mol_weight')]
    print(f"Valid graphs for pretraining: {len(graphs)}")

    # Create data loader
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = StereoPretrainingModel(node_features=21).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    print(f"\nStep 4: Training for {epochs} epochs...")
    print("=" * 60)
    sys.stdout.flush()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            pred_mw, pred_ac, pred_stereo = model(batch.x, batch.edge_index, batch.batch)

            # Compute losses
            loss_mw = mse_loss(pred_mw.view(-1), batch.mol_weight.view(-1))
            loss_ac = mse_loss(pred_ac.view(-1), batch.atom_count.view(-1))
            loss_stereo = bce_loss(pred_stereo.view(-1), batch.has_stereo.view(-1))

            loss = loss_mw + loss_ac + 0.5 * loss_stereo

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / num_batches

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            sys.stdout.flush()

    # Save pretrained model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nPretrained model saved to {save_path}")

    # Also save just the encoder for easy loading
    encoder_path = save_path.replace('.pth', '_encoder_only.pth')
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f"Encoder weights saved to {encoder_path}")

    return model


def load_zinc_subset(num_molecules: int = 50000) -> List[str]:
    """Load a subset of ZINC for pretraining."""
    from datasets import load_dataset

    print(f"Loading {num_molecules} molecules from ZINC...")
    sys.stdout.flush()

    try:
        dataset = load_dataset("zpn/zinc20", split="train", streaming=True, trust_remote_code=True)
        smiles_list = []

        for i, item in enumerate(dataset):
            if i >= num_molecules:
                break
            smiles_list.append(item['smiles'])

            if (i + 1) % 10000 == 0:
                print(f"  Loaded {i+1}/{num_molecules}")
                sys.stdout.flush()

        print(f"Loaded {len(smiles_list)} ZINC molecules")
        return smiles_list

    except Exception as e:
        print(f"Error loading ZINC: {e}")
        print("Falling back to local ZINC file if available...")

        # Try local file
        local_path = "data/zinc_smiles.txt"
        if os.path.exists(local_path):
            with open(local_path, 'r') as f:
                smiles_list = [line.strip() for line in f][:num_molecules]
            print(f"Loaded {len(smiles_list)} from local file")
            return smiles_list

        return []


if __name__ == "__main__":
    # Test stereoisomer enumeration
    print("Testing stereoisomer enumeration...")

    test_smiles = [
        "CC=CC",           # E/Z isomers possible
        "C[C@H](O)CC",     # Chiral center
        "CC(C)=CC=C(C)C",  # Multiple E/Z possible
        "CCO",             # No stereocenters
    ]

    for smiles in test_smiles:
        isomers = enumerate_stereoisomers(smiles)
        chiral, ez = count_stereocenters(smiles)
        print(f"{smiles}: {len(isomers)} isomers (chiral={chiral}, E/Z={ez})")
        for iso in isomers[:4]:
            print(f"  -> {iso}")

    print("\n" + "=" * 70)
    print("To run full pretraining, call:")
    print("  python zinc_stereo_pretraining.py --pretrain")
    print("=" * 70)
