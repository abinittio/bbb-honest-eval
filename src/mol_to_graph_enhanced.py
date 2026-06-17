"""
Enhanced Molecule-to-Graph Conversion with Real DFT + E-Z Isomers

This version integrates:
1. Real DFT quantum descriptors from PubChemQC B3LYP/6-31G* (86M molecules)
2. E-Z isomer (stereochemistry) encoding for geometric isomers
3. Falls back to RDKit approximations when DFT data unavailable

Total features per node: 34 (15 atomic + 13 quantum + 6 stereochemistry)
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from torch_geometric.data import Data
from typing import Optional, Dict, List

# Import our quantum and stereochemistry modules
from quantum_descriptors import QuantumDescriptorCalculator
from pubchemqc_integration import PubChemQCIntegration, StereochemistryEncoder

# Initialize calculators (once)
RDKIT_QUANTUM_CALC = QuantumDescriptorCalculator()
PUBCHEMQC = PubChemQCIntegration()
STEREO_ENCODER = StereochemistryEncoder()

# Electronegativity values (Pauling scale)
ELECTRONEGATIVITY = {
    1: 2.20,   # H
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    35: 2.96,  # Br
    53: 2.66,  # I
}

POLAR_ATOMS = {7, 8, 15, 16}  # N, O, P, S

# Bond feature dimensions
BOND_TYPES = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}

BOND_STEREO = {
    Chem.BondStereo.STEREONONE: 0,
    Chem.BondStereo.STEREOANY: 1,
    Chem.BondStereo.STEREOZ: 2,
    Chem.BondStereo.STEREOE: 3,
    Chem.BondStereo.STEREOCIS: 4,
    Chem.BondStereo.STEREOTRANS: 5,
}


def get_bond_features(bond) -> List[float]:
    """
    Extract features for a single bond (7 features):
    - Bond type one-hot (4): single, double, triple, aromatic
    - Is conjugated (1)
    - Is in ring (1)
    - Stereo type (1): normalized 0-1
    """
    features = []

    # Bond type one-hot (4 features)
    bond_type = BOND_TYPES.get(bond.GetBondType(), 0)
    bond_onehot = [0.0, 0.0, 0.0, 0.0]
    bond_onehot[bond_type] = 1.0
    features.extend(bond_onehot)

    # Is conjugated
    features.append(1.0 if bond.GetIsConjugated() else 0.0)

    # Is in ring
    features.append(1.0 if bond.IsInRing() else 0.0)

    # Stereo type normalized
    stereo = BOND_STEREO.get(bond.GetStereo(), 0)
    features.append(stereo / 5.0)  # Normalize to 0-1

    return features


def get_atom_features(atom) -> List[float]:
    """
    Extract comprehensive features for a single atom (15 features)
    """
    features = []

    # === BASIC FEATURES (1-9) ===
    features.append(atom.GetAtomicNum() / 100.0)
    features.append(atom.GetDegree())
    features.append(atom.GetFormalCharge())

    hybridization_map = {
        Chem.HybridizationType.S: 0,
        Chem.HybridizationType.SP: 1,
        Chem.HybridizationType.SP2: 2,
        Chem.HybridizationType.SP3: 3,
        Chem.HybridizationType.SP3D: 4,
        Chem.HybridizationType.SP3D2: 5,
    }
    features.append(hybridization_map.get(atom.GetHybridization(), 0))

    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(1 if atom.IsInRing() else 0)
    features.append(atom.GetTotalValence() - atom.GetTotalDegree())
    features.append(atom.GetTotalValence())
    features.append(atom.GetMass() / 200.0)

    # === POLARITY FEATURES (10-15) ===
    atomic_num = atom.GetAtomicNum()

    electronegativity = ELECTRONEGATIVITY.get(atomic_num, 2.5)
    features.append(electronegativity / 4.0)

    is_polar = 1 if atomic_num in POLAR_ATOMS else 0
    features.append(is_polar)

    # H-bond donor
    is_h_donor = 0
    if atomic_num in [7, 8]:
        if atom.GetTotalNumHs() > 0:
            is_h_donor = 1
    features.append(is_h_donor)

    # H-bond acceptor
    is_h_acceptor = 0
    if atomic_num == 7:
        if atom.GetDegree() < 4 and atom.GetFormalCharge() <= 0:
            is_h_acceptor = 1
    elif atomic_num == 8:
        if atom.GetFormalCharge() <= 0:
            is_h_acceptor = 1
    features.append(is_h_acceptor)

    # Partial charge approximation
    c_en = 2.55
    charge_approx = (electronegativity - c_en) / 2.0
    features.append(charge_approx)

    # In polar functional group
    in_polar_group = 0
    if atomic_num in POLAR_ATOMS:
        for neighbor in atom.GetNeighbors():
            neighbor_num = neighbor.GetAtomicNum()
            if neighbor_num in POLAR_ATOMS or neighbor_num == 6:
                bond = atom.GetOwningMol().GetBondBetweenAtoms(
                    atom.GetIdx(), neighbor.GetIdx()
                )
                if bond and bond.GetBondTypeAsDouble() >= 2.0:
                    in_polar_group = 1
                    break
        if atom.GetTotalNumHs() > 0:
            in_polar_group = 1
    features.append(in_polar_group)

    return features


def get_quantum_features(smiles: str, use_dft: bool = True) -> np.ndarray:
    """
    Get quantum descriptors - prefers real DFT from PubChemQC, falls back to RDKit.

    Returns 13 features:
    0: HOMO (eV)
    1: LUMO (eV)
    2: HOMO-LUMO gap (eV)
    3: Ionization potential (eV)
    4: Electron affinity (eV)
    5: Electronegativity (Mulliken) (eV)
    6: Chemical hardness (eV)
    7: Chemical softness
    8: Electrophilicity index
    9: Dipole moment (Debye)
    10: Polarizability
    11: Max partial charge
    12: Min partial charge
    """
    quantum_vec = np.zeros(13, dtype=np.float32)
    source = "fallback"

    if use_dft:
        # Try to get real DFT data from PubChemQC
        dft_data = PUBCHEMQC.get_quantum_descriptors(smiles)

        if dft_data is not None:
            source = dft_data.get('source', 'pubchemqc')

            # Extract DFT values
            quantum_vec[0] = dft_data.get('homo_ev', 0)
            quantum_vec[1] = dft_data.get('lumo_ev', 0)
            quantum_vec[2] = dft_data.get('gap_ev', 0)
            quantum_vec[3] = dft_data.get('ionization_potential', 0)
            quantum_vec[4] = dft_data.get('electron_affinity', 0)
            quantum_vec[5] = dft_data.get('electronegativity', 0)
            quantum_vec[6] = dft_data.get('chemical_hardness', 0)
            quantum_vec[7] = dft_data.get('softness', 0)
            quantum_vec[8] = dft_data.get('electrophilicity', 0)
            quantum_vec[9] = dft_data.get('dipole_moment', 0)
            quantum_vec[10] = dft_data.get('polarizability', 0)
            quantum_vec[11] = dft_data.get('max_charge', 0)
            quantum_vec[12] = dft_data.get('min_charge', 0)

            return quantum_vec, source

    # Fall back to RDKit approximations
    rdkit_vec = RDKIT_QUANTUM_CALC.calculate_vector(smiles)
    if rdkit_vec is not None:
        quantum_vec = np.array(rdkit_vec, dtype=np.float32)
        source = "rdkit_approx"

    return quantum_vec, source


def get_stereo_features(smiles: str) -> np.ndarray:
    """
    Get stereochemistry features (6 features):
    0: has_ez_centers (0 or 1)
    1: e_fraction (0-1)
    2: z_fraction (0-1)
    3: has_chiral_centers (0 or 1)
    4: r_fraction (0-1)
    5: s_fraction (0-1)
    """
    stereo_vec = STEREO_ENCODER.get_stereo_feature_vector(smiles)
    return np.array(stereo_vec[:6], dtype=np.float32)


def normalize_quantum_features(quantum_vec: np.ndarray) -> np.ndarray:
    """
    Normalize quantum features to reasonable ranges for neural network.
    """
    normalized = quantum_vec.copy()

    # HOMO: [-12, -4] -> [-1, 1]
    normalized[0] = (quantum_vec[0] + 8) / 4.0

    # LUMO: [-5, 2] -> [-1, 1]
    normalized[1] = (quantum_vec[1] + 1.5) / 3.5

    # HOMO-LUMO gap: [0, 10] -> [0, 1]
    normalized[2] = quantum_vec[2] / 10.0

    # Ionization potential: [4, 12] -> [0, 1]
    normalized[3] = (quantum_vec[3] - 4) / 8.0

    # Electron affinity: [-2, 5] -> [-1, 1]
    normalized[4] = (quantum_vec[4] + 2) / 7.0 * 2 - 1

    # Electronegativity: [2, 8] -> [0, 1]
    normalized[5] = (quantum_vec[5] - 2) / 6.0

    # Hardness: [1, 5] -> [0, 1]
    normalized[6] = (quantum_vec[6] - 1) / 4.0

    # Softness: [0.1, 5] -> [0, 1]
    normalized[7] = min(quantum_vec[7], 5.0) / 5.0

    # Electrophilicity: [0, 10] -> [0, 1]
    normalized[8] = min(quantum_vec[8], 10.0) / 10.0

    # Dipole moment: [0, 15] -> [0, 1]
    normalized[9] = min(quantum_vec[9], 15.0) / 15.0

    # Polarizability: [0, 50] -> [0, 1]
    normalized[10] = min(quantum_vec[10], 50.0) / 50.0

    # Partial charges: already in [-1, 1]
    normalized[11] = 0.0 if np.isnan(quantum_vec[11]) else np.clip(quantum_vec[11], -1, 1)
    normalized[12] = 0.0 if np.isnan(quantum_vec[12]) else np.clip(quantum_vec[12], -1, 1)

    # Clip all to reasonable range
    normalized = np.clip(normalized, -2, 2)

    return normalized


def mol_to_graph_enhanced(
    smiles: str,
    y: Optional[float] = None,
    include_quantum: bool = True,
    include_stereo: bool = True,
    use_dft: bool = True
) -> Optional[Data]:
    """
    Convert SMILES to graph with enhanced features.

    Args:
        smiles: SMILES string
        y: Optional target value
        include_quantum: Whether to include quantum descriptors (13 features)
        include_stereo: Whether to include stereochemistry features (6 features)
        use_dft: Whether to try PubChemQC first (vs RDKit only)

    Returns:
        Data object with:
        - x: Node features [num_atoms, N] where N = 15 + 13 + 6 = 34
        - edge_index: Graph connectivity
        - quantum_features: Graph-level quantum descriptors [13]
        - stereo_features: Graph-level stereochemistry features [6]
        - quantum_source: Whether DFT or RDKit approximation used
        - y: Target value
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # Get atom features (15 features per atom)
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    atom_features = np.array(atom_features, dtype=np.float32)
    num_atoms = atom_features.shape[0]

    # Collect molecular-level features to broadcast
    molecular_features = []
    quantum_source = None
    quantum_vec = None
    stereo_vec = None

    # Get quantum descriptors (13 features)
    if include_quantum:
        quantum_vec, quantum_source = get_quantum_features(smiles, use_dft=use_dft)
        quantum_vec_norm = normalize_quantum_features(quantum_vec)
        molecular_features.append(quantum_vec_norm)

    # Get stereochemistry features (6 features)
    if include_stereo:
        stereo_vec = get_stereo_features(smiles)
        molecular_features.append(stereo_vec)

    # Broadcast molecular features to all atoms
    if molecular_features:
        mol_features = np.concatenate(molecular_features)
        mol_broadcast = np.tile(mol_features, (num_atoms, 1))
        x = np.concatenate([atom_features, mol_broadcast], axis=1)
    else:
        x = atom_features

    x = torch.tensor(x, dtype=torch.float)

    # Get edges (bonds) with features
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)

        # Add both directions with same features
        edge_indices.append([i, j])
        edge_features.append(bond_feat)
        edge_indices.append([j, i])
        edge_features.append(bond_feat)

    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 7), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # Create Data object with edge features
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)

    data.smiles = smiles

    # Store molecular features separately for analysis
    if quantum_vec is not None:
        data.quantum_features = torch.tensor(quantum_vec, dtype=torch.float)
        data.quantum_source = quantum_source

    if stereo_vec is not None:
        data.stereo_features = torch.tensor(stereo_vec, dtype=torch.float)

    return data


def batch_smiles_to_graphs_enhanced(
    smiles_list: List[str],
    y_list: Optional[List[float]] = None,
    include_quantum: bool = True,
    include_stereo: bool = True,
    use_dft: bool = True,
    verbose: bool = True
) -> List[Data]:
    """
    Convert multiple SMILES to graph Data objects with enhanced features.

    Returns list of Data objects and prints statistics about data sources.
    """
    graphs = []
    dft_count = 0
    rdkit_count = 0
    stereo_count = 0

    for i, smiles in enumerate(smiles_list):
        y = y_list[i] if y_list is not None else None
        graph = mol_to_graph_enhanced(
            smiles, y,
            include_quantum=include_quantum,
            include_stereo=include_stereo,
            use_dft=use_dft
        )

        if graph is not None:
            graphs.append(graph)

            # Track sources
            if hasattr(graph, 'quantum_source'):
                if graph.quantum_source == 'pubchemqc':
                    dft_count += 1
                else:
                    rdkit_count += 1

            if hasattr(graph, 'stereo_features'):
                if graph.stereo_features[0] > 0 or graph.stereo_features[3] > 0:
                    stereo_count += 1

    if verbose:
        total = len(graphs)
        print(f"\nGraph conversion complete:")
        print(f"  Total graphs: {total}")
        if include_quantum:
            print(f"  Real DFT data: {dft_count} ({100*dft_count/total:.1f}%)")
            print(f"  RDKit approx:  {rdkit_count} ({100*rdkit_count/total:.1f}%)")
        if include_stereo:
            print(f"  With stereocenters: {stereo_count} ({100*stereo_count/total:.1f}%)")

    return graphs


if __name__ == "__main__":
    print("Testing Enhanced Molecule-to-Graph Conversion")
    print("=" * 70)
    print("Features: 15 atomic + 13 quantum + 6 stereochemistry = 34 total")
    print("=" * 70)

    test_molecules = [
        ('CCO', 'Ethanol'),
        ('c1ccccc1', 'Benzene'),
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'Caffeine'),
        ('CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'Ibuprofen'),
        ('C/C=C/C', 'E-2-butene'),  # E isomer
        ('C/C=C\\C', 'Z-2-butene'),  # Z isomer
        ('C[C@H](O)CC', 'R-2-butanol'),  # Chiral
    ]

    print("\nTesting individual molecules:")
    for smiles, name in test_molecules:
        print(f"\n{name} ({smiles}):")

        graph = mol_to_graph_enhanced(
            smiles, y=0.8,
            include_quantum=True,
            include_stereo=True,
            use_dft=True
        )

        if graph is None:
            print("  Failed to parse")
            continue

        print(f"  Features per atom: {graph.x.shape[1]}")
        print(f"  Atoms: {graph.x.shape[0]}")
        print(f"  Bonds: {graph.edge_index.shape[1] // 2}")

        if hasattr(graph, 'quantum_source'):
            print(f"  Quantum source: {graph.quantum_source}")

        if hasattr(graph, 'stereo_features'):
            sf = graph.stereo_features
            print(f"  Stereo features:")
            print(f"    E-Z: {sf[0]:.0f} (E:{sf[1]:.2f}, Z:{sf[2]:.2f})")
            print(f"    Chiral: {sf[3]:.0f} (R:{sf[4]:.2f}, S:{sf[5]:.2f})")

    print("\n" + "=" * 70)
    print("Enhanced graph conversion ready!")
    print("=" * 70)
