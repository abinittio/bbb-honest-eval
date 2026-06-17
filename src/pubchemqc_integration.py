"""
PubChemQC B3LYP/6-31G* Integration for BBB Prediction

Integrates real DFT-computed quantum descriptors from the PubChemQC database
(86 million molecules with B3LYP/6-31G* calculations) to replace RDKit approximations.

Also handles E-Z isomer (stereochemistry) encoding.

Sources:
- PubChemQC: https://nakatamaho.riken.jp/pubchemqc.riken.jp/
- Hugging Face: https://huggingface.co/datasets/molssiai-hub/pubchemqc-b3lyp
- Paper: https://pubs.acs.org/doi/10.1021/acs.jcim.3c00899
"""

import os
import json
import pickle
import hashlib
from typing import Dict, Optional, List, Tuple
from pathlib import Path

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: datasets library not installed. Run: pip install datasets")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

import numpy as np

# Cache directory for storing looked-up quantum properties
CACHE_DIR = Path("data/pubchemqc_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class PubChemQCIntegration:
    """
    Integrates PubChemQC B3LYP/6-31G* quantum descriptors into BBB prediction.

    Properties available:
    - HOMO energy (alpha/beta)
    - LUMO energy (alpha/beta)
    - HOMO-LUMO gap
    - Dipole moment
    - Total energy
    - Mulliken charges
    - Lowdin charges
    """

    def __init__(self, cache_file: str = "pubchemqc_lookup.pkl", use_streaming: bool = True):
        self.cache_file = CACHE_DIR / cache_file
        self.use_streaming = use_streaming
        self.lookup_cache: Dict[str, Dict] = {}
        self._load_cache()
        self.dataset = None

    def _load_cache(self):
        """Load cached lookups from disk"""
        if self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                self.lookup_cache = pickle.load(f)
            print(f"Loaded {len(self.lookup_cache)} cached PubChemQC entries")

    def _save_cache(self):
        """Save lookup cache to disk"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.lookup_cache, f)

    def _canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """Convert SMILES to canonical form for lookup"""
        if not RDKIT_AVAILABLE:
            return smiles
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None

    def _smiles_to_inchikey(self, smiles: str) -> Optional[str]:
        """Convert SMILES to InChIKey for reliable lookup"""
        if not RDKIT_AVAILABLE:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToInchiKey(mol)
        except:
            return None

    def initialize_dataset(self, subset: str = "b3lyp_pm6_chon500nosalt"):
        """
        Initialize connection to PubChemQC dataset on Hugging Face.

        Subsets available:
        - b3lyp_pm6: Full dataset (86M molecules)
        - b3lyp_pm6_chon300nosalt: C,H,O,N only, MW < 300
        - b3lyp_pm6_chon500nosalt: C,H,O,N only, MW < 500
        - b3lyp_pm6_chnopsfcl300nosalt: C,H,N,O,P,S,F,Cl, MW < 300
        - b3lyp_pm6_chnopsfcl500nosalt: C,H,N,O,P,S,F,Cl, MW < 500
        """
        if not HF_AVAILABLE:
            raise RuntimeError("datasets library required. Run: pip install datasets")

        print(f"Initializing PubChemQC dataset (subset: {subset})...")
        print("Note: Using streaming mode to avoid downloading entire dataset")

        self.dataset = load_dataset(
            path="molssiai-hub/pubchemqc-b3lyp",
            name=subset,
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        print("Dataset initialized successfully!")

    def build_lookup_index(self, smiles_list: List[str], batch_size: int = 10000):
        """
        Build lookup index for a list of SMILES from BBBP dataset.
        Streams through PubChemQC and caches matches.

        Args:
            smiles_list: List of SMILES strings to lookup
            batch_size: Number of PubChemQC entries to process at a time
        """
        if self.dataset is None:
            self.initialize_dataset()

        # Convert target SMILES to canonical forms and InChIKeys
        target_canonical = {}
        target_inchikeys = {}

        for smiles in smiles_list:
            canonical = self._canonicalize_smiles(smiles)
            if canonical:
                target_canonical[canonical] = smiles
                inchikey = self._smiles_to_inchikey(smiles)
                if inchikey:
                    target_inchikeys[inchikey] = smiles

        print(f"Looking up {len(target_canonical)} molecules in PubChemQC...")

        # Track found molecules
        found = 0
        total_scanned = 0

        try:
            for entry in self.dataset:
                total_scanned += 1

                # Check by InChIKey first (most reliable)
                entry_inchikey = entry.get('inchikey', '')
                if entry_inchikey in target_inchikeys:
                    original_smiles = target_inchikeys[entry_inchikey]
                    self._cache_entry(original_smiles, entry)
                    found += 1

                # Also check by SMILES
                entry_smiles = entry.get('smiles', '')
                canonical_entry = self._canonicalize_smiles(entry_smiles)
                if canonical_entry and canonical_entry in target_canonical:
                    original_smiles = target_canonical[canonical_entry]
                    if original_smiles not in self.lookup_cache:
                        self._cache_entry(original_smiles, entry)
                        found += 1

                if total_scanned % batch_size == 0:
                    print(f"Scanned {total_scanned:,} entries, found {found}/{len(smiles_list)} matches")
                    self._save_cache()

                # Early termination if all found
                if found >= len(smiles_list):
                    print(f"All {found} molecules found!")
                    break

        except KeyboardInterrupt:
            print(f"\nInterrupted. Scanned {total_scanned:,}, found {found} matches")

        self._save_cache()
        print(f"Lookup complete. Found {found}/{len(smiles_list)} molecules in PubChemQC")
        return found

    def _cache_entry(self, smiles: str, entry: Dict):
        """Extract and cache relevant quantum properties from PubChemQC entry"""
        properties = {
            # Core electronic properties
            'homo_energy': entry.get('homo', None),
            'lumo_energy': entry.get('lumo', None),
            'homo_lumo_gap': entry.get('homo_lumo_gap', None),
            'homo_alpha': entry.get('homo_alpha', None),
            'lumo_alpha': entry.get('lumo_alpha', None),
            'homo_beta': entry.get('homo_beta', None),
            'lumo_beta': entry.get('lumo_beta', None),

            # Other properties
            'total_energy': entry.get('total_energy', None),
            'dipole_moment': entry.get('dipole_moment', None),

            # Charges (if available)
            'mulliken_charges': entry.get('mulliken_charges', None),
            'lowdin_charges': entry.get('lowdin_charges', None),

            # Metadata
            'formula': entry.get('formula', None),
            'molecular_mass': entry.get('molecular_mass', None),
            'pubchem_cid': entry.get('cid', None),

            # Source info
            'source': 'pubchemqc_b3lyp_631g'
        }
        self.lookup_cache[smiles] = properties

    def get_quantum_descriptors(self, smiles: str) -> Optional[Dict]:
        """
        Get DFT quantum descriptors for a molecule.

        Returns dict with:
        - homo_energy: HOMO energy in Hartrees
        - lumo_energy: LUMO energy in Hartrees
        - homo_lumo_gap: Gap in Hartrees
        - dipole_moment: Dipole moment
        - electronegativity: Mulliken electronegativity from HOMO/LUMO
        - chemical_hardness: Chemical hardness from HOMO/LUMO
        - electrophilicity: Electrophilicity index
        """
        # Check cache first
        if smiles in self.lookup_cache:
            props = self.lookup_cache[smiles]
            return self._compute_derived_descriptors(props)

        # Try canonical SMILES
        canonical = self._canonicalize_smiles(smiles)
        if canonical and canonical in self.lookup_cache:
            props = self.lookup_cache[canonical]
            return self._compute_derived_descriptors(props)

        return None

    def _compute_derived_descriptors(self, props: Dict) -> Dict:
        """Compute derived quantum descriptors from raw DFT values"""
        homo = props.get('homo_energy')
        lumo = props.get('lumo_energy')
        gap = props.get('homo_lumo_gap')

        result = {
            'homo_energy': homo,
            'lumo_energy': lumo,
            'homo_lumo_gap': gap,
            'dipole_moment': props.get('dipole_moment'),
            'total_energy': props.get('total_energy'),
            'source': props.get('source', 'unknown')
        }

        # Compute conceptual DFT descriptors
        if homo is not None and lumo is not None:
            # Convert from Hartrees to eV for interpretability
            homo_ev = homo * 27.2114  # 1 Hartree = 27.2114 eV
            lumo_ev = lumo * 27.2114

            # Mulliken electronegativity: χ = -(HOMO + LUMO) / 2
            electronegativity = -(homo_ev + lumo_ev) / 2

            # Chemical hardness: η = (LUMO - HOMO) / 2
            hardness = (lumo_ev - homo_ev) / 2

            # Chemical softness: S = 1 / (2η)
            softness = 1 / (2 * hardness) if hardness > 0 else 0

            # Electrophilicity index: ω = χ² / (2η)
            electrophilicity = (electronegativity ** 2) / (2 * hardness) if hardness > 0 else 0

            result.update({
                'electronegativity': electronegativity,
                'chemical_hardness': hardness,
                'chemical_softness': softness,
                'electrophilicity': electrophilicity,
                'homo_ev': homo_ev,
                'lumo_ev': lumo_ev,
                'gap_ev': lumo_ev - homo_ev
            })

        return result


class StereochemistryEncoder:
    """
    Encodes E-Z isomer (geometric isomer) information from SMILES.

    E-Z isomers (cis-trans) are encoded using / and \ in SMILES:
    - F/C=C/F is trans (E) 1,2-difluoroethylene
    - F/C=C\F is cis (Z) 1,2-difluoroethylene

    This can significantly affect BBB permeability due to transporter stereoselectivity.
    """

    def __init__(self):
        self.stereo_cache = {}

    def smiles_to_isomeric(self, smiles: str) -> str:
        """Convert standard SMILES to isomeric SMILES with stereochemistry"""
        if not RDKIT_AVAILABLE:
            return smiles
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            # Generate isomeric SMILES (includes stereochemistry)
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            return smiles

    def get_ez_isomer_features(self, smiles: str) -> Dict:
        """
        Extract E-Z isomer features from a molecule.

        Returns:
        - has_double_bonds: Whether molecule has C=C double bonds
        - num_ez_centers: Number of E-Z stereogenic centers
        - e_count: Number of E (trans) configurations
        - z_count: Number of Z (cis) configurations
        - ez_ratio: Ratio of E to total defined stereocenters
        - stereo_defined: Whether stereochemistry is defined
        """
        if smiles in self.stereo_cache:
            return self.stereo_cache[smiles]

        features = {
            'has_double_bonds': False,
            'num_ez_centers': 0,
            'e_count': 0,
            'z_count': 0,
            'ez_ratio': 0.5,
            'stereo_defined': False,
            'num_chiral_centers': 0,
            'r_count': 0,
            's_count': 0
        }

        if not RDKIT_AVAILABLE:
            return features

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return features

            # Check for double bonds
            for bond in mol.GetBonds():
                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    features['has_double_bonds'] = True

                    # Check if it's a stereogenic double bond
                    stereo = bond.GetStereo()
                    if stereo != Chem.rdchem.BondStereo.STEREONONE:
                        features['num_ez_centers'] += 1
                        features['stereo_defined'] = True

                        if stereo == Chem.rdchem.BondStereo.STEREOE:
                            features['e_count'] += 1
                        elif stereo == Chem.rdchem.BondStereo.STEREOZ:
                            features['z_count'] += 1

            # Calculate E/Z ratio
            total_ez = features['e_count'] + features['z_count']
            if total_ez > 0:
                features['ez_ratio'] = features['e_count'] / total_ez

            # Also get chiral center info (R/S stereochemistry)
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            features['num_chiral_centers'] = len(chiral_centers)

            for atom_idx, chirality in chiral_centers:
                if chirality == 'R':
                    features['r_count'] += 1
                elif chirality == 'S':
                    features['s_count'] += 1

        except Exception as e:
            pass

        self.stereo_cache[smiles] = features
        return features

    def get_stereo_feature_vector(self, smiles: str) -> np.ndarray:
        """
        Get stereochemistry features as a numpy vector for ML.

        Returns 8-dimensional vector:
        [has_double_bonds, num_ez_centers, e_count, z_count,
         ez_ratio, num_chiral_centers, r_count, s_count]
        """
        features = self.get_ez_isomer_features(smiles)
        return np.array([
            float(features['has_double_bonds']),
            features['num_ez_centers'],
            features['e_count'],
            features['z_count'],
            features['ez_ratio'],
            features['num_chiral_centers'],
            features['r_count'],
            features['s_count']
        ], dtype=np.float32)


def integrate_quantum_and_stereo_features(
    smiles: str,
    pubchemqc: PubChemQCIntegration,
    stereo_encoder: StereochemistryEncoder,
    fallback_to_rdkit: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Get combined quantum and stereochemistry features for a molecule.

    Args:
        smiles: SMILES string
        pubchemqc: PubChemQC integration instance
        stereo_encoder: Stereochemistry encoder instance
        fallback_to_rdkit: Use RDKit approximations if not in PubChemQC

    Returns:
        Tuple of (feature_vector, feature_dict)
    """
    # Get stereochemistry features (always available via RDKit)
    stereo_features = stereo_encoder.get_stereo_feature_vector(smiles)
    stereo_dict = stereo_encoder.get_ez_isomer_features(smiles)

    # Try to get real quantum descriptors from PubChemQC
    quantum_dict = pubchemqc.get_quantum_descriptors(smiles)

    if quantum_dict is not None and quantum_dict.get('homo_energy') is not None:
        # Use real DFT values
        quantum_features = np.array([
            quantum_dict.get('homo_ev', 0),
            quantum_dict.get('lumo_ev', 0),
            quantum_dict.get('gap_ev', 0),
            quantum_dict.get('electronegativity', 0),
            quantum_dict.get('chemical_hardness', 0),
            quantum_dict.get('chemical_softness', 0),
            quantum_dict.get('electrophilicity', 0),
            quantum_dict.get('dipole_moment', 0) if quantum_dict.get('dipole_moment') else 0,
        ], dtype=np.float32)
        source = 'pubchemqc_dft'
    elif fallback_to_rdkit and RDKIT_AVAILABLE:
        # Fallback to RDKit approximations
        quantum_features = _get_rdkit_quantum_approximations(smiles)
        source = 'rdkit_approximation'
    else:
        # No quantum features available
        quantum_features = np.zeros(8, dtype=np.float32)
        source = 'none'

    # Combine features
    combined = np.concatenate([quantum_features, stereo_features])

    feature_dict = {
        'quantum': quantum_dict,
        'stereo': stereo_dict,
        'source': source
    }

    return combined, feature_dict


def _get_rdkit_quantum_approximations(smiles: str) -> np.ndarray:
    """Fallback: Get approximate quantum features from RDKit"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(8, dtype=np.float32)

        # Gasteiger charges for electronegativity approximation
        AllChem.ComputeGasteigerCharges(mol)
        charges = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge')
                   for i in range(mol.GetNumAtoms())]
        charges = [c for c in charges if not np.isnan(c)]

        # Approximate HOMO/LUMO from molecular properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        # Very rough approximations
        homo_approx = -5.5 - 0.005 * mw + 0.1 * logp  # Typical organic range
        lumo_approx = -1.5 - 0.002 * mw - 0.05 * logp
        gap_approx = lumo_approx - homo_approx

        electronegativity = -(homo_approx + lumo_approx) / 2
        hardness = gap_approx / 2
        softness = 1 / (2 * hardness) if hardness > 0 else 0
        electrophilicity = (electronegativity ** 2) / (2 * hardness) if hardness > 0 else 0

        # Dipole approximation from TPSA
        dipole_approx = tpsa / 20  # Very rough

        return np.array([
            homo_approx, lumo_approx, gap_approx,
            electronegativity, hardness, softness, electrophilicity,
            dipole_approx
        ], dtype=np.float32)

    except:
        return np.zeros(8, dtype=np.float32)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PubChemQC Integration Demo")
    print("=" * 70)

    # Initialize components
    pubchemqc = PubChemQCIntegration()
    stereo = StereochemistryEncoder()

    # Test molecules
    test_smiles = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        r"C/C=C/C",  # trans-2-butene (E isomer)
        r"C/C=C\C",  # cis-2-butene (Z isomer)
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",  # Melatonin
    ]

    print("\nStereochemistry Analysis:")
    print("-" * 70)
    for smiles in test_smiles:
        isomeric = stereo.smiles_to_isomeric(smiles)
        features = stereo.get_ez_isomer_features(smiles)
        print(f"\nSMILES: {smiles}")
        print(f"  Isomeric: {isomeric}")
        print(f"  Has double bonds: {features['has_double_bonds']}")
        print(f"  E-Z centers: {features['num_ez_centers']} (E:{features['e_count']}, Z:{features['z_count']})")
        print(f"  Chiral centers: {features['num_chiral_centers']} (R:{features['r_count']}, S:{features['s_count']})")

    print("\n" + "=" * 70)
    print("To use real PubChemQC DFT values:")
    print("=" * 70)
    print("""
    1. Install: pip install datasets

    2. Build lookup for BBBP dataset:
       pubchemqc = PubChemQCIntegration()
       pubchemqc.initialize_dataset('b3lyp_pm6_chon500nosalt')

       # Get SMILES from BBBP
       bbbp_smiles = [...]  # Your BBBP SMILES list
       pubchemqc.build_lookup_index(bbbp_smiles)

    3. Get quantum features:
       quantum = pubchemqc.get_quantum_descriptors('CCO')
       print(quantum['homo_ev'], quantum['lumo_ev'], quantum['electronegativity'])

    4. Integrate into mol_to_graph.py for GNN training
    """)
