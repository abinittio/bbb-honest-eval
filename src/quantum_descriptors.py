"""
Quantum Descriptors Calculator for BBB Prediction

Calculates quantum chemical descriptors similar to those from Gaussian:
- HOMO (Highest Occupied Molecular Orbital) energy
- LUMO (Lowest Unoccupied Molecular Orbital) energy
- HOMO-LUMO gap (chemical hardness indicator)
- Dipole moment
- Polarizability
- Electrophilicity index
- Chemical hardness/softness
- Electronegativity (Mulliken)

Since we don't have access to actual Gaussian calculations (expensive DFT),
we use semi-empirical approximations via RDKit and Mordred descriptors.

For production, these could be replaced with actual DFT calculations.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdPartialCharges

# Try to import mordred for advanced descriptors
try:
    from mordred import Calculator, descriptors
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False
    print("Note: mordred not installed. Using RDKit approximations for quantum descriptors.")
    print("Install with: pip install mordred")


class QuantumDescriptorCalculator:
    """
    Calculate quantum-like descriptors for molecules.

    These are approximations based on semi-empirical methods and
    empirical correlations, not actual DFT calculations.

    For real quantum descriptors, you would:
    1. Generate 3D conformers
    2. Run Gaussian/ORCA with B3LYP/6-31G* or similar
    3. Extract HOMO, LUMO, dipole from output

    This class provides fast approximations suitable for ML.
    """

    def __init__(self):
        if MORDRED_AVAILABLE:
            # Initialize mordred calculator with specific descriptors
            self.mordred_calc = Calculator(descriptors, ignore_3D=True)

    def calculate(self, smiles):
        """
        Calculate quantum descriptors for a molecule.

        Returns dict with:
            - homo_approx: Approximate HOMO energy (eV)
            - lumo_approx: Approximate LUMO energy (eV)
            - homo_lumo_gap: HOMO-LUMO gap (eV)
            - dipole_moment: Approximate dipole moment (Debye)
            - polarizability: Molecular polarizability (Å³)
            - electrophilicity: Electrophilicity index (eV)
            - hardness: Chemical hardness (eV)
            - softness: Chemical softness (eV⁻¹)
            - electronegativity: Mulliken electronegativity (eV)
            - electron_affinity: Electron affinity approximation (eV)
            - ionization_potential: Ionization potential approximation (eV)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add hydrogens for better calculations
        mol = Chem.AddHs(mol)

        descriptors_dict = {}

        # === HOMO/LUMO Approximations ===
        # Based on empirical correlations with molecular properties
        # Real values require DFT (B3LYP/6-31G* typical)

        # Get basic properties
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        mw = Descriptors.MolWt(mol)
        num_electrons = sum([atom.GetAtomicNum() for atom in mol.GetAtoms()])

        # Aromatic character affects HOMO-LUMO
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        num_aromatic_atoms = len([a for a in mol.GetAtoms() if a.GetIsAromatic()])
        frac_aromatic = num_aromatic_atoms / mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 0

        # Heteroatom effects
        num_n = len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 7])
        num_o = len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 8])
        num_s = len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 16])
        num_halogens = len([a for a in mol.GetAtoms() if a.GetAtomicNum() in [9, 17, 35, 53]])

        # === Empirical HOMO/LUMO estimation ===
        # Based on: Ghose-Crippen correlations and aromatic stabilization
        # These are rough approximations! Real DFT gives better values.

        # HOMO estimation (typically -5 to -9 eV for organic molecules)
        # More aromatic = higher HOMO (less negative)
        # More electronegative atoms = lower HOMO (more negative)
        homo_base = -7.5  # typical organic molecule HOMO
        homo_aromatic_contrib = frac_aromatic * 1.5  # aromatic stabilization
        homo_heteroatom_contrib = -0.1 * (num_n + num_o) - 0.2 * num_s
        homo_halogen_contrib = -0.15 * num_halogens
        homo_size_contrib = -0.001 * (mw - 200)  # size effect

        homo_approx = homo_base + homo_aromatic_contrib + homo_heteroatom_contrib + homo_halogen_contrib + homo_size_contrib
        homo_approx = np.clip(homo_approx, -12, -4)  # physical bounds

        # LUMO estimation (typically 0 to -4 eV for organic molecules)
        # More aromatic = lower LUMO (more negative, better acceptor)
        # Electron-withdrawing groups lower LUMO
        lumo_base = -1.5
        lumo_aromatic_contrib = -0.5 * frac_aromatic - 0.3 * num_aromatic_rings
        lumo_ewg_contrib = -0.2 * num_halogens - 0.1 * num_n
        lumo_size_contrib = -0.0005 * (mw - 200)

        lumo_approx = lumo_base + lumo_aromatic_contrib + lumo_ewg_contrib + lumo_size_contrib
        lumo_approx = np.clip(lumo_approx, -5, 2)  # physical bounds

        # Ensure LUMO > HOMO
        if lumo_approx <= homo_approx:
            lumo_approx = homo_approx + 3.0  # minimum gap

        descriptors_dict['homo_approx'] = homo_approx
        descriptors_dict['lumo_approx'] = lumo_approx
        descriptors_dict['homo_lumo_gap'] = lumo_approx - homo_approx

        # === Derived Quantum Properties ===
        # Koopman's theorem approximations
        ionization_potential = -homo_approx  # IP ≈ -HOMO
        electron_affinity = -lumo_approx  # EA ≈ -LUMO

        descriptors_dict['ionization_potential'] = ionization_potential
        descriptors_dict['electron_affinity'] = electron_affinity

        # Mulliken electronegativity: χ = (IP + EA) / 2
        electronegativity = (ionization_potential + electron_affinity) / 2
        descriptors_dict['electronegativity'] = electronegativity

        # Chemical hardness: η = (IP - EA) / 2 = gap / 2
        hardness = (ionization_potential - electron_affinity) / 2
        descriptors_dict['hardness'] = hardness

        # Chemical softness: S = 1 / (2η)
        softness = 1 / (2 * hardness) if hardness > 0.1 else 5.0
        descriptors_dict['softness'] = softness

        # Electrophilicity index: ω = χ² / (2η)
        electrophilicity = (electronegativity ** 2) / (2 * hardness) if hardness > 0.1 else 0
        descriptors_dict['electrophilicity'] = electrophilicity

        # === Dipole Moment Approximation ===
        # Based on charge separation and molecular geometry
        # Compute Gasteiger charges
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = [float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge'))
                      for i in range(mol.GetNumAtoms())]
            charges = [c if not np.isnan(c) else 0 for c in charges]

            # Dipole approximation based on charge separation
            max_charge = max(charges) if charges else 0
            min_charge = min(charges) if charges else 0
            charge_sep = max_charge - min_charge

            # Estimate dipole (rough approximation)
            # Real dipole needs 3D geometry
            dipole_approx = charge_sep * np.sqrt(mw) * 0.5 + tpsa * 0.02
            dipole_approx = np.clip(dipole_approx, 0, 15)  # Debye, typical range

        except:
            dipole_approx = tpsa * 0.05  # fallback

        descriptors_dict['dipole_moment'] = dipole_approx

        # === Polarizability ===
        # Miller's empirical formula or atom-based additivity
        # α ≈ 0.79 × (number of valence electrons)^0.67 × Å³
        polarizability = 0.79 * (num_electrons ** 0.67)
        descriptors_dict['polarizability'] = polarizability

        # === Additional quantum-relevant descriptors ===
        if MORDRED_AVAILABLE:
            try:
                mol_no_h = Chem.RemoveHs(mol)
                mordred_result = self.mordred_calc(mol_no_h)

                # Get specific mordred descriptors if available
                # These are more accurate than our approximations
                for desc_name in ['HOMO', 'LUMO', 'GapHL']:
                    if desc_name in mordred_result and mordred_result[desc_name] is not None:
                        if not np.isnan(mordred_result[desc_name]):
                            descriptors_dict[f'mordred_{desc_name}'] = float(mordred_result[desc_name])
            except:
                pass

        # === Quantum-relevant molecular properties ===
        # These correlate with quantum behavior

        # Maximum partial charge
        descriptors_dict['max_partial_charge'] = Descriptors.MaxPartialCharge(mol)
        descriptors_dict['min_partial_charge'] = Descriptors.MinPartialCharge(mol)

        # Handle NaN values
        for key in descriptors_dict:
            if descriptors_dict[key] is None or (isinstance(descriptors_dict[key], float) and np.isnan(descriptors_dict[key])):
                descriptors_dict[key] = 0.0

        return descriptors_dict

    def get_feature_names(self):
        """Return list of quantum descriptor names"""
        return [
            'homo_approx',
            'lumo_approx',
            'homo_lumo_gap',
            'ionization_potential',
            'electron_affinity',
            'electronegativity',
            'hardness',
            'softness',
            'electrophilicity',
            'dipole_moment',
            'polarizability',
            'max_partial_charge',
            'min_partial_charge'
        ]

    def calculate_vector(self, smiles):
        """Return quantum descriptors as a numpy array"""
        desc = self.calculate(smiles)
        if desc is None:
            return None

        feature_names = self.get_feature_names()
        return np.array([desc.get(name, 0.0) for name in feature_names], dtype=np.float32)


def test_quantum_descriptors():
    """Test the quantum descriptor calculator"""
    calc = QuantumDescriptorCalculator()

    test_molecules = [
        ("CCO", "Ethanol"),
        ("c1ccccc1", "Benzene"),
        ("CC(=O)Nc1ccc(O)cc1", "Paracetamol"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen"),
    ]

    print("Quantum Descriptors Test")
    print("=" * 80)

    for smiles, name in test_molecules:
        print(f"\n{name} ({smiles})")
        print("-" * 40)

        desc = calc.calculate(smiles)
        if desc:
            print(f"  HOMO (approx):     {desc['homo_approx']:.2f} eV")
            print(f"  LUMO (approx):     {desc['lumo_approx']:.2f} eV")
            print(f"  HOMO-LUMO gap:     {desc['homo_lumo_gap']:.2f} eV")
            print(f"  Electronegativity: {desc['electronegativity']:.2f} eV")
            print(f"  Hardness:          {desc['hardness']:.2f} eV")
            print(f"  Electrophilicity:  {desc['electrophilicity']:.2f} eV")
            print(f"  Dipole moment:     {desc['dipole_moment']:.2f} D")
            print(f"  Polarizability:    {desc['polarizability']:.2f} Å³")
        else:
            print("  Failed to calculate")


if __name__ == "__main__":
    test_quantum_descriptors()
