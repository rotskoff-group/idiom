from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.Chem.rdMolTransforms import GetDihedralRad
import warnings


def filter_neighbors(atm_idx: int, molecule: Chem.Mol) -> bool:
    """
    Filters out the hydrogen atoms from the neighbors of the given atom index

    Args:
        atm_idx: int
            The atom index to filter the neighbors for
        molecule: Chem.Mol
            The molecule object to extract the neighbors from

    Returns:
        bool: True if the atom index has neighbors that are not hydrogen atoms

    Notes:
        Neighbors cannot be:
        1) Hydrogen atoms
        2) Terminal atoms (atoms with only one neighbor)
        3) Atoms that are only connected to hydrogen atoms except for one other connection into the graph (essentially terminal atoms)
    """
    if molecule.GetAtomWithIdx(atm_idx).GetAtomicNum() == 1:
        return False
    nbrs = [
        nbr.GetAtomicNum() for nbr in molecule.GetAtomWithIdx(atm_idx).GetNeighbors()
    ]
    if len(nbrs) == 1:
        return False
    nbrs = np.array(nbrs)
    if np.where(nbrs > 1)[0].shape[0] == 1:
        return False
    return True


def compute_dihedrals_TFD(smiles: str) -> tuple[list[float], list[int]]:
    """
    Computes the dihedral angles over the canonicalized version of the given SMILES string
    Angles are computed in radians

    Args:
        smiles: str
            The SMILES string to compute the dihedral angles over

    Returns:
        dihedral_angles: list[float]
            List of dihedral angles in radians
        dihedral_indices: list[int]
            List of atom indices corresponding to the dihedral angles

    Notes:
        Some molecules fail the embedding by the AllChem.EmbedMolecule method and
        are subsequently excluded.

        This method uses the TorsionFingerPrints module in Rdkit, which automatically
        selects out torsional degrees of freedom from the molecule that are not part of rings
    """
    try:
        # FH: This operation can take some time even for relatively small datasets
        molecule = Chem.MolFromSmiles(smiles)
        molecule = Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule)
        AllChem.UFFOptimizeMolecule(molecule)
        dihedrals = TorsionFingerprints.CalculateTorsionLists(molecule)
        # FH: Separate angles and indices, sin and cosine computed over radians instead of degrees
        dihedral_angles = [
            np.deg2rad(dihedrals[0][i][1]) for i in range(len(dihedrals[0]))
        ]
        dihedral_indices = [dihedrals[0][i][0][0] for i in range(len(dihedrals[0]))]
        return dihedral_angles, dihedral_indices
    except Exception:
        return [], []


def compute_dihedrals_old(smiles: str) -> tuple[list[float], list[int]]:
    """
    Computes dihedral angles exhaustively over the canonicalized version of the given SMILES string
    Angles are computed in radians

    Args:
        smiles: str
            The SMILES string to compute the dihedral angles over

    Returns:
        torsion_angles: list[float]
            List of dihedral angles in radians
        torsion_indices: list[int]
            List of atom indices corresponding to the dihedral angles

    Notes:
        Some molecules fail the embedding by the AllChem.EmbedMolecule method and
        are subsequently excluded.

        This method is more exhaustive in its selection and calculation of dihedral angles
        over the molecule represented by the given SMILES string
    """
    warnings.warn(
        "This method is deprecated, use compute_dihedrals instead \n",
        DeprecationWarning,
    )
    try:
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None
        molecule = Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule)
        AllChem.UFFOptimizeMolecule(molecule)  # Optimize the geometry
        # Get conformer (for the optimized geometry)
        conformer = molecule.GetConformer()
    except Exception:
        # Invalid molecule encountered
        return [], []

    # Separate indices and angles for interface consistency
    torsion_angles = []
    torsion_indices = []
    # Generate the 3D conformer of the molecule (you need 3D coordinates for torsion angle calculation)

    # Find all torsions: bonds connecting two non-terminal atoms (atom pairs)
    for bond in molecule.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()

        # Exclude hydrogens from calculation
        if (
            molecule.GetAtomWithIdx(atom1).GetAtomicNum() == 1
            or molecule.GetAtomWithIdx(atom2).GetAtomicNum() == 1
        ):
            continue

        # Get neighboring atoms for atom1 and atom2
        nbrs1 = [
            nbr.GetIdx()
            for nbr in molecule.GetAtomWithIdx(atom1).GetNeighbors()
            if nbr.GetIdx() != atom2
        ]
        nbrs2 = [
            nbr.GetIdx()
            for nbr in molecule.GetAtomWithIdx(atom2).GetNeighbors()
            if nbr.GetIdx() != atom1
        ]
        nbrs1 = list(filter(lambda x: filter_neighbors(x, molecule), nbrs1))
        nbrs2 = list(filter(lambda x: filter_neighbors(x, molecule), nbrs2))

        # #Exclude terminal atoms (?)
        # if len(nbrs1) == 1 or len(nbrs2) == 1:
        #     continue

        # Torsion requires at least two neighbors for each atom
        if len(nbrs1) > 0 and len(nbrs2) > 0:
            for nbr1 in nbrs1:
                for nbr2 in nbrs2:
                    # Calculate the torsion angle using four atoms (nbr1, atom1, atom2, nbr2)
                    torsion_angle = GetDihedralRad(conformer, nbr1, atom1, atom2, nbr2)
                    torsion_indices.append((nbr1, atom1, atom2, nbr2))
                    torsion_angles.append(torsion_angle)
    # Radians for angles here
    return torsion_angles, torsion_indices


def compute_dihedrals(smiles: str, redundant: bool = False):
    """
    Computes dihedral angles exhaustively over the canonicalized version of the given SMILES string
    Angles are computed in radians

    smiles: The SMILES string to compute the dihedral angles over
    redundant: Whether to include redundant ring-closing dihedrals

    Note: Some molecules fail the embedding by the AllChem.EmbedMolecule method and
    are subsequently excluded.
    """
    try:
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None
        molecule = Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule, randomSeed=42)
        AllChem.UFFOptimizeMolecule(molecule)  # Optimize the geometry
        # Get conformer (for the optimized geometry)
        conformer = molecule.GetConformer()
    except Exception:
        # Invalid molecule encountered
        return [], []

    # Separate indices and angles for interface consistency
    torsion_angles = []
    torsion_indices = []
    if not redundant:
        visited = set()
    # Generate the 3D conformer of the molecule (you need 3D coordinates for torsion angle calculation)

    # Find all torsions: bonds connecting two non-terminal atoms (atom pairs)
    for bond in molecule.GetBonds():
        # Does not include bonds to hydrogen atoms
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()

        # Get neighboring atoms for atom1 and atom2
        nbrs1 = [
            nbr.GetIdx()
            for nbr in molecule.GetAtomWithIdx(atom1).GetNeighbors()
            if nbr.GetIdx() != atom2
        ]
        nbrs2 = [
            nbr.GetIdx()
            for nbr in molecule.GetAtomWithIdx(atom2).GetNeighbors()
            if nbr.GetIdx() != atom1
        ]

        # Exclude terminal atoms
        if len(nbrs1) > 0 and len(nbrs2) > 0:
            # Exclude triple and aromatic bonds
            if (
                bond.GetBondType() != Chem.rdchem.BondType.TRIPLE
                and bond.GetBondType() != Chem.rdchem.BondType.AROMATIC
            ):
                if not redundant:
                    # Exclude atoms closing rings
                    if molecule.GetAtomWithIdx(atom2).IsInRing() and atom2 in visited:
                        continue

                # Get arbitrary neighbors, they define the origin of torsion angle, which the model is oblivious to
                nbr1 = nbrs1[0]
                nbr2 = nbrs2[0]
                # Calculate the torsion angle using four atoms (nbr1, atom1, atom2, nbr2)
                torsion_angle = GetDihedralRad(conformer, nbr1, atom1, atom2, nbr2)
                torsion_indices.append((nbr1, atom1, atom2, nbr2))
                torsion_angles.append(torsion_angle)
        if not redundant:
            if (atom1, atom2) not in visited:
                visited.add(atom1)
    # Radians for angles here
    return torsion_angles, torsion_indices
