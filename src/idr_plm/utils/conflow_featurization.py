# allowable multiple choice node and edge features
from collections import defaultdict
from typing import Callable, Tuple
from copy import deepcopy


import torch
import numpy as np
from datamol.types import Mol
from rdkit import Chem
from torch_geometric.data import Data
import datamol as dm

from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
from torch_cluster import radius_graph

from rdkit.Chem.rdchem import Conformer
from rdkit.Geometry import Point3D


# similar to GeoMol
BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
chirality = {
    ChiralType.CHI_TETRAHEDRAL_CW: -1,
    ChiralType.CHI_TETRAHEDRAL_CCW: 1,
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_OTHER: 0,
}

allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [0, -1, 1, 0, "misc"],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_implicit_valence": [0, 1, 2, 3, 4, 5, 6, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": [
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "S",
        "misc",
    ],  # Changed wrt ET-Flow code. S and misc are the same in the et-flow code (they don't have S)
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}

geom_atom_mapping = {
    5: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    14: 5,
    15: 6,
    16: 7,
    17: 8,
    35: 9,
    53: 10,
    83: 11,
}


# Gradient clipping
class Queue:
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def get_atomic_number_and_charge(mol: Chem.Mol):
    """Returns atoms number and charge for rdkit molecule"""
    return np.array(
        [[atom.GetAtomicNum(), atom.GetFormalCharge()] for atom in mol.GetAtoms()]
    )


def GetNumRings(atom):
    return sum([atom.IsInRingSize(i) for i in range(3, 7)])


def atom_to_feature_vector(atom):
    """Node Invariant Features for an Atom."""
    atom_feature = [
        safe_index(
            allowable_features["possible_chirality_list"],
            chirality[atom.GetChiralTag()],
        ),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(
            allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()
        ),
        safe_index(
            allowable_features["possible_implicit_valence"], atom.GetImplicitValence()
        ),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_hybridization_list"],
            str(atom.GetHybridization()),
        ),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
        GetNumRings(atom),  # added wrt to original ET-Flow code
        int(atom.GetAtomicNum()),  # added wrt to original ET-Flow code
        int(atom.GetTotalValence()),  # added wrt to original ET-Flow code
    ]
    return atom_feature


def safe_index(list_of_elements, element):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return list_of_elements.index(element)
    except Exception as exception:
        print(
            f"Element {element} not found in list {list_of_elements}. Returning last index."
        )
        print(exception)
        return len(list_of_elements) - 1


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(
            allowable_features["possible_bond_type_list"], str(bond.GetBondType())
        ),
        # allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
        # allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def compute_edge_index(
    mol, no_reverse: bool = False, with_edge_attr=False
) -> torch.Tensor:
    """Computes edge index from mol object"""
    edge_list = []
    bond_types = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_list.append((i, j))
        bond_types.append(bond_to_feature_vector(bond))

        if not no_reverse:
            edge_list.append((j, i))
            bond_types.append(bond_to_feature_vector(bond))

    if len(edge_list) == 0:
        return torch.empty((2, 0)).long()

    edge_index = torch.from_numpy(np.array(edge_list).T).long()

    if with_edge_attr:
        edge_attr = torch.tensor(bond_types, dtype=torch.float32)  # (num_edges, 1)
        return edge_index, edge_attr

    return edge_index, None


def _extend_to_radius_graph(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float = 10.0,
    max_num_neighbors: int = 32,
    unspecified_type_number=0,
):
    assert edge_type.dim() == 1
    N = pos.size(0)

    bgraph_adj = torch.sparse_coo_tensor(edge_index, edge_type, torch.Size([N, N]))
    rgraph_edge_index = radius_graph(
        pos, r=cutoff, batch=batch, max_num_neighbors=max_num_neighbors
    )  # (2, E_r)

    rgraph_adj = torch.sparse_coo_tensor(
        rgraph_edge_index,
        torch.ones(rgraph_edge_index.size(1)).long().to(pos.device)
        * unspecified_type_number,
        torch.Size([N, N]),
    )

    composed_adj = (bgraph_adj + rgraph_adj).coalesce()  # Sparse (N, N, T)

    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()

    return new_edge_index, new_edge_type


def extend_graph_order_radius(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float = 10.0,
    max_num_neighbors: int = 32,
    extend_radius: bool = True,
):
    """Extends bond index"""
    if extend_radius:
        edge_index, edge_type = _extend_to_radius_graph(
            pos=pos,
            edge_index=edge_index,
            edge_type=edge_type,
            cutoff=cutoff,
            batch=batch,
            max_num_neighbors=max_num_neighbors,
        )

    return edge_index, edge_type


def get_neighbor_ids(data):
    """
    Takes the edge indices and returns dictionary mapping atom index to neighbor indices
    Note: this only includes atoms with degree > 1
    """
    batch_nbrs = deepcopy(data.neighbors)
    batch_nbrs = [obj[0] for obj in batch_nbrs]
    neighbors = batch_nbrs.pop(0)  # get first element
    n_atoms_per_mol = data.batch.bincount()  # get atom count per graph
    n_atoms_prev_mol = 0

    for i, n_dict in enumerate(batch_nbrs):
        new_dict = {}
        n_atoms_prev_mol += n_atoms_per_mol[i].item()
        for k, v in n_dict.items():
            new_dict[k + n_atoms_prev_mol] = v + n_atoms_prev_mol
        neighbors.update(new_dict)

    return neighbors


def signed_volume(local_coords):
    """
    Compute signed volume given ordered neighbor local coordinates
    From GeoMol

    :param local_coords: (n_tetrahedral_chiral_centers, 4, n_generated_confs, 3)
    :return: signed volume of each tetrahedral center
    (n_tetrahedral_chiral_centers, n_generated_confs)
    """
    v1 = local_coords[:, 0] - local_coords[:, 3]
    v2 = local_coords[:, 1] - local_coords[:, 3]
    v3 = local_coords[:, 2] - local_coords[:, 3]
    cp = v2.cross(v3, dim=-1)
    vol = torch.sum(v1 * cp, dim=-1)
    return torch.sign(vol)


def get_chiral_tensors(mol):
    """Only consider chiral atoms with 4 neighbors"""
    chiral_index = torch.tensor(
        [
            i
            for i, atom in enumerate(mol.GetAtoms())
            if (chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4)
        ],
        dtype=torch.int32,
    ).view(1, -1)  # (1, n_chiral_centers)
    # (n_chiral_centers, 4)
    chiral_nbr_index = torch.tensor(
        [
            [n.GetIdx() for n in atom.GetNeighbors()]
            for atom in mol.GetAtoms()
            if (chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4)
        ],
        dtype=torch.int32,
    ).view(1, -1)  # (1, n_chiral_centers * 4)
    # (n_chiral_centers,)
    chiral_tag = torch.tensor(
        [
            chirality[atom.GetChiralTag()]
            for atom in mol.GetAtoms()
            if (chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4)
        ],
        dtype=torch.float32,
    )

    return chiral_index, chiral_nbr_index, chiral_tag


def get_mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return mol


def cache_decorator(func: Callable):
    """Decorator to handle caching logic."""

    def wrapper(self, smiles: str, *args, **kwargs):
        cache_key = func.__name__
        if smiles in self.cache and cache_key in self.cache[smiles]:
            return self.cache[smiles][cache_key]
        result = func(self, smiles, *args, **kwargs)
        self.cache[smiles][cache_key] = result
        return result

    return wrapper


class MoleculeFeaturizer:
    """A Featurizer Class for Molecules.
    - Give smiles, get mol objects, atom features, bond features, etc.
    - Smiles-based Caching to avoid recomputation.
    """

    def __init__(self):
        # smiles based cache
        self.cache = defaultdict(dict)

    def get_mol(self, smiles: str) -> Mol:
        return dm.to_mol(smiles, remove_hs=False, ordered=True)

    @cache_decorator
    def get_atom_features(self, smiles: str, use_ogb_feat: bool = True) -> torch.Tensor:
        # compute atom features
        mol = self.get_mol(smiles)
        atom_features = self.get_atom_features_from_mol(mol, use_ogb_feat=use_ogb_feat)
        return atom_features

    @cache_decorator
    def get_atomic_numbers(self, smiles: str) -> torch.Tensor:
        # compute atomic numbers
        mol = self.get_mol(smiles)
        atomic_numbers = self.get_atomic_numbers_from_mol(mol)
        return atomic_numbers

    def get_atomic_numbers_from_mol(self, mol: Mol) -> torch.Tensor:
        atomic_numbers = torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()],
            dtype=torch.int32,
        )
        return atomic_numbers

    def get_atom_features_from_mol(
        self, mol: Mol, use_ogb_feat: bool = True
    ) -> torch.Tensor:
        if use_ogb_feat:
            atom_features = torch.tensor(
                [atom_to_feature_vector(atom) for atom in mol.GetAtoms()],
                dtype=torch.float32,
            )
        else:
            atom_features = torch.tensor(
                [atom.GetFormalCharge() for atom in mol.GetAtoms()],
                dtype=torch.float32,
            ).view(-1, 1)
        return atom_features

    @cache_decorator
    def get_chiral_centers(self, smiles: str) -> torch.Tensor:
        # compute chiral centers
        mol = self.get_mol(smiles)
        chiral_index, chiral_nbr_index, chiral_tag = self.get_chiral_centers_from_mol(
            mol
        )

        self.cache[smiles]["chiral_centers"] = (
            chiral_index,
            chiral_nbr_index,
            chiral_tag,
        )
        return chiral_index, chiral_nbr_index, chiral_tag

    def get_chiral_centers_from_mol(self, mol: Mol) -> torch.Tensor:
        chiral_index, chiral_nbr_index, chiral_tag = get_chiral_tensors(mol)
        return chiral_index, chiral_nbr_index, chiral_tag

    @cache_decorator
    def get_mol_with_conformer(self, smiles: str, positions: torch.Tensor) -> Mol:
        mol = self.get_mol(smiles)
        mol.AddConformer(build_conformer(positions))
        return mol

    @cache_decorator
    def get_edge_index(
        self, smiles: str, use_edge_feat: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns edge index and edge attributes for a given smiles."""
        # compute edge index
        mol = self.get_mol(smiles)
        edge_index, edge_attr = self.get_edge_index_from_mol(
            mol, use_edge_feat=use_edge_feat
        )

        self.cache[smiles]["edge_index"] = edge_index
        self.cache[smiles]["edge_attr"] = edge_attr
        return edge_index, edge_attr

    def get_edge_index_from_mol(
        self, mol: Mol, use_edge_feat: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns edge index and edge attributes for a given mol object."""
        edge_index, edge_attr = compute_edge_index(mol, with_edge_attr=use_edge_feat)
        return edge_index, edge_attr

    def get_data_from_smiles(self, smiles: str) -> Data:
        mol = get_mol_from_smiles(smiles)  # added hs
        smiles_changed = dm.to_smiles(
            mol,
            canonical=False,
            explicit_hs=True,
            with_atom_indices=True,
            isomeric=True,
        )
        node_attr = self.get_atom_features_from_mol(mol, True)
        chiral_index, chiral_nbr_index, chiral_tag = self.get_chiral_centers_from_mol(
            mol
        )
        edge_index, edge_attr = self.get_edge_index_from_mol(mol, False)
        atomic_numbers = self.get_atomic_numbers_from_mol(mol)

        graph = Data(
            atomic_numbers=atomic_numbers,
            smiles=smiles_changed,
            edge_index=edge_index,
            chiral_index=chiral_index,
            chiral_nbr_index=chiral_nbr_index,
            chiral_tag=chiral_tag,
            node_attr=node_attr,
            edge_attr=edge_attr,
        )
        return graph


def build_conformer(pos):
    if isinstance(pos, torch.Tensor) or isinstance(pos, np.ndarray):
        pos = pos.tolist()

    conformer = Conformer()

    for i, atom_pos in enumerate(pos):
        conformer.SetAtomPosition(i, Point3D(*atom_pos))

    return conformer
