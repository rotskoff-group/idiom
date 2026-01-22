import torch


def get_edge_attr_and_edge_index(mol):
    bond_matrix = get_bond_matrix(mol)
    edge_index = torch.stack(torch.where(bond_matrix != -1))
    edge_attr = bond_matrix[edge_index[0], edge_index[1]]
    return edge_index, edge_attr


def get_bond_matrix(mol):
    N = mol.GetNumAtoms()
    bonds = get_bonds(mol)

    bond_matrix = torch.zeros((N, N), dtype=int)
    bond_matrix[bonds[:, 0], bonds[:, 1]] = bonds[:, 2]
    bond_matrix[bonds[:, 1], bonds[:, 0]] = bonds[:, 2]
    bond_matrix += torch.eye(N, dtype=torch.long) * -1

    return bond_matrix


def get_bonds(mol):
    bonds = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble())
        for bond in mol.GetBonds()
    ]
    return torch.tensor(bonds, dtype=torch.long)


def check_disconnected_components(mol):
    """Check for disconnected components using Union-Find algorithm."""
    # Initialize parent array for union-find
    n_nodes = mol.GetNumAtoms()
    parent = list(range(n_nodes))

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # Add reverse edge for undirected graph
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    # Process all edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        union(src, dst)

    # Count unique components
    components = {}
    for node in range(n_nodes):
        root = find(node)
        if root not in components:
            components[root] = []
        components[root].append(node)

    return list(components.values())
