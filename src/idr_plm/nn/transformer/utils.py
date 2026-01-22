def get_node_features(mol):
    node_features = []
    for atom in mol.GetAtoms():
        element = atom.GetAtomicNum()
        valency = atom.GetTotalValence()
        hybridization_str = str(atom.GetHybridization())

        if hybridization_str == "SP":
            hybridization = 0
        elif hybridization_str == "SP2":
            hybridization = 1
        elif hybridization_str == "SP3":
            hybridization = 2
        elif hybridization_str == "SP3D":
            hybridization = 3
        elif hybridization_str == "SP3D2":
            hybridization = 4
        elif hybridization_str == "S":
            hybridization = 5
        elif hybridization_str == "UNSPECIFIED":
            hybridization = 6

        node_features.append([element, valency, hybridization])

    return node_features
