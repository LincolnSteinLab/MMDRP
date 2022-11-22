import torch
from rdkit import Chem
# from rdkit.Chem import AllChem
# import gdsctools
import pandas as pd
# import multiprocessing

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets.molecule_net import x_map, e_map

from DataImportModules import GenFeatures

if __name__ == '__main__':

    path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
    # file_name = "chembl_27_chemreps.txt"
    file_name = "CTRP_AAC_SMILES.txt"
    print("Reading file...")
    # all_data = pd.read_csv(path + file_name, engine='c', sep='\t')
    all_data = pd.read_csv(path + file_name, engine='c', sep=',')
    full_train = all_data[["ccl_name", "primary_disease", "cpd_name", "cpd_smiles", "area_above_curve"]]

    cur_compounds = full_train[['cpd_name', 'cpd_smiles']].drop_duplicates()
    cur_smiles = cur_compounds['cpd_smiles'].tolist()
    cur_cpd_names = cur_compounds['cpd_name'].tolist()
    print("Converting SMILES to GNN Features...")

    data_list = []
    # all_symbols = []
    for smiles, cpd_name in zip(cur_smiles, cur_cpd_names):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        xs = []
        for atom in mol.GetAtoms():
            # all_symbols.append(atom.GetSymbol())
            x = []
            x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
            x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
            x.append(x_map['degree'].index(atom.GetTotalDegree()))
            x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
            x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
            x.append(x_map['num_radical_electrons'].index(
                atom.GetNumRadicalElectrons()))
            x.append(x_map['hybridization'].index(
                str(atom.GetHybridization())))
            x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
            x.append(x_map['is_in_ring'].index(atom.IsInRing()))
            xs.append(x)

        x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            e = []
            e.append(e_map['bond_type'].index(str(bond.GetBondType())))
            e.append(e_map['stereo'].index(str(bond.GetStereo())))
            e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

            edge_indices += [[i, j], [j, i]]
            edge_attrs += [e, e]

        edge_index = torch.tensor(edge_indices)
        edge_index = edge_index.t().to(torch.long).view(2, -1)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

        # Sort indices.
        if edge_index.numel() > 0:
            perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
            edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=cpd_name,
                    smiles=smiles)

        data = GenFeatures()(data)

        data_list.append(data)

    torch.save(InMemoryDataset.collate(data_list), path+"/GNN_Mol_Features.pt")

# data, slices = torch.load(path+"/GNN_Mol_Features.pt")
# temp[0]
# temp[1]
