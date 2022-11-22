# This file contains data loading classes for all DRP modules

# To install into conda environment, install pip in conda, then install using the /bin/ path for pip
import copy
import random
from collections import Counter
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from scipy.ndimage import convolve1d
from sklearn.feature_selection import f_regression, SelectKBest
from torch.utils import data
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.datasets.molecule_net import x_map, e_map

from CustomFunctions import produce_amount_keys
from label_smoothing import get_bin_idx, get_lds_kernel_window
from torch_geometric.data import InMemoryDataset, Data


class OmicData(data.Dataset):
    """
    General class that loads and holds omics data from one of mut, cnv, exp or prot.
    """

    def __init__(self, path, omic_file_name, to_gpu: bool = False, verbose: bool = False):
        self.path = path
        self.omic_file_name = omic_file_name
        self.all_data = pd.read_hdf(path + self.omic_file_name, 'df')
        # Separate data info columns
        possible_info_columns_set = {"stripped_cell_line_name", "tcga_sample_id", "DepMap_ID",
                                     "cancer_type", "primary_disease", "lineage", "lineage_subtype"}
        cur_info_columns = list(
            possible_info_columns_set & set(self.all_data.columns))
        self.data_info = self.all_data[cur_info_columns]
        self.full_train = self.all_data.drop(list(possible_info_columns_set),
                                             axis=1, errors='ignore')
        self.column_names = self.full_train.columns.to_list()
        self.full_train = self.full_train.to_numpy(dtype=float)
        self.full_train = torch.from_numpy(self.full_train)
        self.full_train = self.full_train.float()

        if to_gpu is True:
            # Moving to the GPU makes training about 10x faster on V100, as it avoids RAM to VRAM transfers
            if verbose is True:
                print("Moving Omic data to GPU")
            self.full_train = self.full_train.cuda()
        else:
            if verbose is True:
                print("Keeping data on RAM")

        if verbose is True:
            print("Data length:", len(self),
                  "\nData width:", self.width())

    def __len__(self):
        # Return the number of rows in the training data
        return self.full_train.shape[0]

    def width(self):
        return self.full_train[0].shape[0]

    def standardize(self, train_idx=None):
        if train_idx is not None:
            print("Standardizing omics data based on training data, assuming training and validation are distinct...")
            cur_means = torch.mean(self.full_train[train_idx], 0, True)
            cur_stds = torch.std(self.full_train[train_idx], 0, True)
        else:
            print("Standardizing all data! (no training/validation separation)")
            cur_means = torch.mean(self.full_train, 0, True)
            cur_stds = torch.std(self.full_train, 0, True)

        self.full_train = (self.full_train - cur_means) / (cur_stds + 1e-6)

    def __getitem__(self, idx):
        # Exclude the key column and return as a numpy array of type float
        return self.full_train[idx]


class MorganData(data.Dataset):
    """
    Reads Morgan fingerprint data from its file and creates a Pandas data frame.
    """

    def __init__(self, path, morgan_file_name, model_type="dnn"):
        # Using 'self' makes these elements accessible after instantiation
        self.path = path
        self.morgan_file_name = morgan_file_name
        self.model_type = model_type
        self.morgan_train = list(pd.read_pickle(path + morgan_file_name)["morgan"])

        # Remove NoneType values (molecules whose Morgan fingerprints weren't calculated)
        # TODO: It is difficult to convert to numpy first, but may reduce computation during training
        # TODO: Use PairData approach
        cur_len = len(self.morgan_train)
        # TODO SETUP to_GPU function
        self.morgan_train = list(filter(None, self.morgan_train))
        # morgan_train = list(filter(None, morgan_train))
        # morgan_train = [list(i) for i in morgan_train]
        # morgan_train[0]
        print("Removed ", str(cur_len - len(self.morgan_train)), "NoneType values from data")
        # Convert to numpy array once
        # self.morgan_train = np.array(self.morgan_train, dtype="int")
        # print("shape:", self.morgan_train.shape)
        # if self.model_type == "cnn":
        # Add the channel dimension
        # self.morgan_train.reshape(shape=)

    def __len__(self):
        # Return the length of the list
        # return len(self.morgan_train[:, 0, ...])
        return len(self.morgan_train)

    def width(self):
        # Return the length of the first element in the list
        return len(self.morgan_train[0])
        # return len(self.morgan_train[0, ...])

    def __getitem__(self, idx):
        # skip_idx = np.delete(np.arange(0, 1165), idx)
        # cur_exp = pd.read_csv(self.path+self.exp_file_name, engine='c', sep=',',
        #                       skiprows=skip_idx.tolist()).iloc[:, 1:].values
        # Return paired expression and classification data
        # TODO: This may be an expensive operation
        if self.model_type == "cnn":
            return np.array(list(self.morgan_train[idx]), dtype=int).reshape((1, self.width()))
        else:
            return np.array(list(self.morgan_train[idx]), dtype=int)
        # return self.morgan_train[idx, ...]


class GenFeatures(object):
    def __init__(self):
        # self.symbols = [
        #     'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br',
        #     'Te', 'I', 'At', 'other'
        # ]

        self.symbols = [
            'As', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'Pt', 'S', 'other'
        ]
        self.hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
            'other',
        ]

        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]

    def __call__(self, data):
        # Generate AttentiveFP features according to Table 1.
        mol = Chem.MolFromSmiles(data.smiles)

        xs = []
        for atom in mol.GetAtoms():
            # Create 1-hot encoding
            symbol = [0.] * len(self.symbols)
            # try:
            symbol[self.symbols.index(atom.GetSymbol())] = 1.
            # except:
            #     print(data.smiles)
            #     temp = [atom.GetSymbol() for atom in mol.GetAtoms()]
            #     print(temp)
            #     exit(0)

            degree = [0.] * 6
            degree[atom.GetDegree()] = 1.
            formal_charge = atom.GetFormalCharge()
            radical_electrons = atom.GetNumRadicalElectrons()
            hybridization = [0.] * len(self.hybridizations)
            hybridization[self.hybridizations.index(
                atom.GetHybridization())] = 1.
            aromaticity = 1. if atom.GetIsAromatic() else 0.
            hydrogens = [0.] * 5
            hydrogens[atom.GetTotalNumHs()] = 1.
            chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
            chirality_type = [0.] * 2
            if atom.HasProp('_CIPCode'):
                chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.

            x = torch.tensor(symbol + degree + [formal_charge] +
                             [radical_electrons] + hybridization +
                             [aromaticity] + hydrogens + [chirality] +
                             chirality_type)
            xs.append(x)

        data.x = torch.stack(xs, dim=0)

        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
            conjugation = 1. if bond.GetIsConjugated() else 0.
            ring = 1. if bond.IsInRing() else 0.
            stereo = [0.] * 4
            stereo[self.stereos.index(bond.GetStereo())] = 1.

            edge_attr = torch.tensor(
                [single, double, triple, aromatic, conjugation, ring] + stereo)

            edge_attrs += [edge_attr, edge_attr]

        if len(edge_attrs) == 0:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(edge_indices).t().contiguous()
            data.edge_attr = torch.stack(edge_attrs, dim=0)

        return data


class MyGNNData(Data):
    def __cat_dim__(self, key, item, *args):
        if key in ['omic_data', 'target', 'loss_weight']:
            return -1
        else:
            return super().__cat_dim__(key, item)


# import torch_geometric
# foo_1 = torch.randn(1006)
# foo_2 = torch.randn(1009)
#
# edge_index = torch.tensor([
#    [0, 1, 1, 2],
#    [1, 0, 2, 1],
# ])
# data = MyGNNData(edge_index=edge_index)
# data.omic_data = [foo_1, foo_2]
#
# data_list = [data, data]
# loader = torch_geometric.data.DataLoader(data_list, batch_size=4)
# batch = next(iter(loader))
# batch.omic_data
# len(batch.omic_data[0])
# batch.omic_data.shape[0]
#
# t1 = batch.omic_data
# len_enc = len(batch.omic_data[0])
# len_batch = len(batch.omic_data)
# cur_list = []
# for i in range(len_enc):
#     cur_list.append(torch.vstack([t1[j][i] for j in range(len_batch)]))
#
# batch.omic_data.shape[0]
# torch.cat(batch.omic_data, dim=1)
#
#
#
# print(batch)
#
# MyGNNData(omic_data=torch)
# PATH = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
# train_file = "CTRP_AAC_SMILES.txt"
# cur_data = GnnDRCurveData(path=PATH, file_name=train_file, key_column="ccl_name", class_column="primary_disease",
#                        target_column="area_above_curve", to_gpu=False, gnn_pre_transform = GenFeatures(),
#                        transform=None)
# temp = cur_data.drug_feat_dict[cur_data.drug_names.to_numpy()[30001]]
# temp.x
# temp.edge_index
# temp.edge_attr
# temp.batch
# temp.omic_data = torch.Tensor([1,2,3])
# temp.list_data = [[1,2,3], [1321, 1231]]
# temp.__cat_dim__ = None
# cur_data.drug_names.to_numpy()

class GnnDRCurveData(data.Dataset):
    """
    Args:
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, path: str, file_name: str, key_column: str = "ccl_name", smiles_column: str = "cpd_smiles",
                 class_column: str = "primary_disease", target_column: str = "area_above_curve", transform: str = None,
                 cpd_name_column: str = "cpd_name", random_morgan: bool = False, to_gpu: bool = False,
                 verbose: bool = False, name: str = "GNNData", gnn_pre_transform=None):

        if Chem is None:
            raise ImportError('`MoleculeNet` requires `rdkit`.')

        self.name = name.lower()
        # super(GnnDRCurveData, self).__init__(path + file_name, transform, pre_transform,
        #                                      pre_filter)
        super(GnnDRCurveData, self).__init__()
        self.file_name = file_name
        self.smiles_column = smiles_column
        self.key_column = key_column
        self.target_column = target_column
        self.cpd_name_column = cpd_name_column
        self.class_column = class_column
        self.gnn_pre_transform = gnn_pre_transform
        self.verbose = verbose

        if transform is not None:
            assert transform in ["log", "sqrt"], "Available DR transformation methods are log and sqrt"
        self.transformation = transform

        # Read and subset the data

        all_data = pd.read_csv(path + file_name, engine='c', sep=',')
        self.all_data = all_data[[self.key_column, self.class_column, self.cpd_name_column, self.smiles_column,
                                  self.target_column]]
        # self.all_data = all_data[["ccl_name", "primary_disease", "cpd_name", "cpd_smiles", "area_above_curve"]]
        # key_column = "ccl_name"
        # class_column = "primary_disease"
        # cpd_name_column = "cpd_name"
        # morgan_column = "morgan"
        # target_column = "area_above_curve"
        # all_data = all_data[[key_column, class_column, cpd_name_column, morgan_column,
        #                             target_column]]

        # cur_len = len(self.all_data.index)
        # self.all_data = all_data[all_data[morgan_column].notnull()]

        self.preprocess()

        if self.transformation is not None:
            self.transform()

        # Process and make drug:feature dict
        self.process()

        # Features must be in the same order as in the full_train DataFrame
        # self.gnn_feat_df = pd.DataFrame.from_dict(self.data, dtype=object, orient='index')
        # self.gnn_feat_df.reset_index(level=0, inplace=True)
        # self.gnn_feat_df.columns = [self.cpd_name_column, "edge_attr", "edge_index", "smiles", "x"]
        #
        # self.all_data.merge(self.gnn_feat_df, on=self.cpd_name_column, how='right')
        #
        # gnn_feat_list = self.all_data[["edge_attr", "edge_index", "smiles", "x"]].to_records(index=False).tolist()
        #
        # self.gnn_drug_data, self.slices = InMemoryDataset.collate(gnn_feat_list)

        if to_gpu is True:
            self.to_gpu()

        print("Removed ", set(self.drug_names.drop_duplicates().tolist()) ^ set(list(self.drug_feat_dict.keys())),
              "drugs from data")

    def preprocess(self):
        self.full_train = self.all_data[[self.smiles_column, self.target_column]]
        self.data_info = self.all_data[[self.key_column, self.class_column, self.cpd_name_column, self.smiles_column]]
        self.drug_names = self.data_info[self.cpd_name_column]

        # Get the index of the DR column for better pandas indexing later on
        self.drug_dr_col_idx = self.full_train.columns.get_loc(self.target_column)

        if self.verbose:
            print("Converting drug data targets to float32 torch tensors")

        # self.drug_data_keys = self.data_info[key_column].to_numpy()
        drug_data_targets = self.full_train[self.target_column].to_numpy()
        drug_data_targets = np.array(drug_data_targets, dtype='float').reshape(
            drug_data_targets.shape[0], 1)
        self.drug_data_targets: torch.FloatTensor = torch.from_numpy(drug_data_targets).float()

    def transform(self):
        if self.transformation == 'log':
            print("Log transforming DR data...")
            # self.drug_data_targets = np.log(self.drug_data_targets + 1)
            self.full_train[self.target_column] = np.log(self.full_train[self.target_column] + 1)
        else:
            print("Square Root transforming DR data...")
            # self.drug_data_targets = np.sqrt(self.drug_data_targets)
            self.full_train[self.target_column] = np.sqrt(self.full_train[self.target_column])

    def to_gpu(self):
        self.drug_data_targets: torch.FloatTensor = self.drug_data_targets.cuda()
        for key in self.drug_feat_dict:
            self.drug_feat_dict[key] = self.drug_feat_dict[key].cuda()

    def process(self):

        smiles_and_names = self.all_data[[self.smiles_column, self.cpd_name_column]].drop_duplicates()
        all_smiles = smiles_and_names[self.smiles_column].tolist()
        # cur_targets = self.full_train[self.target_column].tolist()
        all_cpd_names = smiles_and_names[self.cpd_name_column].tolist()

        print("Converting SMILES to GNN Features...")

        data_dict = {}
        for cur_smiles, cur_cpd_name in zip(all_smiles, all_cpd_names):
            mol = Chem.MolFromSmiles(cur_smiles)

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

            data = MyGNNData(x=x, edge_index=edge_index, edge_attr=edge_attr,
                             smiles=cur_smiles)

            if self.gnn_pre_transform:
                data = self.gnn_pre_transform(data)

            data_dict[cur_cpd_name] = data
            # data_list.append(data)

        self.drug_feat_dict = data_dict
        # return InMemoryDataset.collate(data_list)
        # torch.save(InMemoryDataset.collate(data_list), path+"/GNN_Mol_Features.pt")


class DRCurveData(data.Dataset):
    """
    Reads and prepares dose-response curves summarized by AAC value and drug information as morgan fingerprint.
    TODO Should the Morgan part of this class use the Morgan Class?
    """

    def __init__(self, path: str, file_name: str, key_column: str = "ccl_name", morgan_column: str = "morgan",
                 smiles_column: str = "cpd_smiles", class_column: str = "primary_disease",
                 target_column: str = "area_above_curve", transform: str = None,
                 cpd_name_column: str = "cpd_name", random_morgan: bool = False, to_gpu: bool = False,
                 verbose: bool = False):
        self.file_name = file_name
        self.key_column = key_column
        self.morgan_column = morgan_column
        self.target_column = target_column
        self.cpd_name_column = cpd_name_column
        self.class_column = class_column
        self.random_morgan = random_morgan
        self.smiles_column = smiles_column
        if transform is not None:
            assert transform in ["log", "sqrt"], "Available DR transformation methods are log and sqrt"
        self.transform = transform

        # Read and subset the data
        all_data = pd.read_hdf(path + file_name, 'df')
        self.all_data = all_data[
            [self.key_column, self.class_column, self.cpd_name_column, self.smiles_column, self.morgan_column,
             self.target_column]]
        # key_column = "ccl_name"
        # class_column = "primary_disease"
        # cpd_name_column = "cpd_name"
        # morgan_column = "morgan"
        # target_column = "area_above_curve"
        # all_data = all_data[[key_column, class_column, cpd_name_column, morgan_column,
        #                             target_column]]

        cur_len = len(self.all_data.index)
        self.all_data = all_data[all_data[morgan_column].notnull()]

        self.full_train = self.all_data[[morgan_column, target_column]]
        self.data_info = self.all_data[[key_column, class_column, cpd_name_column, self.smiles_column]]
        self.drug_names = self.data_info[cpd_name_column]

        print("Removed ", str(cur_len - len(self.full_train.index)), "NoneType values from data")

        # Get the index of the DR column for better pandas indexing later on
        self.drug_dr_col_idx = self.full_train.columns.get_loc(target_column)

        if verbose:
            print("Converting drug data targets to float32 torch tensors")
        self.drug_data_keys = self.data_info[key_column].to_numpy()
        drug_data_targets = self.full_train[target_column].to_numpy()
        drug_data_targets = np.array(drug_data_targets, dtype='float').reshape(
            drug_data_targets.shape[0], 1)
        self.drug_data_targets: torch.FloatTensor = torch.from_numpy(drug_data_targets).float()

        if verbose:
            print("Converting fingerprints from str list to float32 torch tensor")

        if random_morgan is True:
            print("Creating random morgan fingerprints")
            drug_fps = list(produce_amount_keys(500, length=len(self.full_train[self.morgan_column][0])))
            drug_fps = random.choices(drug_fps, k=self.full_train.shape[0])
        else:
            drug_fps = self.full_train[morgan_column].values

        # Convert fingerprints list to a numpy array prepared for training
        drug_fps = [torch.from_numpy((np.fromstring(cur_drug, 'i1') - 48).astype('float')).float() for cur_drug in
                    drug_fps]

        self.drug_feat_dict: Dict = dict(zip(self.drug_names.tolist(), drug_fps))

        # drug_fps = np.vstack(drug_fps)
        # self.drug_fps: torch.FloatTensor = torch.from_numpy(drug_fps).float()

        if transform is not None:
            # Must transform the target data
            if transform == 'log':
                print("Log transforming DR data...")
                self.drug_data_targets = np.log(self.drug_data_targets + 1)
            else:
                print("Square Root transforming DR data...")
                self.drug_data_targets = np.sqrt(self.drug_data_targets)

        if to_gpu is True:
            self.drug_data_targets: torch.FloatTensor = self.drug_data_targets.cuda()
            for key in self.drug_feat_dict:
                self.drug_feat_dict[key] = self.drug_feat_dict[key].cuda()

        print("Removed ", set(self.drug_names.drop_duplicates().tolist()) ^ set(list(self.drug_feat_dict.keys())),
              "drugs from data")

    def __len__(self):
        return self.full_train.shape[0]

    def width(self):
        return len(self.full_train[self.morgan_column][0])

    def __getitem__(self, idx):
        fingerprint = self.drug_fps[idx]
        target = self.drug_data_targets[idx]

        return fingerprint, target


# ctrp_data = DRCurveData(path="/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/",
#                         file_name="CTRP_AAC_MORGAN_512.hdf")
# ctrp_data.all_data.to_csv("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/CTRP_AAC_MORGAN_512.csv")
class PairData(data.Dataset):
    """
    Note: Assumes that drug dose-response data is provided and is at the first index of data_list
    Takes data.Dataset classes and pairs their data based on a column name(s)
    NOTE: This module assumes that there is only one omics datapoint for each datatype per sample! Otherwise the code
    will not work, and you might get a torch.stack error.
    It handles the fact that multiple drugs are tested on the same cell line, by holding a "relational database"
    structure, without duplicating the omics data.
    :param required_data_indices The indices of data from data_list that you want to be returned. This can be used to
    create complete samples from the given data, then return a subset of these complete samples based on the provided indices.
    This can be used when training is to be restricted to data that is available to all modules.
    """

    def __init__(self, data_list: List, key_columns: List[str], key_attribute_names: List[str],
                 data_attribute_names: List[str] = ["full_train"], drug_index: int = 0, drug_dr_column=None,
                 one_hot_drugs: bool = False, gnn_mode: bool = False, omic_standardize: bool = False,
                 mode: str = 'train', verbose: bool = False, bottleneck_keys: List[str] = None,
                 to_gpu: bool = False):
        """

        :param data_list:
        :param key_columns: The name of the ID column for each data type. Can be either a list with a single element
        if the same name is used for all data types, or should have the same length as the number of data types supplied
        :param key_attribute_names: Name of the attr of the data object that has key/ID information.
        :param data_attribute_names: Name of the attr of the data object that has key/ID information.
        :param drug_index:
        :param drug_dr_column:
        :param infer_mode: whether drug and cell line names for the current batch is also returned
        :param verbose:
        :param bottleneck_keys:
        :param to_gpu:
        """
        self.omic_standardize = omic_standardize
        self.gnn_mode = gnn_mode
        self.key_columns = key_columns
        self.one_hot_drugs = one_hot_drugs
        self.mode = mode
        assert self.mode in ['train', 'infer', 'interpret'], "Mode must be one of train, infer, or interpret!"
        if type(data_attribute_names) != list:
            data_attribute_names = list(data_attribute_names)
        if len(data_attribute_names) == 1:
            data_attribute_names = data_attribute_names * len(data_list)

        if type(key_attribute_names) != list:
            key_attribute_names = list(key_attribute_names)
        if len(key_attribute_names) == 1:
            key_attribute_names = key_attribute_names * len(data_list)

        if type(key_columns) != list:
            key_columns = list(key_columns)
        if len(key_columns) == 1:
            key_columns = key_columns * len(data_list)
        self.key_columns = key_columns

        assert drug_index == 0, "Drug data should be put first in data_list, and throughout training/testing"
        self.drug_index = drug_index
        self.drug_dr_column = drug_dr_column
        self.bottleneck = False
        if bottleneck_keys is not None:
            assert len(bottleneck_keys) > 0, "At least one key should be provided for the bottleneck mode"
            self.bottleneck = True
        self.bottleneck_keys = bottleneck_keys
        self.verbose = verbose
        self.to_gpu = to_gpu
        self.data_list = data_list
        self.data_attribute_names = data_attribute_names
        self.key_attribute_names = key_attribute_names

        if self.omic_standardize is True:
            self.standardize()

        # Pair the given data
        self.pair_data()

    def pair_data(self):
        # Get data tensors from each data type object
        all_data_tensors = [getattr(self.data_list[i], self.data_attribute_names[i]) for i in
                            range(len(self.data_list))]

        # Get data info for each object
        self.data_infos = [getattr(self.data_list[i], self.key_attribute_names[i]) for i in range(len(self.data_list))]

        # Duplicate for later use
        # original_data_infos = copy.deepcopy(data_infos)

        # Identify shared keys among different data types
        # TODO having list() might result in a double list if input is a list
        for i in range(len(self.data_infos)):
            # Only keep English alphabet characters from key names, convert to upper case
            self.data_infos[i][self.key_columns[i]] = self.data_infos[i][self.key_columns[i]].str.replace('\W+', '')
            self.data_infos[i][self.key_columns[i]] = self.data_infos[i][self.key_columns[i]].str.upper()
        # If multiple column names are given, then each should match the data_list order
        key_col_list = [list(panda[self.key_columns[i]]) for panda, i in
                        zip(self.data_infos, range(len(self.key_columns)))]

        # Get the intersection of all keys
        self.key_col_sect = list(set.intersection(*map(set, key_col_list)))

        # Subset cell lines with list of cell lines given to be used as data bottleneck
        if self.bottleneck:
            self.key_col_sect = list(set.intersection(set(self.key_col_sect), set(self.bottleneck_keys)))
            if self.verbose:
                print("Restricting keys to those that overlap with the provided bottleneck keys")

        if self.verbose:
            print("Total overlapping keys:", str(len(self.key_col_sect)))

        # Concatenate omic data info and tensors + get omic data column names (e.g. gene names)
        if self.drug_index is not None:
            # Ignore the drug data which is the first element
            omic_pandas = [pd.concat([cur_data_info[self.key_columns[i]],
                                      pd.DataFrame(cur_data_tensor.cpu().numpy())],
                                     axis=1) for i, cur_data_info, cur_data_tensor in
                           zip(range(1, len(self.data_infos)), self.data_infos[1:], all_data_tensors[1:])]
            cur_keys = self.key_columns[1:]
            self.omic_column_names = [self.data_list[i].column_names for i in range(1, len(self.data_list))]
        else:
            omic_pandas = [pd.concat([cur_data_info[self.key_columns[i]],
                                      pd.DataFrame(cur_data_tensor.numpy())],
                                     axis=1) for i, cur_data_info, cur_data_tensor in
                           zip(range(len(self.data_infos)), self.data_infos, all_data_tensors)]
            cur_keys = self.key_columns
            self.omic_column_names = [self.data_list[i].column_names for i in range(len(self.data_list))]

        # TODO How to take advantage of all cell lines, even those with replicates?! Perhaps use an ID other than ccl_name that is more unique to each replicate
        # Remove duplicated cell lines in each
        if self.verbose:
            print("Removing cell line replicate data (!), keeping first match")
        self.omic_pandas = [panda.drop_duplicates(subset=cur_keys[i], keep="first") for panda, i in
                            zip(omic_pandas, range(len(cur_keys)))]

        if self.verbose:
            print("Converting pandas dataframes to dictionaries, where key is the cell line name")
        # Convert each df to a dictionary based on the key_column, making queries much faster
        dicts = [panda.set_index(cur_keys[i]).T.to_dict('list') for panda, i in
                 zip(self.omic_pandas, range(len(cur_keys)))]

        if self.verbose:
            print("Converting lists in the dictionary to float32 torch tensors")
        # Subset dict by overlapping keys
        # for cur_dict in dicts:
        #     cur_dict = {k: cur_dict[k] for k in self.key_col_sect}
        dicts = [{k: cur_dict[k] for k in self.key_col_sect} for cur_dict in dicts]

        # Convert all list values to flattened numpy arrays, then to torch tensors
        if self.to_gpu is True:
            for cur_dict in dicts:
                for key, value in cur_dict.items():
                    # cur_dict[key] = torch.from_numpy(np.array(cur_dict[key], dtype='float').flatten()).float().cuda()
                    cur_dict[key] = torch.tensor(cur_dict[key]).cuda()
        else:
            for cur_dict in dicts:
                for key, value in cur_dict.items():
                    # cur_dict[key] = torch.from_numpy(np.array(cur_dict[key], dtype='float').flatten()).float()
                    cur_dict[key] = torch.tensor(cur_dict[key])

        self.dicts = dicts

        # Subset data infos to keys that overlap (this will be used in CV fold creation)
        self.data_infos = [panda[panda[self.key_columns[i]].isin(self.key_col_sect)] for panda, i in
                           zip(self.data_infos, range(len(self.key_columns)))]

        # Process drug data
        if self.drug_index is not None:
            drug_data = self.data_list[self.drug_index]
            drug_data_targets = drug_data.drug_data_targets
            # Subset dose-response data based on overlapping keys (e.g. cell lines)
            overlap_idx = np.where(drug_data.data_info[self.key_columns[self.drug_index]].isin(self.key_col_sect))[0]
            # TODO Are drug_data_targets and drug_data_keys in the same order?
            self.drug_data_targets = drug_data_targets[overlap_idx]
            self.drug_data_keys = drug_data.data_info[self.key_columns[self.drug_index]][
                drug_data.data_info[self.key_columns[self.drug_index]].isin(self.key_col_sect)].to_list()

            self.cpd_name_column = drug_data.cpd_name_column
            self.drug_names: np.ndarray = drug_data.data_info[self.cpd_name_column].iloc[
                overlap_idx].to_numpy().reshape(-1, 1)

            # Get the index of the DR column for better pandas indexing later on
            self.drug_dr_col_idx = drug_data.drug_dr_col_idx

            if self.one_hot_drugs is False:
                # Subset drug info by
                self.drug_feat_dict = drug_data.drug_feat_dict

                if self.gnn_mode is False:
                    # print("Drug type is:", type(self.drug_feat_dict[list(self.drug_feat_dict.keys())[0]]))
                    assert isinstance(self.drug_feat_dict[list(self.drug_feat_dict.keys())[0]], torch.Tensor), \
                        "Drug dictionary values have incorrect format!"
                    self.drug_input_width = drug_data.width()
                # self.drug_input = drug_fps[overlap_idx]

            else:
                raise NotImplementedError
                # use one-hot encoded drug labels instead of fingerprints
                ohe = OneHotEncoder()
                # X = [['Male'], ['Female'], ['Female']]
                ohe.fit(self.drug_names)
                print("# of one-hot drug categories", len(ohe.categories_[0]))
                self.drug_input: torch.FloatTensor = torch.from_numpy(ohe.transform(self.drug_names).toarray()).float()

            if self.to_gpu is True:
                self.drug_data_targets = self.drug_data_targets.cuda(non_blocking=True)
                for key in self.drug_feat_dict:
                    self.drug_feat_dict[key] = self.drug_feat_dict[key].cuda(non_blocking=True)

            # print("self.drug_input shape:", self.drug_input.shape)
            # self.drug_input_width = self.drug_input.shape[1]

            # Reconvert drug_names back to list for compatibilty
            self.drug_names: List = self.drug_names.tolist()
            # Unflatten list
            self.drug_names = [x[0] for x in self.drug_names]

            # Instantiate loss weights to 1 (equal losses)
            self.loss_weights = torch.Tensor([1] * len(self.drug_data_targets))
            self.loss_weights = self.loss_weights.reshape(len(self.loss_weights), 1)

            if self.to_gpu is True:
                self.loss_weights = self.loss_weights.cuda(non_blocking=True)

    def standardize(self, train_idx=None):
        print("Standardizing omics data...")
        if train_idx is None:
            print("Assuming training and validation cell lines are already separated!")
        # Must only standardize omics data and not drug data
        if self.drug_index is not None:
            k = 1
        else:
            k = 0
        # Get data tensors from each data type object
        all_data_tensors = [getattr(self.data_list[i], self.data_attribute_names[i]) for i in
                            range(k, len(self.data_list))]
        # Move all to CPU
        all_data_tensors = [cur_tensor.cpu() for cur_tensor in all_data_tensors]

        all_train_means = []
        all_train_stds = []

        if train_idx is not None:
            print("Getting standardization statistics from training data...")
            all_train_cells = [self.drug_data_keys[idx] for idx in train_idx]
            # Not the most efficient way of doing this...
            for cur_dict in self.dicts:
                for key, value in cur_dict.items():
                    cur_dict[key] = torch.tensor(cur_dict[key]).cpu()

            for cur_dict in self.dicts:
                cur_train_omic_data = [cur_dict[cur_cell] for cur_cell in all_train_cells]
                cur_train_omic_data = torch.vstack(cur_train_omic_data)
                # Get train data mean and stds for transformation, move stats to cpu
                cur_means = torch.mean(cur_train_omic_data, 0, True).cpu()
                cur_stds = torch.std(cur_train_omic_data, 0, True).cpu()
                # Append standardized data as well as transformation parameters to use on validation data
                all_train_means.append(cur_means)
                all_train_stds.append(cur_stds)

        else:
            Warning("Getting standardization statistics from all data, not just training data!")
            # Will standardize all data, disregarding training/validation separation
            for cur_tensor in all_data_tensors:
                all_train_means.append(torch.mean(cur_tensor, 0, True))
                all_train_stds.append(torch.std(cur_tensor, 0, True))

        # Standardize (Z-score normalization)
        all_standardized_tensors = []
        for cur_tensor, cur_means, cur_stds in zip(all_data_tensors, all_train_means, all_train_stds):
            all_standardized_tensors.append((cur_tensor - cur_means) / (cur_stds + 1e-6))

        # Now update internal class data
        for i, j in zip(range(len(all_standardized_tensors)), range(k, len(self.data_list))):
            setattr(self.data_list[j], self.data_attribute_names[j], all_standardized_tensors[i])

        # Pair original data
        self.pair_data()

    def __len__(self):
        # NOTE: It is crucial that this truly reflects the number of samples in the current dataset
        # Pytorch's data.DataLoader determines the number of batches based on this length. So having a smaller length
        # results in some data to never be shown to the model!
        if self.drug_index is not None:
            return len(self.drug_data_keys)
        else:
            # Number of overlapping cell lines among selected data types
            return len(self.key_col_sect)

    def __getitem__(self, idx: int):
        # Get the cell line name for the given index
        cur_cell = self.drug_data_keys[idx]
        cur_cpd = self.drug_names[idx]

        # Get the cell line omics data from omic dicts
        cur_data = [cur_dict[cur_cell] for cur_dict in self.dicts]
        # Get fingerprint data to omics data
        cur_drug_feat = self.drug_feat_dict[cur_cpd]

        if self.gnn_mode is True:
            # cur_drug_feat is a pytorch geometric Data object, must decorate with omics data
            # cur_drug_feat.omic_data = cur_data
            # cur_drug_feat.target = self.drug_data_targets[idx]
            # cur_drug_feat.loss_weight = self.loss_weights[idx]

            if self.mode in ['train', 'infer']:
                # return cur_drug_feat, cur_data, self.drug_data_targets[idx], self.loss_weights[idx]
                # elif self.mode == 'infer':
                return {"cell_line_name": cur_cell,
                        "drug_name": self.drug_names[idx]}, \
                       cur_drug_feat, cur_data, self.drug_data_targets[idx], self.loss_weights[idx]

            else:
                return ({"cell_line_name": cur_cell,
                         "drug_name": self.drug_names[idx],
                         "smiles": cur_drug_feat.smiles},
                        [cur_drug_feat.x, cur_drug_feat.edge_index,
                         cur_drug_feat.edge_attr],
                        cur_data, self.drug_data_targets[idx], self.loss_weights[idx])

        else:
            # cur_drug_feat is a Morgan fingerprint, simply add to omic data list
            final_data = [cur_drug_feat] + cur_data

            # Must return a dose-response summary if once is requested. It should be in the drug data
            # RETURNS: {cell_line_name, drug name}, (encoder data), dose-response target (AAC)
            if self.mode in ['train', 'infer']:
                # return torch.Tensor([0]), tuple(final_data), self.drug_data_targets[idx], self.loss_weights[idx]
                # else:
                return {"cell_line_name": cur_cell,
                        "drug_name": self.drug_names[idx]}, \
                       tuple(final_data), self.drug_data_targets[idx], self.loss_weights[idx]

    def subset_cells(self, cell_lines: [str] = None):
        if len(list(self.key_columns)) == 1:
            # self.data_infos[i][self.key_columns[i]]
            for i in range(1, len(self.data_list)):
                self.data_list[i].all_data = self.data_list[1].all_data[self.data_list[1].all_data[self.key_columns[1]].isin(cell_lines)]
        else:
            for i in range(1, len(self.data_list)):
                self.data_list[i].all_data = self.data_list[i].all_data[self.data_list[i].all_data[self.key_columns[i]].isin(cell_lines)]

        # Now subset drugs based on remaining cells
        remaining_cpds = self.data_list[self.drug_index].all_data[self.key_columns[self.drug_index]]
        self.subset_cpds(cpd_names=remaining_cpds)

    def subset_cpds(self, cpd_names: [str] = None, partial_match: bool = False):
        assert self.drug_index is not None, "No drug data in this object, cannot subset by compound names"
        print("Subsetting drug compounds to:", cpd_names)

        if partial_match is False:
            self.data_list[self.drug_index].all_data = self.data_list[self.drug_index].all_data[
                self.data_list[self.drug_index].all_data[self.data_list[self.drug_index].cpd_name_column].isin(cpd_names)]
        else:
            self.data_list[self.drug_index].all_data = self.data_list[self.drug_index].all_data[
                self.data_list[self.drug_index].all_data[self.data_list[self.drug_index].cpd_name_column].str.contains('|'.join(cpd_names)) is True]

        self.data_list[self.drug_index].preprocess()
        if self.data_list[self.drug_index].transformation is not None:
            self.data_list[self.drug_index].transform()

        self.data_list[self.drug_index].process()

        # Now subset other omics data based on the remaining cell lines
        remaining_cells = self.data_list[self.drug_index].all_data[self.key_columns[self.drug_index]]
        self.subset_cells(cell_lines=remaining_cells)

        # print("Number of cell lines in subset:", len(set(list(remaining_cells))))
        # if len(list(self.key_columns)) == 1:
        #     self.data_list = [panda[panda[self.key_columns]].isin(remaining_cells) for panda in self.data_list]
        # else:
        #     self.data_list = [panda[panda[self.key_columns[i]].isin(remaining_cells)] for panda, i in
        #                       zip(self.data_list, range(len(self.key_columns)))]

    def subset(self, cell_lines: [str] = None, cpd_names: [str] = None, partial_match: bool = False,
               min_target: float = None, max_target: float = None, make_main: bool = False):
        """
        This function will subset data by cell line, compound name, target (e.g. AAC) or a combination of these.

        :make_main: Change the default data to the subset made by this function.
        :return: A list of pandas that is subsetted by the given arguments.
        """
        # pandas = copy.deepcopy(self.omic_pandas)
        # Get data tensors from each data type object

        if cell_lines is not None:
            # exit("Cell line subsetting not yet implemented")
            # assert self.drug_index is not None, "No drug data in this object, cannot subset by compound names"
            print("Subsetting cell lines to:", cell_lines)

            # Assuming that drug data is in the front of the list!!!
            if len(list(self.key_columns)) == 1:
                # self.data_infos[i][self.key_columns[i]]
                for i in range(1, len(self.data_list)):
                    self.data_list[i].all_data = self.data_list[1].all_data[self.data_list[1].all_data[self.key_columns[1]].isin(cell_lines)]
            else:
                for i in range(1, len(self.data_list)):
                    self.data_list[i].all_data = self.data_list[i].all_data[self.data_list[i].all_data[self.key_columns[i]].isin(cell_lines)]

                # self.data_list[1:] = [data_info[data_info[self.key_columns[i]].isin(cell_lines)] for data_info, i in
                #                   zip(self.data_infos[1:], range(1, len(self.key_columns)))]

            # for i in range(1, len(self.data_list)):
                # self.data_list[i].preprocess()
                # if self.data_list[i].transformation is not None:
                #     self.data_list[i].transform()
                # self.data_list[i].standardize()

        if cpd_names is not None:
            # exit("Compound name subsetting not yet implemented")
            assert self.drug_index is not None, "No drug data in this object, cannot subset by compound names"
            print("Subsetting drug compounds to:", cpd_names)

            if partial_match is False:
                self.data_list[self.drug_index].all_data = self.data_list[self.drug_index].all_data[
                    self.data_list[self.drug_index].all_data[self.data_list[self.drug_index].cpd_name_column].isin(cpd_names)]
            else:
                self.data_list[self.drug_index].all_data = self.data_list[self.drug_index].all_data[
                    self.data_list[self.drug_index].all_data[self.data_list[self.drug_index].cpd_name_column].str.contains('|'.join(cpd_names)) is True]

            self.data_list[self.drug_index].preprocess()
            if self.data_list[self.drug_index].transformation is not None:
                self.data_list[self.drug_index].transform()

            self.data_list[self.drug_index].process()
            # Now subset other omics data based on the remaining cell lines
            remaining_cells = self.data_list[self.drug_index][self.key_columns[self.drug_index]]
            # print("Number of cell lines in subset:", len(set(list(remaining_cells))))
            if len(list(self.key_columns)) == 1:
                self.data_list = [panda[panda[self.key_columns]].isin(remaining_cells) for panda in self.data_list]
            else:
                self.data_list = [panda[panda[self.key_columns[i]].isin(remaining_cells)] for panda, i in
                                  zip(self.data_list, range(len(self.key_columns)))]

        if min_target is not None:
            # raise NotImplementedError
            assert self.drug_index is not None, "No drug data in this object, cannot subset by target values"
            print("Subsetting AAC with minimum of", min_target)
            # Update the drug info attribute of data_list
            self.data_list[self.drug_index].all_data = self.data_list[self.drug_index].all_data[
                self.data_list[self.drug_index].all_data[self.data_list[self.drug_index].target_column] >= min_target]

            self.data_list[self.drug_index].preprocess()
            if self.data_list[self.drug_index].transformation is not None:
                self.data_list[self.drug_index].transform()

            self.data_list[self.drug_index].process()

            #
            # subset_bool = self.data_list[self.drug_index].drug_data_targets.ge(min_target)
            # subset_idx = subset_bool.nonzero()[:, 0]
            # subset_bool = subset_bool.numpy()
            #
            # new_drug_info = getattr(self.data_list[self.drug_index], self.key_attribute_names[self.drug_index])[
            #     subset_bool]
            # setattr(self.data_list[self.drug_index], self.key_attribute_names[self.drug_index], new_drug_info)
            #
            # # Update drug data targets and fingerprints
            # self.data_list[self.drug_index].drug_data_targets = self.data_list[self.drug_index].drug_data_targets[
            #     subset_idx]
            # self.data_list[self.drug_index].drug_feat_dict = s-elf.data_list[self.drug_index].drug_feat_dict.index_select(0,
            #                                                                                                  subset_idx)

            # if len(list(self.key_columns)) == 1:
            #     self.data_list[self.drug_index] = self.data_list[self.drug_index][
            #         self.data_list[self.drug_index].iloc[:, self.drug_dr_col_idx] >= min_target]
            # else:
            #     self.data_list[self.drug_index] = self.data_list[self.drug_index][
            #         self.data_list[self.drug_index].iloc[:, self.drug_dr_col_idx] >= min_target]
            # remaining_cells = self.data_list[self.drug_index][self.key_columns[self.drug_index]]
            # print("Number of cell lines in subset:", len(set(list(remaining_cells))))
            # if len(list(self.key_columns)) == 1:
            #     self.data_list = [panda[panda[self.key_columns]].isin(remaining_cells) for panda in self.data_list]
            # else:
            #     self.data_list = [panda[panda[self.key_columns[i]].isin(remaining_cells)] for panda, i in
            #               zip(self.data_list, range(len(self.key_columns)))]

        if max_target is not None:
            # raise NotImplementedError
            assert self.drug_index is not None, "No drug data in this object, cannot subset by target values"
            # Update the drug info attribute of data_list
            subset_bool = self.data_list[self.drug_index].drug_data_targets.le(max_target)
            subset_idx = subset_bool.nonzero()[:, 0]
            subset_bool = subset_bool.numpy()

            new_drug_info = getattr(self.data_list[self.drug_index], self.key_attribute_names[self.drug_index])[
                subset_bool]
            setattr(self.data_list[self.drug_index], self.key_attribute_names[self.drug_index], new_drug_info)

            # Update drug data targets and fingerprints
            self.data_list[self.drug_index].drug_data_targets = self.data_list[self.drug_index].drug_data_targets[
                subset_idx]
            self.data_list[self.drug_index].drug_fps = self.data_list[self.drug_index].drug_fps.index_select(0,
                                                                                                             subset_idx)
        print("Changing internal data! Results achieved when reusing this new data may be different!")
        self.pair_data()

        # if len(list(self.key_columns)) == 1:
        #     pandas[self.drug_index] = pandas[self.drug_index][
        #         pandas[self.drug_index].iloc[:, self.drug_dr_col_idx] <= max_target]
        # else:
        #     pandas[self.drug_index] = pandas[self.drug_index][
        #         pandas[self.drug_index].iloc[:, self.drug_dr_col_idx] <= max_target]
        # remaining_cells = pandas[self.drug_index][self.key_columns[self.drug_index]]
        # print("Number of cell lines in subset:", len(set(list(remaining_cells))))
        # if len(list(self.key_columns)) == 1:
        #     pandas = [panda[panda[self.key_columns]].isin(remaining_cells) for panda in pandas]
        # else:
        #     pandas = [panda[panda[self.key_columns[i]].isin(remaining_cells)] for panda, i in
        #               zip(pandas, range(len(self.key_columns)))]

        # if make_main is True:

        # # Update available keys
        # self.drug_data_keys = pandas[self.drug_index][self.key_columns[self.drug_index]].to_numpy()
        #
        # # Drop the "master key" column from data frames
        # self.cur_keys = self.key_columns[:self.drug_index] + self.key_columns[self.drug_index + 1:]
        # cur_pandas = pandas[:self.drug_index] + pandas[self.drug_index + 1:]
        #
        # # Update self.dicts
        # self.dicts = [panda.set_index(self.cur_keys[i]).T.to_dict('list') for panda, i in
        #               zip(cur_pandas, range(len(self.cur_keys)))]
        #
        # # Convert all list values to flattened numpy arrays. Doing this once saves time in __getitem__
        # # print("Converting dictionary list elements to numpy arrays")
        # for cur_dict in self.dicts:
        #     for key, value in cur_dict.items():
        #         cur_dict[key] = np.array(cur_dict[key], dtype='float').flatten()

        # return pandas

    # def label_dist_smoothing(self):
    #     eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    #     weights = [np.float32(1 / x) for x in eff_num_per_label]
    #
    #     # calculate loss
    #     loss = weighted_mse_loss(preds, labels, weights=weights)

    def label_dist_smoothing(self, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        # Delving into Deep Imbalanced Regression (https://arxiv.org/abs/2102.09554)
        # assert reweight in {'none', 'inverse', 'sqrt_inv'}
        # assert reweight != 'none' if lds else True, \
        #     "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        print("Re-weighting loss using LDS...")
        # assign each label to its corresponding bin (start from 0)
        # with your defined get_bin_idx(), return bin_index_per_label: [Ns,]
        bin_index_per_label = [get_bin_idx(label[0]) for label in self.drug_data_targets.tolist()]

        # calculate empirical (original) label distribution: [Nb,]
        # "Nb" is the number of bins
        Nb = max(bin_index_per_label) + 1
        num_samples_of_bins = dict(Counter(bin_index_per_label))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

        # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
        lds_kernel_window = get_lds_kernel_window(kernel=lds_kernel, ks=lds_ks, sigma=lds_sigma)
        # calculate effective label distribution: [Nb,]
        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
        # self.eff_label_dist = eff_label_dist

        eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
        # Calculate inverse weights and scale
        self.loss_weights = [np.float32(1 / x) for x in eff_num_per_label]
        self.scaling = len(self.loss_weights) / np.sum(self.loss_weights)
        # TODO is it safe to scale the losses higher?
        # self.scaling *= 100
        self.loss_weights = [weights * self.scaling for weights in self.loss_weights]

        # Convert to Tensor, change shape
        self.loss_weights = torch.Tensor(self.loss_weights)
        self.loss_weights = self.loss_weights.reshape(len(self.loss_weights), 1)

        if self.to_gpu is True:
            self.loss_weights = self.loss_weights.cuda()

        # calculate loss
        # loss = weighted_mse_loss(preds, labels, weights=weights)
        #
        # value_dict = {x: 0 for x in range(max_target)}
        # labels = self.df['age'].values
        # for label in labels:
        #
        #     value_dict[min(max_target - 1, int(label))] += 1
        # if reweight == 'sqrt_inv':
        #     value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        # elif reweight == 'inverse':
        #     value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        # num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        # if not len(num_per_label) or reweight == 'none':
        #     return None
        # print(f"Using re-weighting: [{reweight.upper()}]")
        #
        # if lds:
        #     lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        #     print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        #     smoothed_value = convolve1d(
        #         np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
        #     num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]
        #
        # weights = [np.float32(1 / x) for x in num_per_label]
        # scaling = len(weights) / np.sum(weights)
        # weights = [scaling * x for x in weights]
        # return weights

    def feature_selection(self, train_idx, method=f_regression, k: int = 1000):
        """
        Select features using sklearn's SelectKBest, subsets omic (exp) data only
        :param method: method to use, default: f_regression
        :param k: number of features to select
        :return: None
        """
        # Get data tensors from each data type object
        all_data_tensors = [getattr(self.data_list[i], self.data_attribute_names[i]) for i in
                            range(k, len(self.data_list))]
        # Move all to CPU
        all_data_tensors = [cur_tensor.cpu().numpy() for cur_tensor in all_data_tensors]

        print("Using training data for feature selection")
        # Get all training cells
        all_train_cells = [self.drug_data_keys[idx] for idx in train_idx]
        # Not the most efficient way of doing this...
        for cur_dict in self.dicts:
            for key, value in cur_dict.items():
                cur_dict[key] = torch.tensor(cur_dict[key]).cpu().numpy()
        all_train_targets = [self.drug_data_targets[idx].numpy() for idx in train_idx]
        all_train_targets = np.array(all_train_targets)

        all_feature_selectors = []
        for cur_dict in self.dicts:
            cur_train_omic_data = [cur_dict[cur_cell] for cur_cell in all_train_cells]
            cur_train_omic_data = np.vstack(cur_train_omic_data)

            # define feature selection
            fs = SelectKBest(score_func=method, k=k)
            # apply feature selection
            fs.fit(cur_train_omic_data, all_train_targets)
            all_feature_selectors.append(fs)

        # Standardize (Z-score normalization)
        all_shrunken_tensors = []
        for cur_tensor, cur_fs in zip(all_data_tensors, all_feature_selectors):
            all_shrunken_tensors.append(cur_fs.transform(cur_tensor))

        # Now update internal class data
        for i, j in zip(range(len(all_shrunken_tensors)), range(k, len(self.data_list))):
            print("Shape of data number", i, ":", all_shrunken_tensors[i].shape)
            setattr(self.data_list[j], self.data_attribute_names[j], torch.from_numpy(all_shrunken_tensors[i]))

        # Pair original data
        self.pair_data()


class AutoEncoderPrefetcher():
    """
    This class creates a stream from a data.DataSet object and loads the next iteration
    on the cuda stream on the GPU. Basically, preloading data on the GPU.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def __iter__(self):
        return self

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            # Convert to float then transfer to GPU
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.cuda(non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        if input is not None:
            input.record_stream(torch.cuda.current_stream())

        self.preload()
        return input


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def __iter__(self):
        return self

    def preload(self):
        try:
            self.next_info, self.next_input, self.next_target, self.next_weight = next(self.loader)
        except StopIteration:

            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            # Convert to float then transfer to GPU non-blocking-ly
            self.next_input = [enc_data.float() for enc_data in self.next_input]
            self.next_target = self.next_target.float()

            self.next_input = [enc_data.cuda(non_blocking=True) for enc_data in self.next_input]
            self.next_target = self.next_target.cuda(non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        weight = self.next_weight

        if input is not None:
            for i in input:
                i.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if weight is not None:
            weight.record_stream(torch.cuda.current_stream())

        self.preload()
        return 0, input, target, weight


class GNNDataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def __iter__(self):
        return self

    def preload(self):
        try:
            self.next_gnn_data = next(self.loader)
        except StopIteration:
            self.next_gnn_data = None
            return
        with torch.cuda.stream(self.stream):
            self.next_gnn_data = self.next_gnn_data.cuda(non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        gnn_data = self.next_gnn_data
        if gnn_data is not None:
            gnn_data.record_stream(torch.cuda.current_stream())

        self.preload()
        return gnn_data

# import os.path as osp
# from math import sqrt
#
# import torch
# import torch.nn.functional as F
# from rdkit import Chem
#
# from torch_geometric.data import DataLoader, InMemoryDataset
# from torch_geometric.datasets import MoleculeNet
# from torch_geometric.nn.models import AttentiveFP
# #
# # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'AFP_Mol')
# dataset = MoleculeNet("/Users/ftaj/Downloads/", name='ESOL', pre_transform=GenFeatures()).shuffle()
# # #
# dataset[0]
# dataset[1]
# dataset[2]
# dataset[2]
# dataset[500]
# dataset[1000]

# # N = len(dataset) // 10
# # val_dataset = dataset[:N]
# # test_dataset = dataset[N:2 * N]
# # train_dataset = dataset[2 * N:]
# #
# train_loader = DataLoader(dataset, batch_size=200, shuffle=True)
# temp1 = iter(train_loader)
# x = next(temp1)
# x.batch.shape

# val_loader = DataLoader(val_dataset, batch_size=200)
# test_loader = DataLoader(test_dataset, batch_size=200)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = AttentiveFP(in_channels=39, hidden_channels=200, out_channels=1,
#                     edge_dim=10, num_layers=2, num_timesteps=2,
#                     dropout=0.2).to(device)
#
# optimizer = torch.optim.Adam(model.parameters(), lr=10**-2.5,
#                              weight_decay=10**-5)
#
#
# def train():
#     total_loss = total_examples = 0
#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index, data.edge_attr, data.batch)
#         loss = F.mse_loss(out, data.y)
#         loss.backward()
#         optimizer.step()
#         total_loss += float(loss) * data.num_graphs
#         total_examples += data.num_graphs
#     return sqrt(total_loss / total_examples)
#
#
# @torch.no_grad()
# def test(loader):
#     mse = []
#     for data in loader:
#         data = data.to(device)
#         out = model(data.x, data.edge_index, data.edge_attr, data.batch)
#         mse.append(F.mse_loss(out, data.y, reduction='none').cpu())
#     return float(torch.cat(mse, dim=0).mean().sqrt())
#
#
# for epoch in range(1, 201):
#     train_rmse = train()
#     val_rmse = test(val_loader)
#     test_rmse = test(test_loader)
#     print(f'Epoch: {epoch:03d}, Loss: {train_rmse:.4f} Val: {val_rmse:.4f} '
#           f'Test: {test_rmse:.4f}')
