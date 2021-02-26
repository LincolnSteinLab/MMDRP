# This file contains data loading classes for all DRP modules

# To install into conda environment, install pip in conda, then install using the /bin/ path for pip
import copy
import time
import numpy as np
import pandas as pd
import torch
from torch.utils import data


# def MorganFingerPrint(molecule, radius=2, nBits=2048):
#     try:
#         return AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=nBits)
#     except (ValueError, TypeError):
#         return None

file_name_dict = {"drug_file_name": "CTRP_AUC_MORGAN.hdf",
                  "mut_file_name": "DepMap_20Q2_CGC_Mutations_by_Cell.hdf",
                  "cnv_file_name": "DepMap_20Q2_CopyNumber.hdf",
                  "exp_file_name": "DepMap_20Q2_Expression.hdf",
                  "prot_file_name": "DepMap_20Q2_No_NA_ProteinQuant.hdf",
                  "tum_file_name": "DepMap_20Q2_Line_Info.csv",
                  "gdsc1_file_name": "GDSC1_AUC_MORGAN.hdf",
                  "gdsc2_file_name": "GDSC2_AUC_MORGAN.hdf",
                  "mut_embed_file_name": "optimal_autoencoders/MUT_Omic_AutoEncoder_Checkpoint",
                  "cnv_embed_file_name": "optimal_autoencoders/CNV_Omic_AutoEncoder_Checkpoint",
                  "exp_embed_file_name": "optimal_autoencoders/EXP_Omic_AutoEncoder_Checkpoint",
                  "prot_embed_file_name": "optimal_autoencoders/PROT_Omic_AutoEncoder_Checkpoint",
                  "drug_embed_file_name": "optimal_autoencoders/Morgan_4096_AutoEncoder_Checkpoint"}


class OmicData(data.Dataset):
    """
    General class that loads and holds omics data from one of mut, cnv, exp or prot.
    """

    def __init__(self, path, omic_file_name):
        self.path = path
        self.omic_file_name = omic_file_name
        self.full_train = pd.read_hdf(path + self.omic_file_name, 'df')

    def __len__(self):
        # Return the number of rows in the training data
        return self.full_train.shape[0]

    def width(self):
        # Exclude the key column
        return self.full_train.shape[1] - 1

    def __getitem__(self, idx):
        # Exclude the key column and return as a numpy array of type float
        return self.full_train.iloc[idx, 1:].to_numpy(dtype=float)

# path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
# morgan_file_name = "ChEMBL_Morgan_1024.pkl"
# model_type = "dnn"

class MorganData(data.Dataset):
    """
    Reads Morgan fingerprint data from its file and creates a Pandas data frame.
    """

    def __init__(self, path, morgan_file_name, model_type="dnn"):
        # Using 'self' makes these elements accessible after instantiation
        self.path = path
        self.morgan_file_name = morgan_file_name
        self.model_type = model_type
        # self.tum_file_name = tum_file_name
        # Read from file paths
        # self.tum_data = pd.read_csv(path+tum_file_name, engine='c', sep=',')
        # For expression data, the first column is the stripped_cell_line_name, which we don't need
        # tum_data = pd.read_csv(path+tum_file_name, engine='c', sep=',')
        self.morgan_train = list(pd.read_pickle(path + morgan_file_name)["morgan"])

        # Remove NoneType values (molecules whose Morgan fingerprints weren't calculated)
        # TODO: It is diffult to convert to numpy first
        cur_len = len(self.morgan_train)
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
        # tum_train = tum_data['primary_disease'].values
        # self.full_train = pd.merge(exp_train, self.tum_data[['DepMap_ID', 'primary_disease']], how='left', on='DepMap_ID')
        # Delete exp_train since it takes too much RAM
        # del exp_train
        # full_train = pd.merge(exp_train, tum_data[['DepMap_ID', 'primary_disease']], how='left', on='DepMap_ID')
        # self.tum_train = self.full_train['primary_disease'].values
        # Create one-hot labels for tumor classes, based on the order in the table above (for pairing with DepMap_ID)
        # self.le = LabelEncoder()
        # self.le.fit(self.tum_train)
        # self.tum_labels = self.le.transform(self.tum_train)

        # Now separate expression from class data, while discarding stripped cell line name and DepMap ID
        # self.full_train = self.full_train.iloc[:, 2:-1]
        # full_train = full_train.iloc[:, 2:-1]

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

# temp = MorganData(path,morgan_file_name, model_type="cnn")
# temp.width()
# len(temp)
# next(iter(temp))
# temp[0].shape

# def SmilesToMol(smiles, default=None):
#     try:
#         return Chem.MolFromSmiles(smiles)
#     except (ValueError, TypeError):
#         return default


class DRCurveData(data.Dataset):
    """
    Reads and prepares dose-response curves summarized by AUC value. Also grabs drug information as smiles and converts
    to a morgan fingerprint of desired size and radius using the RDKit package.
    """

    def __init__(self, path, file_name, key_column, morgan_column, target_column, cpd_name_column="cpd_name"):
        self.file_name = file_name
        self.key_column = key_column
        self.morgan_column = morgan_column
        self.target_column = target_column
        self.cpd_name_column = cpd_name_column

        # Read and subset the data
        all_data = pd.read_hdf(path + file_name, 'df')
        self.full_train = all_data[[self.key_column, self.cpd_name_column, self.morgan_column, self.target_column]]

    def __len__(self):
        return self.full_train.shape[0]

    def width(self):
        return len(self.full_train[self.morgan_column][0])

    def __getitem__(self, idx):
        fingerprint = self.full_train[self.morgan_column][idx]
        target = self.full_train[self.target_column][idx]

        return fingerprint, target


class PairData(data.Dataset):
    """
    Note: Assumes that drug dose-response data is provided and is at the first index of data_module_list
    Takes data.Dataset classes and pairs their data based on a column name(s)
    NOTE: This module assumes that there is only one omics datapoint for each datatype per sample! Otherwise the code
    will not work, and you might get a torch.stack error.
    It handles the fact that multiple drugs are tested on the same cell line, by holding a "relational database"
    structure, without duplicating the omics data.
    :param required_data_indices The indices of data from data_module_list that you want to be returned. This can be used to
    create complete samples from the given data, then return a subset of these complete samples based on the provided indices.
    This can be used when training is to be restricted to data that is available to all modules.
    """

    def __init__(self, data_module_list, key_columns, drug_index=None, drug_dr_column=None, required_data_indices=None,
                 test_mode=False):
        self.key_columns = key_columns
        assert drug_index == 0, "Drug data should be put first in data_module_list, and throughout training/testing"
        self.drug_index = drug_index
        self.drug_dr_column = drug_dr_column
        self.required_data_indices = required_data_indices
        self.test_mode = test_mode
        pandas = [getattr(data_module_list[i], "full_train") for i in range(len(data_module_list))]
        # We will use the key_column entities that are shared among all given datasets
        if len(list(key_columns)) == 1:
            # Convert keys (e.g. cell line names) to upper case and remove hyphens
            for i in range(len(pandas)):
                pandas[i][key_columns] = pandas[i][key_columns].str.upper()
                pandas[i][key_columns] = pandas[i][key_columns].str.replace('-', '')
            # If one column name is given, then it must be shared between all given datasets
            key_col_list = [list(panda[key_columns]) for panda in pandas]
        else:
            for i in range(len(pandas)):
                pandas[i][key_columns[i]] = pandas[i][key_columns[i]].str.upper()
                pandas[i][key_columns[i]] = pandas[i][key_columns[i]].str.replace('-', '')
            # If multiple column names are given, then each should match the data_module_list order
            key_col_list = [list(panda[key_columns[i]]) for panda, i in zip(pandas, range(len(key_columns)))]

        # Get the intersection of all keys
        self.key_col_sect = list(set.intersection(*map(set, key_col_list)))
        # print("Total overlapping keys:", str(len(self.key_col_sect)))

        # Subset all data based on these overlapping keys
        if len(list(key_columns)) == 1:
            pandas = [panda[panda[key_columns]].isin(self.key_col_sect) for panda in pandas]
        else:
            pandas = [panda[panda[key_columns[i]].isin(self.key_col_sect)] for panda, i in
                      zip(pandas, range(len(key_columns)))]

        self.cur_pandas = pandas

        if self.drug_index is not None:
            drug_data = pandas[drug_index]
            self.len_drug_data = drug_data.shape[0]
            self.morgan_col = data_module_list[drug_index].morgan_column
            self.cpd_name_column = data_module_list[drug_index].cpd_name_column
            self.drug_names = drug_data[self.cpd_name_column]

            # Get the index of the DR column for better pandas indexing later on
            if self.drug_dr_column is not None:
                self.drug_dr_col_idx = drug_data.columns.get_loc(self.drug_dr_column)
            else:
                # Assume it's in the 3rd position
                self.drug_dr_col_idx = 2

            self.drug_data_keys = drug_data[self.key_columns[self.drug_index]].to_numpy()
            self.drug_data_targets = drug_data.iloc[:, self.drug_dr_col_idx].to_numpy()
            self.drug_data_targets = np.array(self.drug_data_targets, dtype='float').reshape(
                self.drug_data_targets.shape[0], 1)

            # Convert smiles to finger prints once and use later on
            # Grab unique molecules and convert them to Morgan fingerprints with indicated radius and width using Pool
            self.drug_fps = drug_data[self.morgan_col].values
            self.morgan_width = len(self.drug_fps[0])

            # Convert fingerprints list to a numpy array prepared for training
            # print("Converting the fingerprints list to numpy array")
            start_time = time.time()
            self.drug_fps = [(np.fromstring(cur_drug, 'i1') - 48).astype('float') for cur_drug in self.drug_fps]
            self.drug_fps = np.vstack(self.drug_fps)

            # TODO: Understand why I moved cur_keys and cur_pandas outside and explain here
            # Must remove (using .drop) the drug key column (ccl_name) and the drug data from pandas using slicing
            # Note that if we use 'del', it will mutate the original list as well!
            self.cur_keys = self.key_columns[:self.drug_index] + self.key_columns[self.drug_index + 1:]
            final_pandas = pandas[:self.drug_index] + pandas[self.drug_index + 1:]

            # TODO How to take advantage of all cell lines, even those with replicates!
            # print("Creating dictionaries of omics data...")
            # Remove duplicated cell lines in each
            final_pandas = [panda.drop_duplicates(subset=self.cur_keys[i], keep="first") for panda, i in
                          zip(final_pandas, range(len(self.cur_keys)))]

            # Convert each df to a dictionary based on the key_column, making queries much faster
            self.dicts = [panda.set_index(self.cur_keys[i]).T.to_dict('list') for panda, i in
                          zip(final_pandas, range(len(self.cur_keys)))]

            # Convert all list values to flattened numpy arrays
            # print("Converting dictionary list elements to numpy arrays")
            for cur_dict in self.dicts:
                for key, value in cur_dict.items():
                    cur_dict[key] = np.array(cur_dict[key], dtype='float').flatten()

            # TODO
            # Except for drug data, combine other omic data types into a single np.array

    def __len__(self):
        if self.drug_index is not None:
            return self.len_drug_data
        else:
            return len(self.key_col_sect)

    def __getitem__(self, idx: int):
        # Get the cell line name for the given index
        cur_cell = self.drug_data_keys[idx]
        # Get the cell line omics data from non-drug pandas, and remove the key/cell line column for training
        cur_data = [cur_dict[cur_cell] for cur_dict in self.dicts]
        # Add fingerprint data to omics data
        cur_data = [self.drug_fps[idx, :]] + cur_data
        # Subset data based on requested data types (in the case of bottleneck selection)
        if self.required_data_indices is not None:
            cur_data = [cur_data[i] for i in self.required_data_indices]

        # Must return a dose-response summary if once is requested. It should be in the drug data
        # RETURNS: {cell_line_name, drug name}, (encoder data), dose-response target (AUC)
        if self.test_mode is False:
            return np.array([0]), tuple(cur_data), np.array(self.drug_data_targets[idx])
        else:
            return ({"cell_line_name": cur_cell,
                     "drug_name": self.drug_names.iloc[idx]},
                    tuple(encoder_data for encoder_data in cur_data),
                    self.drug_data_targets[idx])

    def get_subset(self, cell_lines: [str] = None, cpd_names: [str] = None, partial_match: bool = False,
                   min_target: float = None, max_target: float = None, make_main: bool = False):
        """
        This function will subset data by cell line, compound name, target (e.g. AUC) or a combination of these.

        :make_main: Change the default data to the subset held by this class.
        :return: A list of pandas that is subsetted by the given arguments.
        """
        pandas = copy.deepcopy(self.cur_pandas)
        if cell_lines is not None:
            # Subset by more cell lines, similar to the bottleneck condition
            if len(list(self.key_columns)) == 1:
                pandas = [panda[panda[self.key_columns]].isin(cell_lines) for panda in pandas]
            else:
                pandas = [panda[panda[self.key_columns[i]].isin(cell_lines)] for panda, i in
                          zip(pandas, range(len(self.key_columns)))]

        if cpd_names is not None:
            assert self.drug_index is not None, "No drug data in this object, cannot subset by compound names"
            # Subset drug data first, then use the remaining cell lines to subset other omics data
            # Note: Assuming that drug data is in the front of the list
            if partial_match is False:
                pandas[self.drug_index] = pandas[self.drug_index][pandas[self.drug_index][self.cpd_name_column].isin(cpd_names)]
            else:
                pandas[self.drug_index] = pandas[self.drug_index][pandas[self.drug_index][self.cpd_name_column].str.contains('|'.join(cpd_names)) == True]

            # Now subset other omics data based on the remaining cell lines
            remaining_cells = pandas[self.drug_index][self.key_columns[self.drug_index]]
            print("Number of cell lines in subset:", len(set(list(remaining_cells))))
            if len(list(self.key_columns)) == 1:
                pandas = [panda[panda[self.key_columns]].isin(remaining_cells) for panda in pandas]
            else:
                pandas = [panda[panda[self.key_columns[i]].isin(remaining_cells)] for panda, i in
                          zip(pandas, range(len(self.key_columns)))]
        if min_target is not None:
            assert self.drug_index is not None, "No drug data in this object, cannot subset by compound names"
            if len(list(self.key_columns)) == 1:
                pandas[self.drug_index] = pandas[self.drug_index][pandas[self.drug_index].iloc[:, self.drug_dr_col_idx] >= min_target]
            else:
                pandas[self.drug_index] = pandas[self.drug_index][pandas[self.drug_index].iloc[:, self.drug_dr_col_idx] >= min_target]
            remaining_cells = pandas[self.drug_index][self.key_columns[self.drug_index]]
            print("Number of cell lines in subset:", len(set(list(remaining_cells))))
            if len(list(self.key_columns)) == 1:
                pandas = [panda[panda[self.key_columns]].isin(remaining_cells) for panda in pandas]
            else:
                pandas = [panda[panda[self.key_columns[i]].isin(remaining_cells)] for panda, i in
                          zip(pandas, range(len(self.key_columns)))]
        if max_target is not None:
            assert self.drug_index is not None, "No drug data in this object, cannot subset by compound names"
            if len(list(self.key_columns)) == 1:
                pandas[self.drug_index] = pandas[self.drug_index][pandas[self.drug_index].iloc[:, self.drug_dr_col_idx] <= max_target]
            else:
                pandas[self.drug_index] = pandas[self.drug_index][pandas[self.drug_index].iloc[:, self.drug_dr_col_idx] <= max_target]
            remaining_cells = pandas[self.drug_index][self.key_columns[self.drug_index]]
            print("Number of cell lines in subset:", len(set(list(remaining_cells))))
            if len(list(self.key_columns)) == 1:
                pandas = [panda[panda[self.key_columns]].isin(remaining_cells) for panda in pandas]
            else:
                pandas = [panda[panda[self.key_columns[i]].isin(remaining_cells)] for panda, i in
                          zip(pandas, range(len(self.key_columns)))]

        if make_main is True:
            print("Changing internal data! Results achieved when reusing this new data may be different!")
            # Update available keys
            self.drug_data_keys = pandas[self.drug_index][self.key_columns[self.drug_index]].to_numpy()

            # Drop the "master key" column from data frames
            self.cur_keys = self.key_columns[:self.drug_index] + self.key_columns[self.drug_index + 1:]
            cur_pandas = pandas[:self.drug_index] + pandas[self.drug_index + 1:]

            # Update self.dicts
            self.dicts = [panda.set_index(self.cur_keys[i]).T.to_dict('list') for panda, i in
                          zip(cur_pandas, range(len(self.cur_keys)))]

            # Convert all list values to flattened numpy arrays. Doing this once saves time in __getitem__
            # print("Converting dictionary list elements to numpy arrays")
            for cur_dict in self.dicts:
                for key, value in cur_dict.items():
                    cur_dict[key] = np.array(cur_dict[key], dtype='float').flatten()

        return pandas


class PairDataIterator(object):
    """
    Takes data.Dataset classes and pairs their data based on a column name(s)
    NOTE: This module assumes that there is only one omics datapoint for each datatype per sample! Otherwise the code
    will not work, and you might get a torch.stack error.
    It handles the fact that multiple drugs are tested on the same cell line, by holding a "relational database"
    structure, without duplicating the omics data.
    :param required_data_indices The indices of data from data_module_list that you want to be returned. This can be used to
    create complete samples from the given data, then return a subset of these complete samples based on the provided indices.
    This can be used when training is to be restricted to data that is available to all modules.
    """

    def __init__(self, data_module_list, key_columns, drug_index=None, drug_dr_column=None, valid_size=0.1,
                 valid_mode=False,
                 required_data_indices=None, batch_size=32,
                 test_mode=False):
        # assert end > start, "this iterator only works with end >= start"
        # self.start, self.end = start, end
        self.key_columns = key_columns
        # TODO: Assess whether fixing drug data's position is beneficial
        assert drug_index == 0, "Drug data should be put first in data_module_list, and throughout training/testing"
        self.drug_index = drug_index
        self.drug_dr_column = drug_dr_column
        self.valid_size = valid_size
        self.valid_mode = valid_mode
        self.required_data_indices = required_data_indices
        self.batch_size = batch_size
        self.test_mode = test_mode
        self.pandas = [getattr(data_module_list[i], "full_train") for i in range(len(data_module_list))]
        self.len_datatypes = len(self.pandas)
        # We will use the key_column entities that are shared among all given datasets
        if len(list(key_columns)) == 1:
            # Convert keys (e.g. cell line names) to upper case and remove hyphens
            for i in range(len(self.pandas)):
                self.pandas[i][key_columns] = self.pandas[i][key_columns].str.upper()
                self.pandas[i][key_columns] = self.pandas[i][key_columns].str.replace('-', '')
            # If one column name is given, then it must be shared between all given datasets
            key_col_list = [list(panda[key_columns]) for panda in self.pandas]
        else:
            for i in range(len(self.pandas)):
                self.pandas[i][key_columns[i]] = self.pandas[i][key_columns[i]].str.upper()
                self.pandas[i][key_columns[i]] = self.pandas[i][key_columns[i]].str.replace('-', '')
            # If multiple column names are given, then each should match the data_module_list order
            key_col_list = [list(panda[key_columns[i]]) for panda, i in zip(self.pandas, range(len(key_columns)))]

        # Get the intersection of all keys
        self.key_col_sect = list(set.intersection(*map(set, key_col_list)))
        print("Total overlapping keys:", str(len(self.key_col_sect)))

        # Subset all data based on these overlapping keys
        if len(list(key_columns)) == 1:
            self.pandas = [panda[panda[key_columns]].isin(self.key_col_sect) for panda in self.pandas]
        else:
            self.pandas = [panda[panda[key_columns[i]].isin(self.key_col_sect)] for panda, i in
                           zip(self.pandas, range(len(key_columns)))]

        self.cur_pandas = self.pandas

        if self.drug_index is not None:
            self.drug_data = self.pandas[drug_index]
            # Split data (pseudo) randomly
            num_train = len(self.drug_data)
            split = int(np.floor(valid_size * num_train))
            # train_len = num_train - valid_len

            indices = list(range(num_train))
            np.random.seed(42)
            np.random.shuffle(indices)
            self.train_idx, self.valid_idx = indices[split:], indices[:split]

            # Subset drug data based on wheter we are in validation mode or not
            # TODO Create lenient vs strict validation schemes
            if self.valid_mode is True:
                self.drug_data = self.drug_data.iloc[self.valid_idx, :]
            else:
                self.drug_data = self.drug_data.iloc[self.train_idx, :]

            self.len_drug_data = self.drug_data.shape[0]
            self.morgan_col = data_module_list[drug_index].morgan_column
            self.cpd_name_column = data_module_list[drug_index].cpd_name_column
            self.drug_names = self.drug_data[self.cpd_name_column]

            # Get the index of the DR column for better pandas indexing later on
            if self.drug_dr_column is not None:
                self.drug_dr_col_idx = self.drug_data.columns.get_loc(self.drug_dr_column)

            self.drug_data_keys = self.drug_data[self.key_columns[self.drug_index]].to_numpy()
            self.drug_data_targets = self.drug_data.iloc[:, self.drug_dr_col_idx].to_numpy()
            self.drug_data_targets = np.array(self.drug_data_targets, dtype='float').reshape(
                self.drug_data_targets.shape[0], 1)

            # Convert smiles to finger prints once and use later on
            # Grab unique molecules and convert them to Morgan fingerprints with indicated radius and width using Pool
            self.drug_fps = self.drug_data[self.morgan_col].values

            # Convert fingerprints list to a numpy array prepared for training
            # print("Converting the fingerprints list to numpy array")
            start_time = time.time()
            self.drug_fps = [(np.fromstring(cur_drug, 'i1') - 48).astype('float') for cur_drug in self.drug_fps]
            self.drug_fps = np.vstack(self.drug_fps)
            # print("Done in:", time.time() - start_time)

            # TODO: Understand why I moved cur_keys and cur_pandas outside and explain here
            # Must remove (using .drop) the drug key column (ccl_name) and the drug data from pandas using slicing
            # Note that if we use 'del', it will mutate the original list as well!
            self.cur_keys = self.key_columns[:self.drug_index] + self.key_columns[self.drug_index + 1:]
            self.cur_pandas = self.pandas[:self.drug_index] + self.pandas[self.drug_index + 1:]

            # TODO How to take advantage of all cell lines, even those with replicates
            # print("Creating dictionaries of omics data...")
            # Remove duplicated cell lines in each
            self.cur_pandas = [panda.drop_duplicates(subset=self.cur_keys[i], keep="first") for panda, i in
                               zip(self.cur_pandas, range(len(self.cur_keys)))]
            # Convert each df to a dictionary based on the key_column, making queries much faster
            self.dicts = [panda.set_index(self.cur_keys[i]).T.to_dict('list') for panda, i in
                          zip(self.cur_pandas, range(len(self.cur_keys)))]

            # Convert all list values to flattened numpy arrays
            # print("Converting dictionary list elements to numpy arrays")
            for cur_dict in self.dicts:
                for key, value in cur_dict.items():
                    cur_dict[key] = torch.from_numpy(np.array(cur_dict[key], dtype='float').flatten())

            # TODO Use the streaming iterator tutorial to return items, otherwise creating an iter(list) object
            # TODo: will cause a bottleneck itself!

            # Must reduce DataSet size as it is instantiated on each worker
            # TODO: Don't need to have "self" if they are to be removed
            del self.pandas
            del self.cur_pandas
            del self.drug_data

    def size(self, ):
        return self.len_drug_data

    def __iter__(self):
        # Reset the index counter
        print("Reset the index counter...")
        self.i = 0
        # Shuffle the training data (or the order of cell lines in AUC data)
        if self.valid_mode is False:
            np.random.seed(42)
            np.random.shuffle(self.drug_data_keys)

        return self

    def __next__(self):
        batch = []
        targets = []

        for _ in range(self.batch_size):
            # Get the cell line name for the given index
            cur_cell = self.drug_data_keys[self.i]
            # Get the cell line omics data from non-drug pandas, and remove the key/cell line column for training
            cur_data = [cur_dict[cur_cell] for cur_dict in self.dicts]
            # cur_data = [np.array(cur_d, dtype='float').flatten() for cur_d in cur_data]

            # cur_data = [panda[panda[self.cur_keys[i]].isin([cur_cell])].drop(self.cur_keys[i], 1) for panda, i in zip(self.cur_pandas, range(len(self.cur_keys)))]
            # TODO: Set a condition for when there are multiple rows in a data type that match the key, the first match is returned only. Why not return all matches, but schedule for later?
            # Here we only have one cell line, and we assume that we have 1 sample. To make sure, subset the first row
            # cur_data = [panda.iloc[0, ] for panda in cur_data]

            cur_data = [self.drug_fps[self.i, :]] + cur_data

            # Subset data based on requested data types (in the case of bottleneck selection)
            if self.required_data_indices is not None:
                cur_data = [cur_data[i] for i in self.required_data_indices]

            batch.append(cur_data)
            targets.append(self.drug_data_targets[self.i])

            # Update self.i
            self.i += 1
            # Stop iteration if reached end of dataset
            if self.i >= self.len_drug_data:
                print("No more batches left in epoch!")
                raise StopIteration

        return batch, targets

    next = __next__


class AutoEncoderPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

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

    def next(self):
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

    def preload(self):
        try:
            _, self.next_input, self.next_target = next(self.loader)
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

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            for i in input:
                i.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())

        self.preload()
        return 0, input, target


# class ExternalSourcePipeline(Pipeline):
#     def __init__(self, batch_size, pdi, num_threads, device_id, cuda_stream):
#         super(ExternalSourcePipeline, self).__init__(batch_size,
#                                                      num_threads,
#                                                      device_id,
#                                                      prefetch_queue_depth=10,
#                                                      seed=12)
#         self.pdi = pdi
#         self.external_source_list = []
#         # Extra ExternalSource for the targets
#         for i in range(self.pdi.len_datatypes + 1):
#             self.external_source_list.append(ops.ExternalSource(device="gpu", cuda_stream=cuda_stream))
#         # Cast data to float32 on the GPU
#         self.cast = ops.Cast(device="gpu", dtype=types.FLOAT)
#         self.iterator = iter(self.pdi)
#
#     def define_graph(self):
#         # Directly transfer to GPU, and cast as float32
#         self.encoders_and_target = [ext_source() for ext_source in self.external_source_list]
#         casted_output = [self.cast(data_type) for data_type in self.encoders_and_target]
#
#         # Return as a whole list
#         return casted_output
#
#     def iter_setup(self):
#         # the external data iterator is consumed here and fed as input to Pipeline
#         try:
#             encoder_data, dr_target = next(self.pdi)
#
#             # TODO Must input a single numpy array, not a list.
#             # Must return the whole batch for each data type at once
#             # Use reverse zip, which returns another nested list but with items at the same index grouped together
#             unzipped_encoder_data = list(zip(*encoder_data))
#             for i in range(len(unzipped_encoder_data)):
#                 self.feed_input(self.encoders_and_target[i], list(unzipped_encoder_data[i]))
#
#             # We know that target info always comes last, and it's idx is equivalent with encoder_data length
#             self.feed_input(self.encoders_and_target[len(self.external_source_list) - 1], dr_target)
#
#         except StopIteration:
#             # raise exception when the iterator runs out of data and renew iterator
#             print("Finished iteration, resetting...")
#             self.iterator = iter(self.pdi)
#             raise StopIteration


