# This file contains data loading classes for all DRP modules

# To install into conda environment, install pip in conda, then install using the /bin/ path for pip
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils import data
import re

file_name_dict = {"drug_file_name": "CTRP_AAC_MORGAN.hdf",
                  "mut_file_name": "DepMap_20Q2_CGC_Mutations_by_Cell.hdf",
                  "cnv_file_name": "DepMap_20Q2_CopyNumber.hdf",
                  "exp_file_name": "DepMap_20Q2_Expression.hdf",
                  "prot_file_name": "DepMap_20Q2_No_NA_ProteinQuant.hdf",
                  "tum_file_name": "DepMap_20Q2_Line_Info.csv",
                  "gdsc1_file_name": "GDSC1_AAC_MORGAN.hdf",
                  "gdsc2_file_name": "GDSC2_AAC_MORGAN.hdf",
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
        # Exclude potential key columns
        return self.full_train.iloc[1, :].drop(["stripped_cell_line_name", "tcga_sample_id",
                                                          "DepMap_ID", "cancer_type"], errors='ignore').shape[0]

    def __getitem__(self, idx):
        # Exclude the key column and return as a numpy array of type float
        return self.full_train.iloc[idx, :].drop(["stripped_cell_line_name", "tcga_sample_id",
                                                          "DepMap_ID", "cancer_type"], errors='ignore').to_numpy(dtype=float)


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


class DRCurveData(data.Dataset):
    """
    Reads and prepares dose-response curves summarized by AAC value. Also grabs drug information as smiles and converts
    to a morgan fingerprint of desired size and radius using the RDKit package.
    """

    def __init__(self, path: str, file_name: str, key_column: str, morgan_column: str, class_column: str, target_column: str,
                 cpd_name_column: str = "cpd_name"):
        self.file_name = file_name
        self.key_column = key_column
        self.morgan_column = morgan_column
        self.target_column = target_column
        self.cpd_name_column = cpd_name_column
        self.class_column = class_column

        # Read and subset the data
        all_data = pd.read_hdf(path + file_name, 'df')
        self.full_train = all_data[[self.key_column, self.class_column, self.cpd_name_column, self.morgan_column, self.target_column]]

        cur_len = len(self.full_train.index)
        self.full_train = self.full_train[self.full_train[self.morgan_column].notnull()]
        # morgan_train = list(filter(None, morgan_train))
        # morgan_train = [list(i) for i in morgan_train]
        # morgan_train[0]
        print("Removed ", str(cur_len - len(self.full_train.index)), "NoneType values from data")

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

    def __init__(self, data_module_list, key_columns: [str], drug_index: int = 0, drug_dr_column=None,
                 test_mode=False, verbose: bool = False, bottleneck_keys: [str] = None):
        self.key_columns = key_columns
        assert drug_index == 0, "Drug data should be put first in data_module_list, and throughout training/testing"
        self.drug_index = drug_index
        self.drug_dr_column = drug_dr_column
        self.bottleneck = False
        if bottleneck_keys is not None:
            assert len(bottleneck_keys) > 0, "At least one key should be provided for the bottleneck mode"
            self.bottleneck = True
        self.bottleneck_keys = bottleneck_keys
        self.test_mode = test_mode

        pandas = [getattr(data_module_list[i], "full_train") for i in range(len(data_module_list))]
        # We will use the key_column entities that are shared among all given datasets
        # TODO having list() might result in a double list if input is a list
        if len(list(key_columns)) == 1:
            # Convert keys (e.g. cell line names) to upper case, remove dashes, slashes and spaces
            for i in range(len(pandas)):
                pandas[i][key_columns] = pandas[i][key_columns].str.replace('\W+', '')
                pandas[i][key_columns] = pandas[i][key_columns].str.upper()
            # If one column name is given, then all data sets should have this column
            key_col_list = [list(panda[key_columns]) for panda in pandas]
        else:
            for i in range(len(pandas)):
                pandas[i][key_columns[i]] = pandas[i][key_columns[i]].str.replace('\W+', '')
                pandas[i][key_columns[i]] = pandas[i][key_columns[i]].str.upper()
            # If multiple column names are given, then each should match the data_module_list order
            key_col_list = [list(panda[key_columns[i]]) for panda, i in zip(pandas, range(len(key_columns)))]

        # Get the intersection of all keys
        self.key_col_sect = list(set.intersection(*map(set, key_col_list)))

        if self.bottleneck:
            self.key_col_sect = list(set.intersection(set(self.key_col_sect), set(bottleneck_keys)))
            if verbose:
                print("Restricting keys to those that overlap with the provided bottleneck keys")

        if verbose:
            print("Total overlapping keys:", str(len(self.key_col_sect)))

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

            if verbose:
                print("Converting drug data keys and targets to numpy arrays")
            self.drug_data_keys = drug_data[self.key_columns[self.drug_index]].to_numpy()
            self.drug_data_targets = drug_data.iloc[:, self.drug_dr_col_idx].to_numpy()
            self.drug_data_targets = np.array(self.drug_data_targets, dtype='float').reshape(
                self.drug_data_targets.shape[0], 1)

            self.drug_fps = drug_data[self.morgan_col].values
            self.morgan_width = len(self.drug_fps[0])

            if verbose:
                print("Converting fingerprints from str list to float numpy array")
            # Convert fingerprints list to a numpy array prepared for training
            self.drug_fps = [(np.fromstring(cur_drug, 'i1') - 48).astype('float') for cur_drug in self.drug_fps]
            self.drug_fps = np.vstack(self.drug_fps)

            # TODO: Understand why I moved cur_keys and cur_pandas outside and explain here
            # Must remove (using .drop) the drug key column (ccl_name) and the drug data from pandas using slicing
            # Note that if we use 'del', it will mutate the original list as well!
            if verbose:
                print("Separating drug data from omics data")
            self.cur_keys = self.key_columns[:self.drug_index] + self.key_columns[self.drug_index + 1:]
            final_pandas = pandas[:self.drug_index] + pandas[self.drug_index + 1:]

            # TODO How to take advantage of all cell lines, even those with replicates?! Perhaps use an ID other than ccl_name that is more unique to each replicate
            # Remove duplicated cell lines in each
            if verbose:
                print("Removing cell line replicate data (!), keeping first match")
            final_pandas = [panda.drop_duplicates(subset=self.cur_keys[i], keep="first") for panda, i in
                            zip(final_pandas, range(len(self.cur_keys)))]

            if verbose:
                print("Converting pandas dataframes to dictionaries, where key is the cell line name")
            # Convert each df to a dictionary based on the key_column, making queries much faster
            self.dicts = [panda.set_index(self.cur_keys[i]).T.to_dict('list') for panda, i in
                          zip(final_pandas, range(len(self.cur_keys)))]

            if verbose:
                print("Converting lists in the dictionary to numpy arrays")
            # Convert all list values to flattened numpy arrays
            for cur_dict in self.dicts:
                for key, value in cur_dict.items():
                    cur_dict[key] = np.array(cur_dict[key], dtype='float').flatten()

            # TODO
            # Except for drug data, combine other omic data types into a single np.array

    def __len__(self):
        # NOTE: It is crucial that this truly reflects the number of samples in the current dataset
        # Pytorch's data.DataLoader determines the number of batches based on this length. So having a smaller length
        # results in some data to never be shown to the model!
        if self.drug_index is not None:
            return self.len_drug_data
        else:
            # Number of overlapping cell lines among selected data types
            return len(self.key_col_sect)

    def __getitem__(self, idx: int):
        # Get the cell line name for the given index
        cur_cell = self.drug_data_keys[idx]
        # Get the cell line omics data from non-drug pandas, and remove the key/cell line column for training
        cur_data = [cur_dict[cur_cell] for cur_dict in self.dicts]
        # Add fingerprint data to omics data
        cur_data = [self.drug_fps[idx, :]] + cur_data

        # Must return a dose-response summary if once is requested. It should be in the drug data
        # RETURNS: {cell_line_name, drug name}, (encoder data), dose-response target (AAC)
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
        This function will subset data by cell line, compound name, target (e.g. AAC) or a combination of these.

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
                pandas[self.drug_index] = pandas[self.drug_index][
                    pandas[self.drug_index][self.cpd_name_column].isin(cpd_names)]
            else:
                pandas[self.drug_index] = pandas[self.drug_index][
                    pandas[self.drug_index][self.cpd_name_column].str.contains('|'.join(cpd_names)) == True]

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
                pandas[self.drug_index] = pandas[self.drug_index][
                    pandas[self.drug_index].iloc[:, self.drug_dr_col_idx] >= min_target]
            else:
                pandas[self.drug_index] = pandas[self.drug_index][
                    pandas[self.drug_index].iloc[:, self.drug_dr_col_idx] >= min_target]
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
                pandas[self.drug_index] = pandas[self.drug_index][
                    pandas[self.drug_index].iloc[:, self.drug_dr_col_idx] <= max_target]
            else:
                pandas[self.drug_index] = pandas[self.drug_index][
                    pandas[self.drug_index].iloc[:, self.drug_dr_col_idx] <= max_target]
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
