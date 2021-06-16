# This file contains data loading classes for all DRP modules

# To install into conda environment, install pip in conda, then install using the /bin/ path for pip
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils import data


class OmicData(data.Dataset):
    """
    General class that loads and holds omics data from one of mut, cnv, exp or prot.
    """

    def __init__(self, path, omic_file_name, to_gpu=False):
        self.path = path
        self.omic_file_name = omic_file_name
        self.all_data = pd.read_hdf(path + self.omic_file_name, 'df')
        # Separate data info columns
        info_columns = list(
            {"stripped_cell_line_name", "tcga_sample_id", "DepMap_ID", "cancer_type", "primary_disease"} & set(self.all_data.columns))
        self.data_info = self.all_data[info_columns]
        self.full_train = self.all_data.drop(["stripped_cell_line_name", "tcga_sample_id", "primary_disease",
                                              "DepMap_ID", "cancer_type"],
                                             axis=1, errors='ignore')
        self.column_names = self.full_train.columns.to_list()
        self.full_train = self.full_train.to_numpy(dtype=float)
        self.full_train = torch.from_numpy(self.full_train)
        self.full_train = self.full_train.float()
        if to_gpu is True:
            # Moving to the GPU makes training about 10x faster on V100, as it avoids RAM to VRAM transfers
            self.full_train = self.full_train.cuda()

    def __len__(self):
        # Return the number of rows in the training data
        return self.full_train.shape[0]

    def width(self):
        return self.full_train[0].shape[0]

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


class DRCurveData(data.Dataset):
    """
    Reads and prepares dose-response curves summarized by AAC value and drug information as morgan fingerprint.
    TODO Should the Morgan part of this class use the Morgan Class?
    """

    def __init__(self, path: str, file_name: str, key_column: str = "ccl_name", morgan_column: str = "morgan",
                 class_column: str = "primary_disease", target_column: str = "area_above_curve",
                 cpd_name_column: str = "cpd_name", to_gpu: bool = False, verbose: bool = False):
        self.file_name = file_name
        self.key_column = key_column
        self.morgan_column = morgan_column
        self.target_column = target_column
        self.cpd_name_column = cpd_name_column
        self.class_column = class_column

        # Read and subset the data
        all_data = pd.read_hdf(path + file_name, 'df')
        self.all_data = all_data[[self.key_column, self.class_column, self.cpd_name_column, self.morgan_column,
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
        self.data_info = self.all_data[[key_column, class_column, cpd_name_column]]
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
        self.drug_data_targets = torch.from_numpy(drug_data_targets).float()

        if verbose:
            print("Converting fingerprints from str list to float32 torch tensor")
        # Convert fingerprints list to a numpy array prepared for training
        drug_fps = self.full_train[morgan_column].values
        drug_fps = [(np.fromstring(cur_drug, 'i1') - 48).astype('float') for cur_drug in drug_fps]
        drug_fps = np.vstack(drug_fps)
        self.drug_fps = torch.from_numpy(drug_fps).float()

        if to_gpu is True:
            self.drug_data_targets = self.drug_data_targets.cuda()
            self.drug_fps = self.drug_fps.cuda()

    def __len__(self):
        return self.full_train.shape[0]

    def width(self):
        return len(self.full_train[self.morgan_column][0])

    def __getitem__(self, idx):
        fingerprint = self.drug_fps[idx]
        target = self.drug_data_targets[idx]

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

    def __init__(self, data_module_list, key_columns: [str], key_attribute_names: [str],
                 data_attribute_names: [str] = ["full_train"], drug_index: int = 0, drug_dr_column=None,
                 test_mode=False, verbose: bool = False, bottleneck_keys: [str] = None, to_gpu: bool = False):
        """

        :param data_module_list:
        :param key_columns: The name of the ID column for each data type. Can be either a list with a single element
        if the same name is used for all data types, or should have the same length as the number of data types supplied
        :param key_attribute_names: Name of the attr of the data object that has key/ID information.
        :param data_attribute_names: Name of the attr of the data object that has key/ID information.
        :param drug_index:
        :param drug_dr_column:
        :param test_mode: whether drug and cell line names for the current batch is also returned
        :param verbose:
        :param bottleneck_keys:
        :param to_gpu:
        """
        self.key_columns = key_columns
        if type(data_attribute_names) != list:
            data_attribute_names = list(data_attribute_names)
        if len(list(data_attribute_names)) == 1:
            data_attribute_names = data_attribute_names * len(data_module_list)

        if type(key_attribute_names) != list:
            key_attribute_names = list(key_attribute_names)
        if len(list(key_attribute_names)) == 1:
            key_attribute_names = key_attribute_names * len(data_module_list)

        if type(key_columns) != list:
            key_columns = list(key_columns)
        if len(list(key_columns)) == 1:
            key_columns = key_columns * len(data_module_list)

        assert drug_index == 0, "Drug data should be put first in data_module_list, and throughout training/testing"
        self.drug_index = drug_index
        self.drug_dr_column = drug_dr_column
        self.bottleneck = False
        if bottleneck_keys is not None:
            assert len(bottleneck_keys) > 0, "At least one key should be provided for the bottleneck mode"
            self.bottleneck = True
        self.bottleneck_keys = bottleneck_keys
        self.test_mode = test_mode

        # Get data tensors from each data type object
        all_data_tensors = [getattr(data_module_list[i], data_attribute_names[i]) for i in
                             range(len(data_module_list))]

        # Get data info for each object
        self.data_infos = [getattr(data_module_list[i], key_attribute_names[i]) for i in range(len(data_module_list))]

        # Duplicate for later use
        # original_data_infos = copy.deepcopy(data_infos)

        # Identify shared keys among different data types
        # TODO having list() might result in a double list if input is a list
        for i in range(len(self.data_infos)):
            self.data_infos[i][key_columns[i]] = self.data_infos[i][key_columns[i]].str.replace('\W+', '')
            # temp = pd.DataFrame({'A': ['a', 'b', 'c'],
            #                      'B': ['1', '2', '3']})
            # temp['A'] = temp['A'].str.replace('a', 'A')
            self.data_infos[i][key_columns[i]] = self.data_infos[i][key_columns[i]].str.upper()
        # If multiple column names are given, then each should match the data_module_list order
        key_col_list = [list(panda[key_columns[i]]) for panda, i in zip(self.data_infos, range(len(key_columns)))]

        # Get the intersection of all keys
        self.key_col_sect = list(set.intersection(*map(set, key_col_list)))

        # Subset cell lines with list of cell lines given to be used as data bottleneck
        if self.bottleneck:
            self.key_col_sect = list(set.intersection(set(self.key_col_sect), set(bottleneck_keys)))
            if verbose:
                print("Restricting keys to those that overlap with the provided bottleneck keys")

        if verbose:
            print("Total overlapping keys:", str(len(self.key_col_sect)))

        # Concatenate omic data info and tensors + get omic data column names (e.g. gene names)
        if self.drug_index is not None:
            # Ignore the drug data which is the first element
            omic_pandas = [pd.concat([cur_data_info[key_columns[i]],
                                       pd.DataFrame(cur_data_tensor.cpu().numpy())],
                                      axis=1) for i, cur_data_info, cur_data_tensor in
                            zip(range(1, len(self.data_infos)), self.data_infos[1:], all_data_tensors[1:])]
            cur_keys = key_columns[1:]
            self.omic_column_names = [data_module_list[i].column_names for i in range(1, len(data_module_list))]
        else:
            omic_pandas = [pd.concat([cur_data_info[key_columns[i]],
                                       pd.DataFrame(cur_data_tensor.numpy())],
                                      axis=1) for i, cur_data_info, cur_data_tensor in
                            zip(range(len(self.data_infos)), self.data_infos, all_data_tensors)]
            cur_keys = key_columns
            self.omic_column_names = [data_module_list[i].column_names for i in range(len(data_module_list))]

        # TODO How to take advantage of all cell lines, even those with replicates?! Perhaps use an ID other than ccl_name that is more unique to each replicate
        # Remove duplicated cell lines in each
        if verbose:
            print("Removing cell line replicate data (!), keeping first match")
        self.omic_pandas = [panda.drop_duplicates(subset=cur_keys[i], keep="first") for panda, i in
                            zip(omic_pandas, range(len(cur_keys)))]

        if verbose:
            print("Converting pandas dataframes to dictionaries, where key is the cell line name")
        # Convert each df to a dictionary based on the key_column, making queries much faster
        dicts = [panda.set_index(cur_keys[i]).T.to_dict('list') for panda, i in
                 zip(self.omic_pandas, range(len(cur_keys)))]

        if verbose:
            print("Converting lists in the dictionary to float32 torch tensors")
        # Subset dict by overlapping keys
        # for cur_dict in dicts:
        #     cur_dict = {k: cur_dict[k] for k in self.key_col_sect}
        dicts = [{k: cur_dict[k] for k in self.key_col_sect} for cur_dict in dicts]

        # Convert all list values to flattened numpy arrays, then to torch tensors
        if to_gpu is True:
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
        # for cur_dict in dicts:
        #     for key, value in cur_dict.items():
        #         assert isinstance(cur_dict[key], torch.Tensor)
        # torch.from_numpy(np.array(input[2], dtype='float').flatten())
        # isinstance(torch.from_numpy(np.array(input[2], dtype='float').flatten()), torch.Tensor)
        # torch.tensor(input[2])
        # type(torch.tensor(input[2]))
        # Subset data infos to keys that overlap (this will be used in CV fold creation)
        self.data_infos = [panda[panda[key_columns[i]].isin(self.key_col_sect)] for panda, i in
                           zip(self.data_infos, range(len(key_columns)))]
        # TODO
        # Except for drug data, combine other omic data types into a single np.array
        # Subset all data based on these overlapping keys
        # Use the original indices of overlapping keys to subset data
        # if len(list(key_columns)) == 1:
        #     # TODO Ensure .isin is giving the correct subset!!!
        #     data_infos = [panda[panda[key_columns]].isin(self.key_col_sect) for panda in original_data_infos]
        #     data_infos = [np.where(panda[panda[key_columns]].isin(self.key_col_sect)) for panda in original_data_infos]
        #     # np.where(data_infos)
        # else:
        #     overlap_indices = [np.where(panda[key_columns[i]].isin(key_col_sect)) for panda, i in
        #               zip(original_data_infos, range(len(key_columns)))]

        # if len(list(key_columns)) == 1:
        #     data_infos = [panda[panda[key_columns]].isin(self.key_col_sect) for panda in original_data_infos]
        # else:
        #     data_infos = [panda[panda[key_columns[i]].isin(self.key_col_sect)] for panda, i in
        #               zip(original_data_infos, range(len(key_columns)))]

        # self.cur_pandas = pandas
        # omic_pandas = pandas

        if self.drug_index is not None:
            drug_data = data_module_list[drug_index]
            drug_fps = drug_data.drug_fps
            drug_data_targets = drug_data.drug_data_targets
            # Subset drug info by
            # self.drug_info = self.data_infos[drug_index]

            # Subset drug data based on overlapping keys
            overlap_idx = np.where(drug_data.data_info[key_columns[drug_index]].isin(self.key_col_sect))[0]
            self.drug_fps = drug_fps[overlap_idx]
            self.drug_data_targets = drug_data_targets[overlap_idx]
            self.drug_data_keys = drug_data.data_info[key_columns[drug_index]][
                drug_data.data_info[key_columns[drug_index]].isin(self.key_col_sect)].to_list()

            if to_gpu is True:
                self.drug_fps = self.drug_fps.cuda()
                self.drug_data_targets = self.drug_data_targets.cuda()

            # self.morgan_col = data_module_list[drug_index].morgan_column
            self.cpd_name_column = drug_data.cpd_name_column
            self.drug_names = drug_data.data_info[self.cpd_name_column].to_list()

            # Get the index of the DR column for better pandas indexing later on
            self.drug_dr_col_idx = drug_data.drug_dr_col_idx

            # TODO: Understand why I moved cur_keys and cur_pandas outside and explain here
            # Must remove (using .drop) the drug key column (ccl_name) and the drug data from pandas using slicing
            # Note that if we use 'del', it will mutate the original list as well!
            # if verbose:
            #     print("Separating drug data from omics data")
            # self.cur_keys = self.key_columns[:self.drug_index] + self.key_columns[self.drug_index + 1:]
            # omic_pandas = pandas[:self.drug_index] + pandas[self.drug_index + 1:]

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
        # Get the cell line omics data from omic dicts
        cur_data = [cur_dict[cur_cell] for cur_dict in self.dicts]
        # Add fingerprint data to omics data
        cur_data = [self.drug_fps[idx]] + cur_data

        # Must return a dose-response summary if once is requested. It should be in the drug data
        # RETURNS: {cell_line_name, drug name}, (encoder data), dose-response target (AAC)
        if self.test_mode is False:
            return torch.Tensor([0]), tuple(cur_data), self.drug_data_targets[idx]
        else:
            return ({"cell_line_name": cur_cell,
                     "drug_name": self.drug_names[idx]},
                    tuple(encoder_data for encoder_data in cur_data),
                    self.drug_data_targets[idx])

    def get_subset(self, cell_lines: [str] = None, cpd_names: [str] = None, partial_match: bool = False,
                   min_target: float = None, max_target: float = None, make_main: bool = False):
        """
        This function will subset data by cell line, compound name, target (e.g. AAC) or a combination of these.

        :make_main: Change the default data to the subset held by this class.
        :return: A list of pandas that is subsetted by the given arguments.
        """
        pandas = copy.deepcopy(self.omic_pandas)
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
    """
    This class creates a stream from a data.DataSet object and loads the next iteration
    on the cuda stream on the GPU. Basically, preloading data on the GPU.
    """

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
