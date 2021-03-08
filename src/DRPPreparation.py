import sys

import pandas
from sklearn.model_selection import StratifiedKFold
from itertools import cycle, islice
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from DataImportModules import OmicData, DRCurveData, PairData
from ModuleLoader import ExtractEncoder

file_name_dict = {"drug_file_name": "CTRP_AAC_MORGAN.hdf",
                  "mut_file_name": "DepMap_20Q2_CGC_Mutations_by_Cell.hdf",
                  "cnv_file_name": "DepMap_20Q2_CopyNumber.hdf",
                  "exp_file_name": "DepMap_20Q2_Expression.hdf",
                  "prot_file_name": "DepMap_20Q2_No_NA_ProteinQuant.hdf",
                  "tum_file_name": "DepMap_20Q2_Line_Info.csv",
                  "gdsc1_file_name": "GDSC1_AAC_MORGAN.hdf",
                  "gdsc2_file_name": "GDSC2_AAC_MORGAN.hdf",
                  "mut_embed_file_name": "optimal_autoencoders/MUT_Omic_AutoEncoder_Checkpoint.pt",
                  "cnv_embed_file_name": "optimal_autoencoders/CNV_Omic_AutoEncoder_Checkpoint.pt",
                  "exp_embed_file_name": "optimal_autoencoders/EXP_Omic_AutoEncoder_Checkpoint.pt",
                  "prot_embed_file_name": "optimal_autoencoders/PROT_Omic_AutoEncoder_Checkpoint.pt",
                  "4096_drug_embed_file_name": "optimal_autoencoders/Morgan_4096_AutoEncoder_Checkpoint.pt",
                  "2048_drug_embed_file_name": "optimal_autoencoders/Morgan_2048_AutoEncoder_Checkpoint.pt",
                  "1024_drug_embed_file_name": "optimal_autoencoders/Morgan_1024_AutoEncoder_Checkpoint.pt",
                  "512_drug_embed_file_name": "optimal_autoencoders/Morgan_512_AutoEncoder_Checkpoint.pt"}


def create_cv_folds(train_data, n_folds: int = 10, class_data_index: int = 0,
                    class_column_name: str = "primary_disease", subset_type: str = None, seed: int = 42,
                    verbose: bool = False) -> []:
    """
    This function uses sklearn to create n_folds folds of the train_data, and uses the class class_column_name
    to ensure both splits have a roughly equal representation of the given classes.

    :return: list of tuples of np.arrays in (train_indices, valid_indices) format
    """

    assert subset_type in ["cell_line", "drug", "both", None], "subset_type should be one of: cell_line, drug or both"
    if subset_type in ["drug", "both"]:
        exit("Drug and Both subsetting for cross validation is not yet implemented")
    np.random.seed(seed)
    class_data = train_data.cur_pandas[class_data_index]

    # Stratify folds and maintain class distribution
    skf = StratifiedKFold(n_splits=n_folds)
    all_folds = list(skf.split(X=class_data,
                               y=class_data[class_column_name]))

    # Strict or lenient splitting:
    # NOTE: Strict splitting of validation sets most often cannot result in exclusive sets across folds,
    # so the goal here is to minimize the overlap between the validation sets from each fold
    if subset_type == "cell_line":
        if verbose:
            print("Strictly splitting training/validation data based on cell lines while maintaining class distributions")
        # Subset (fully remove) an equal set of cell lines from each class
        all_classes = list(set(class_data[class_column_name].to_list()))

        # Get class proportions and add it to the data frame
        class_proportions = class_data[class_column_name].value_counts(normalize=True)
        class_proportions = pandas.DataFrame({class_column_name: class_proportions.index,
                                              'proportions': class_proportions.iloc[0]})
        class_data = class_proportions.merge(class_data, on=class_column_name)

        all_class_cell_cycles = []
        num_cells_per_fold = []
        for cur_class in all_classes:
            # Get current class cell lines, sample with the number of folds, and separate from train
            cur_class_cells = list(set(list(class_data[class_data[class_column_name] == cur_class]['ccl_name'])))
            cur_class_cells.sort(key=str.lower)
            num_cells_in_fold = int(np.ceil(len(cur_class_cells) / n_folds))
            cur_class_cells = cycle(cur_class_cells)
            all_class_cell_cycles.append(cur_class_cells)
            num_cells_per_fold.append(num_cells_in_fold)

        for i_fold in range(n_folds):
            # For each fold, get cells from each cycle
            cur_validation_cells = []
            for cyc, num_cells in zip(all_class_cell_cycles, num_cells_per_fold):
                cur_validation_cells.append(list(islice(cyc, num_cells)))
            # Flatten the list
            cur_validation_cells = [cur_cell for cur_cells in cur_validation_cells for cur_cell in cur_cells]

            # Separate validation cell lines from training cells
            before_train_len = len(all_folds[i_fold][0])
            before_valid_len = len(all_folds[i_fold][1])
            # Determine indices of validation and training sets
            all_folds[i_fold] = (class_data.index[~class_data['ccl_name'].isin(cur_validation_cells)].to_numpy(),
                                 class_data.index[class_data['ccl_name'].isin(cur_validation_cells)].to_numpy())
            if verbose:
                print("Train data length before:", before_train_len, ", after:", len(all_folds[i_fold][0]),
                      ", Validation data length before:", before_valid_len, ", after:", len(all_folds[i_fold][1]))

    return all_folds


def autoencoder_create_datasets(train_data, train_idx=None, valid_idx=None, valid_size=0.2):
    if (train_idx is not None) and (valid_idx is not None):
        pass
    else:
        # Obtain training indices that will be used for validation
        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.seed(42)
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # return train_loader, valid_loader, train_idx, valid_idx
    # Return train_data for the sake of consistency with the drp_create_datasets function
    return train_data, train_sampler, valid_sampler, train_idx, valid_idx


def drp_load_datatypes(train_file: str, module_list: [str], PATH: str, file_name_dict: {}, device: str = 'gpu',
                       verbose: bool = False):
    """
    This function loads encoder's data and torch models indicated in the module_list argument. It loads
    drug response data from the train_file at the PATH directory.

    :param train_file: File that contains drug response data e.g. from CTRPv2
    :param module_list: A list of strings containing a combination of drug, mut, cnv, exp and prot
    :param PATH: The full path to the directory that contains the autoencoders and input data
    :param file_name_dict: dictionary containing the name of each file
    :param device: either cpu or gpu, for loading the data
    :return: list of input data, list of auto-encoders, list of key column name for each data type
    """
    data_list = []
    autoencoder_list = []
    key_columns = []
    gpu_locs = []

    if "drug" in module_list:
        # TODO: Implement a way to distinguish training dataset, e.g. CTRP vs GDSC
        # Putting drug first ensures that the drug data and module is in a fixed index, i.e. 0
        if verbose:
            print("Loading drug data...")
        cur_data = DRCurveData(path=PATH, file_name=train_file, key_column="ccl_name", class_column="primary_disease",
                               morgan_column="morgan",
                               target_column="area_above_curve")
        data_list.append(cur_data)
        if verbose:
            print("Drug data width:", cur_data.width())
        # Load the whole model
        autoencoder_list.append(torch.load(PATH + file_name_dict[str(cur_data.width()) + "_drug_embed_file_name"],
                                           map_location=torch.device('cpu')))
        key_columns.append("ccl_name")
        if device == "multi_gpu":
            gpu_locs.append(0)
        elif device == "cpu":
            gpu_locs.append(None)
        else:
            gpu_locs.append(0)
    else:
        sys.exit("Drug data must be provided for drug-response prediction")

    if "mut" in module_list:
        if verbose:
            print("Loading mutational data...")
        cur_data = OmicData(path=PATH, omic_file_name=file_name_dict["mut_file_name"])
        data_list.append(cur_data)
        if verbose:
            print("Mut data width:", cur_data.width())
        autoencoder_list.append(
            torch.load(PATH + file_name_dict["mut_embed_file_name"], map_location=torch.device('cpu')))
        key_columns.append("stripped_cell_line_name")
        if device == "multi_gpu":
            gpu_locs.append(0)
        elif device == "cpu":
            gpu_locs.append(None)
        else:
            gpu_locs.append(0)

    if "cnv" in module_list:
        if verbose:
            print("Loading copy number variation data...")
        cur_data = OmicData(path=PATH, omic_file_name=file_name_dict["cnv_file_name"])
        data_list.append(cur_data)
        if verbose:
            print("CNV data width:", cur_data.width())
        autoencoder_list.append(
            torch.load(PATH + file_name_dict["cnv_embed_file_name"], map_location=torch.device('cpu')))

        key_columns.append("stripped_cell_line_name")
        if device == "multi_gpu":
            gpu_locs.append(0)
        elif device == "cpu":
            gpu_locs.append(None)
        else:
            gpu_locs.append(0)

    if "exp" in module_list:
        if verbose:
            print("Loading gene expression data...")
        cur_data = OmicData(path=PATH, omic_file_name=file_name_dict["exp_file_name"])
        data_list.append(cur_data)
        if verbose:
            print("Exp data width:", cur_data.width())
        autoencoder_list.append(
            torch.load(PATH + file_name_dict["exp_embed_file_name"], map_location=torch.device('cpu')))
        key_columns.append("stripped_cell_line_name")
        if device == "multi_gpu":
            gpu_locs.append(0)
        elif device == "cpu":
            gpu_locs.append(None)
        else:
            gpu_locs.append(0)

    if "prot" in module_list:
        if verbose:
            print("Loading protein quantity data...")
        cur_data = OmicData(path=PATH, omic_file_name=file_name_dict["prot_file_name"])
        data_list.append(cur_data)
        if verbose:
            print("Prot data width:", cur_data.width())
        autoencoder_list.append(
            torch.load(PATH + file_name_dict["prot_embed_file_name"], map_location=torch.device('cpu')))

        key_columns.append("stripped_cell_line_name")
        if device == "multi_gpu":
            gpu_locs.append(0)
        elif device == "cpu":
            gpu_locs.append(None)
        else:
            gpu_locs.append(0)

    return data_list, autoencoder_list, key_columns, gpu_locs


def drp_create_datasets(data_list, key_columns, train_idx=None, valid_idx=None, n_folds: int = 10,
                        drug_index: int = 0, drug_dr_column: str = "area_above_curve",
                        class_column_name: str = "primary_disease", subset_type: str = None, test_drug_data=None,
                        bottleneck_keys: [str] = None, verbose: bool = False):
    """
    This function takes a list of data for each auto-encoder and passes it to the PairData function. It creates
    a reproducible, lenient validation split of the data and provides the indices as well as the paired data
    """
    if bottleneck_keys is not None:
        assert len(bottleneck_keys) > 0, "Must have at least one key in the bottleneck keys argument"
    if test_drug_data is not None:
        # Only the drug data is different
        data_list[0] = test_drug_data
        if bottleneck_keys is not None:
            test_data = PairData(data_module_list=data_list, key_columns=key_columns, drug_index=drug_index,
                                 drug_dr_column=drug_dr_column, bottleneck_keys=bottleneck_keys,
                                 test_mode=True)
        else:
            test_data = PairData(data_module_list=data_list, key_columns=key_columns, drug_index=drug_index,
                                 drug_dr_column=drug_dr_column, test_mode=True)

        # load test data in batches
        return test_data
        # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=NUM_CPU,
        #                                           # prefetch_factor=4,
        #                                           # pin_memory=False
        #                                           )

    # Pair given data types based on fixed/assumed key column names
    if bottleneck_keys is not None:
        train_data = PairData(data_module_list=data_list, key_columns=key_columns, drug_index=drug_index,
                              drug_dr_column=drug_dr_column, bottleneck_keys=bottleneck_keys)

    else:
        train_data = PairData(data_module_list=data_list, key_columns=key_columns, drug_index=drug_index,
                              drug_dr_column=drug_dr_column)

    # print("Length of selected data set is:", len(train_data))

    # if (train_idx is not None) and (valid_idx is not None):
    #     pass
    # else:
        # Obtain training indices that will be used for validation
    cv_folds = create_cv_folds(train_data=train_data, n_folds=n_folds, class_data_index=drug_index,
                               subset_type=subset_type, class_column_name=class_column_name, seed=42,
                               verbose=verbose)
    # num_train = len(train_data)
    # indices = list(range(num_train))
    # np.random.seed(42)
    # np.random.shuffle(indices)
    # split = int(np.floor(valid_size * num_train))
    # train_idx, valid_idx = indices[split:], indices[:split]
    #
    # # define samplers for obtaining training and validation batches
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    return train_data, cv_folds


def drp_main_prep(module_list, train_file, path="/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/",
                  device="gpu", bottleneck: bool = False, bottleneck_file: str = "bottleneck_celllines.csv"):
    """
    This function takes a list of data types and uses them to load data and auto-encoders via the
     drp_load_datatypes function, and also extracts the encoders from the auto-encoders. Furthermore, it
     determines the
    """

    assert "drug" in module_list, "Drug data must be provided for drug-response prediction (training or testing)"
    assert len(module_list) > 1, "Data types to be used must be indicated by: mut, cnv, exp, prot and drug"

    # if bottleneck:
    #     try:
    #         bottleneck_keys = pd.read_csv(bottleneck_file)
    #     except:
    #         exit("Could not read bottleneck keys from indicated file")

    # Check to see if checkpoint exists if file name is given
    final_address = []
    if "drug" in module_list:
        final_address.append("drug")
    if "mut" in module_list:
        final_address.append("mut")
    if "cnv" in module_list:
        final_address.append("cnv")
    if "exp" in module_list:
        final_address.append("exp")
    if "prot" in module_list:
        final_address.append("prot")

    assert len(final_address) > 1, "More than valid data types from [drug, mut, cnv, exp, prot] should be given"

    data_list, autoencoder_list, key_columns, \
    gpu_locs = drp_load_datatypes(train_file,
                                  module_list=final_address,
                                  PATH=path,
                                  file_name_dict=file_name_dict,
                                  device=device)

    final_address = "_".join(final_address)

    # Convert autoencoders to encoders; assuming fixed encoder paths
    encoder_list = [ExtractEncoder(autoencoder) for autoencoder in autoencoder_list]

    subset_data = data_list
    subset_encoders = encoder_list
    subset_keys = key_columns

    return device, final_address, subset_data, subset_keys, subset_encoders, data_list, key_columns
