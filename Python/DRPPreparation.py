import sys
import time

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

from DataImportModules import OmicData, DRCurveData
from DataImportModules import PairData
from ModuleLoader import ExtractEncoder

file_name_dict = {"drug_file_name": "CTRP_AUC_MORGAN.hdf",
                  "mut_file_name": "DepMap_20Q2_CGC_Mutations_by_Cell.hdf",
                  "cnv_file_name": "DepMap_20Q2_CopyNumber.hdf",
                  "exp_file_name": "DepMap_20Q2_Expression.hdf",
                  "prot_file_name": "DepMap_20Q2_No_NA_ProteinQuant.hdf",
                  "tum_file_name": "DepMap_20Q2_Line_Info.csv",
                  "gdsc1_file_name": "GDSC1_AUC_MORGAN.hdf",
                  "gdsc2_file_name": "GDSC2_AUC_MORGAN.hdf",
                  "mut_embed_file_name": "optimal_autoencoders/MUT_Omic_AutoEncoder_Checkpoint.pt",
                  "cnv_embed_file_name": "optimal_autoencoders/CNV_Omic_AutoEncoder_Checkpoint.pt",
                  "exp_embed_file_name": "optimal_autoencoders/EXP_Omic_AutoEncoder_Checkpoint.pt",
                  "prot_embed_file_name": "optimal_autoencoders/PROT_Omic_AutoEncoder_Checkpoint.pt",
                  "4096_drug_embed_file_name": "optimal_autoencoders/Morgan_4096_AutoEncoder_Checkpoint.pt",
                  "2048_drug_embed_file_name": "optimal_autoencoders/Morgan_2048_AutoEncoder_Checkpoint.pt",
                  "1024_drug_embed_file_name": "optimal_autoencoders/Morgan_1024_AutoEncoder_Checkpoint.pt",
                  "512_drug_embed_file_name": "optimal_autoencoders/Morgan_512_AutoEncoder_Checkpoint.pt"}


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


def drp_create_datasets(data_list, key_columns, train_idx=None, valid_idx=None, valid_size=0.1,
                        drug_index=0,
                        drug_dr_column="area_under_curve",
                        test_drug_data=None, bottleneck=False, required_data_indices=None):
    """
    :param valid_size:
    :param valid_idx:
    :param train_idx:
    :param data_list:
    :param key_columns:
    :param drug_index:
    :param drug_dr_column:
    :param test_drug_data:
    :param bottleneck: If True, then the assumption is that data_list contains all omic data types to be used for the
    create of a complete sample, and that required_data_indices indicates the data that has to be returned during run time.
    This is for training on data that is available to all modules.
    :param required_data_indices:
    :return: Test loader if test data is given, and train + validation loaders with the chosen indices
    TODO: returning the indices makes sense only if PairData always produces results in the same order! Ensure this is the case!
    """
    # TODO: FIX: If PairData is given all data types, then it returns all data types, not the ones that are in data_list (duh)
    if bottleneck is True:
        assert required_data_indices is not None, "required_data_indices should be provided if bottleneck==True"

    if test_drug_data is not None:
        # Only the drug data is different
        data_list[0] = test_drug_data
        if bottleneck is True:
            test_data = PairData(data_module_list=data_list, key_columns=key_columns, drug_index=drug_index,
                                 drug_dr_column=drug_dr_column, required_data_indices=required_data_indices,
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
    if bottleneck is True:
        train_data = PairData(data_module_list=data_list, key_columns=key_columns, drug_index=drug_index,
                              drug_dr_column=drug_dr_column, required_data_indices=required_data_indices)

    else:
        train_data = PairData(data_module_list=data_list, key_columns=key_columns, drug_index=drug_index,
                              drug_dr_column=drug_dr_column)

    # print("Length of selected data set is:", len(train_data))

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

    # Create data_loader
    # train_loader = data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
    #                                num_workers=NUM_CPU,
    #                                # persistent_workers=True,
    #                                # prefetch_factor=4,
    #                                pin_memory=True)
    #
    # # load validation data in batches double the size
    # valid_loader = data.DataLoader(train_data, batch_size=batch_size * 8, sampler=valid_sampler,
    #                                num_workers=NUM_CPU,
    #                                # persistent_workers=False,
    #                                # prefetch_factor=4,
    #                                pin_memory=True)

    # return train_loader, valid_loader, train_idx, valid_idx
    return train_data, train_sampler, valid_sampler, train_idx, valid_idx


def drp_load_datatypes(train_file, module_list, PATH, file_name_dict, device='gpu'):
    # Create data-loader and autoencoder modules based on the data types requested in the command line
    data_list = []
    autoencoder_list = []
    key_columns = []
    # state_paths = []
    gpu_locs = []

    if "drug" in module_list:
        # TODO: Implement a way to distinguish training dataset, e.g. CTRP vs GDSC
        # Putting drug first ensures that the drug data and module is in a fixed index, i.e. 0
        # print("Loading drug data...")
        cur_data = DRCurveData(path=PATH, file_name=train_file, key_column="ccl_name", morgan_column="morgan",
                               target_column="area_under_curve")
        data_list.append(cur_data)
        # print("Drug data width:", cur_data.width())
        # Load the whole model
        autoencoder_list.append(torch.load(PATH + file_name_dict[str(cur_data.width()) + "_drug_embed_file_name"],
                                           map_location=torch.device('cpu')))
        # autoencoder_list.append(DNNAutoEncoder(cur_data.width(), first_layer_size=621,
        #                                        code_layer_size=876, num_layers=3, batchnorm=False, act_fun="none",
        #                                        dropout=1.1387e-5, name="morgan"))
        key_columns.append("ccl_name")
        # Load the drug model that was trained on fingerprints with given width
        # state_paths.append(PATH + file_name_dict[str(cur_data.width()) + "_drug_embed_file_name"])
        if device == "multi_gpu":
            gpu_locs.append(0)
        elif device == "cpu":
            gpu_locs.append(None)
        else:
            gpu_locs.append(0)
    else:
        sys.exit("Drug data must be provided for drug-response prediction")

    if "mut" in module_list:
        # print("Loading mutational data...")
        cur_data = OmicData(path=PATH, omic_file_name=file_name_dict["mut_file_name"])
        data_list.append(cur_data)
        # print("Mut data width:", cur_data.width())
        # autoencoder_list.append(
        #     DNNAutoEncoder(input_dim=cur_data.width(), first_layer_size=1022, code_layer_size=435, num_layers=2,
        #                    batchnorm=True, act_fun='none', name="mut"))
        autoencoder_list.append(
            torch.load(PATH + file_name_dict["mut_embed_file_name"], map_location=torch.device('cpu')))
        key_columns.append("stripped_cell_line_name")
        # state_paths.append(PATH + file_name_dict["mut_embed_file_name"])
        if device == "multi_gpu":
            gpu_locs.append(0)
        elif device == "cpu":
            gpu_locs.append(None)
        else:
            gpu_locs.append(0)

    if "cnv" in module_list:
        # print("Loading copy number variation data...")
        cur_data = OmicData(path=PATH, omic_file_name=file_name_dict["cnv_file_name"])
        data_list.append(cur_data)
        # print("CNV data width:", cur_data.width())
        # autoencoder_list.append(
        #     DNNAutoEncoder(input_dim=cur_data.width(), first_layer_size=7293, code_layer_size=260, num_layers=2,
        #                    batchnorm=True, act_fun='none', name="cnv"))
        autoencoder_list.append(
            torch.load(PATH + file_name_dict["cnv_embed_file_name"], map_location=torch.device('cpu')))

        key_columns.append("stripped_cell_line_name")
        # state_paths.append(PATH + file_name_dict["cnv_embed_file_name"])
        if device == "multi_gpu":
            gpu_locs.append(0)
        elif device == "cpu":
            gpu_locs.append(None)
        else:
            gpu_locs.append(0)

    if "exp" in module_list:
        # print("Loading gene expression data...")
        cur_data = OmicData(path=PATH, omic_file_name=file_name_dict["exp_file_name"])
        data_list.append(cur_data)
        # print("Exp data width:", cur_data.width())
        # autoencoder_list.append(
        #     DNNAutoEncoder(input_dim=cur_data.width(), first_layer_size=3397, code_layer_size=2312, num_layers=2,
        #                    batchnorm=False, act_fun='none', name="exp"))
        autoencoder_list.append(
            torch.load(PATH + file_name_dict["exp_embed_file_name"], map_location=torch.device('cpu')))
        key_columns.append("stripped_cell_line_name")
        # state_paths.append(PATH + file_name_dict["exp_embed_file_name"])
        if device == "multi_gpu":
            gpu_locs.append(0)
        elif device == "cpu":
            gpu_locs.append(None)
        else:
            gpu_locs.append(0)

    if "prot" in module_list:
        # print("Loading protein quantity data...")
        cur_data = OmicData(path=PATH, omic_file_name=file_name_dict["prot_file_name"])
        data_list.append(cur_data)
        # print("Prot data width:", cur_data.width())
        # autoencoder_list.append(
        #     DNNAutoEncoder(input_dim=cur_data.width(), first_layer_size=5910, code_layer_size=4725, num_layers=2,
        #                    batchnorm=False, act_fun='none', name="prot"))
        autoencoder_list.append(
            torch.load(PATH + file_name_dict["prot_embed_file_name"], map_location=torch.device('cpu')))

        key_columns.append("stripped_cell_line_name")
        # state_paths.append(PATH + file_name_dict["prot_embed_file_name"])
        if device == "multi_gpu":
            gpu_locs.append(0)
        elif device == "cpu":
            gpu_locs.append(None)
        else:
            gpu_locs.append(0)

    return data_list, autoencoder_list, key_columns, gpu_locs


def drp_main_prep(module_list, train_file, path="/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/",
                  device="gpu"):
    """
    This function prepares data, encoders, and DRP model configuration for a specific combo of omic data.
    :device: can be either cpu, gpu or multi_gpu
    :return: Yields a combo of omics data and associated DRP model

    """
    PATH = path
    # TODO setup arg based file selection based on width
    # train_file = "CTRP_AUC_MORGAN_" + str(fp_width) + ".hdf"
    # PATH = "/u/ftaj/anaconda3/envs/Python/Data/DRP_Training_Data/"
    # PATH = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"

    main_start = time.time()
    # machine = args.machine
    # num_epochs = int(args.num_epochs)
    # opt_level = args.opt_level
    # num_gpus = int(torch.cuda.device_count())
    # batch_size = int(config["batch_size"])
    # resume = bool(int(args.resume))
    # interpret = bool(int(args.interpret))

    assert len(module_list) > 0, "Data types to be used must be indicated by: mut, cnv, exp, prot and drug"
    # assert opt_level in ["O0", "O1", "O2", "O3"], "Second argument (AMP opt_level) must be one of O0, O1, O2 or O3"
    assert "drug" in module_list, "Drug data must be provided for drug-response prediction (training or testing)"

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

    final_address = "_".join(final_address)
    # if resume is True:
    #     assert os.path.isfile(args.name_tag + '/drp_' + str(
    #         final_address) + "_" + args.name_tag + '_best_checkpoint.pth.tar'), "A model checkpoint doesn't exist for the given data types"

    # Check to see if we have sufficient number of GPUs depending on chosen data types
    # TODO Have to adjust based on available GPU memory

    # Read ALL data types, then subset later on
    data_list, autoencoder_list, key_columns, \
    gpu_locs = drp_load_datatypes(train_file,
                                  module_list=['drug', 'mut', 'cnv', 'exp', 'prot'],
                                  PATH=PATH,
                                  file_name_dict=file_name_dict,
                                  device=device)
    # TODO What if the user doesn't want combinations and only has 1 GPU? Loading all encoders is going to CRASH!
    # Convert autoencoders to encoders; assuming fixed encoder paths
    encoder_list = [ExtractEncoder(autoencoder) for autoencoder in autoencoder_list]

    # TODO BYPASS:
    # Generate all combinations of the omics data types requested and train a model on each combo, if all combos are requested
    all_omics = ['mut', 'cnv', 'exp', 'prot']
    all_omic_combinations = []
    # if bool(int(args.all_combos)) is True:
    #     # assert num_gpus > 1, "More than 1 GPU is needed if all combinations of data types are requested"
    #     print("Training on all possible combinations of omics data, ignoring your inputs...")
    #     for L in range(1, len(all_omics) + 1):
    #         for subset in itertools.combinations(all_omics, L):
    #             all_omic_combinations.append(['drug'] + list(subset))
    # else:
    all_omic_combinations.append(final_address.split('_'))

    for combo in all_omic_combinations:
        # Subset encoder list based on data types (drug data is always present)
        subset_encoders = [encoder_list[0]]
        subset_gpu_locs = [gpu_locs[0]]
        subset_data = [data_list[0]]
        required_data_indices = [0]
        subset_keys = ["ccl_name"]
        if "mut" in combo:
            subset_encoders.append(encoder_list[1])
            subset_gpu_locs.append(gpu_locs[1])
            subset_data.append(data_list[1])
            required_data_indices.append(1)
            subset_keys.append(key_columns[1])
        if "cnv" in combo:
            subset_encoders.append(encoder_list[2])
            subset_gpu_locs.append(gpu_locs[2])
            subset_data.append(data_list[2])
            required_data_indices.append(2)
            subset_keys.append(key_columns[2])
        if "exp" in combo:
            subset_encoders.append(encoder_list[3])
            subset_gpu_locs.append(gpu_locs[3])
            subset_data.append(data_list[3])
            required_data_indices.append(3)
            subset_keys.append(key_columns[3])
        if "prot" in combo:
            subset_encoders.append(encoder_list[4])
            subset_gpu_locs.append(gpu_locs[4])
            subset_data.append(data_list[4])
            required_data_indices.append(4)
            subset_keys.append(key_columns[4])

        final_address = '_'.join(combo)

        # Check to see if a model with these modules has already been trained, and skip if true
        # if resume is False:
        #     if os.path.isfile(args.name_tag + '/drp_' + str(
        #             final_address) + '_' + args.name_tag + '_best_checkpoint.pth.tar') is True:
        #         if args.force is None:
        #             print(
        #                 args.name_tag + '/drp_' + str(final_address) + '_' + args.name_tag + '_best_checkpoint.pth.tar',
        #                 "already exists, skipping...")
        #             continue
        #         else:
        #             print(
        #                 args.name_tag + '/drp_' + str(final_address) + '_' + args.name_tag + '_best_checkpoint.pth.tar',
        #                 "already exists, forcing a re-run...")

        # print("Valid key columns used in data types are:", subset_keys)
        # if multi_gpu:
        #     print("Locations of each module on GPUs are:", subset_gpu_locs)

        yield device, final_address, subset_data, subset_keys, subset_encoders, data_list, key_columns, required_data_indices
