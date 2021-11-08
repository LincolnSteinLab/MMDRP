# import argparse
# import os
# import time
#
# import pandas as pd
# import torch
# from torch.utils import data
# from loss_functions import RMSLELoss, RMSELoss
import torch
from ray import tune
from torch_geometric.data import DataLoader

from CustomFunctions import MyLoggerCreator
from DRPPreparation import drp_create_datasets, drp_load_datatypes
from TrainFunctions import cross_validate, gnn_drp_train
from TuneTrainables import file_name_dict, DRPTrainable
from loss_functions import RMSELoss

data_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
local_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/"
cur_device = "cpu"
to_gpu = False

# data_dir = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/"
# local_dir = "/scratch/l/lstein/ftaj/"
# cur_device = "cuda"
# to_gpu = True


tag = "ResponseOnly"
experiment_name = "HyperOpt_DRP_ResponseOnly_gnndrug_exp_HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GNNDrugs_gnndrug_exp"
data_types = "gnndrug exp"
transform = None

data_dict, _, key_columns, gpu_locs = drp_load_datatypes(
    # train_file=args.data_set,
    train_file="CTRP_AAC_SMILES.txt",
    # train_file="GDSC2_AAC_SMILES.txt",
    # train_file="CTRP_AAC_MORGAN_1024.hdf",
    # module_list=args.data_types,
    module_list=["gnndrug", "exp"],
    # module_list=["gnndrug"],
    PATH=data_dir,
    file_name_dict=file_name_dict,
    load_autoencoders=False,
    _pretrain=False,
    transform=transform,
    device='cpu',
    verbose=False)
data_list = list(data_dict.values())

# data_list[0].full_train
# data_list[0].drug_names
# data_list[0].data_info


one_hot_drugs = True if 'OneHotDrugs' in experiment_name else False

cur_train_data, cur_cv_folds = drp_create_datasets(data_list=data_list,
                                                   key_columns=key_columns,
                                                   n_folds=5,
                                                   drug_index=0,
                                                   drug_dr_column="area_above_curve",
                                                   class_column_name="primary_disease",
                                                   subset_type="both",
                                                   test_drug_data=None,
                                                   # infer_mode=False,
                                                   mode='train',
                                                   gnn_mode=True,
                                                   one_hot_drugs=one_hot_drugs,
                                                   to_gpu=to_gpu,
                                                   # lds=True,
                                                   # dr_sub_min_target=args.min_dr_target,
                                                   # dr_sub_max_target=args.max_dr_target,
                                                   verbose=True)

len(cur_train_data.drug_data_keys)
cur_train_data.standardize()
cur_train_data.subset(min_target=0.6)
len(cur_train_data)
cur_train_data.drug_data_keys
cur_train_data.data_list[0].drug_feat_dict
cur_train_data.data_list[0].drug_data_targets
cur_train_data.drug_data_keys
cur_train_data.data_list[0].all_data[cur_train_data.data_list[0].all_data[cur_train_data.data_list[0].target_column] > 0.8]
# cur_trainable = tune.with_parameters(DRPTrainable,
#                                      train_file="CTRP_AAC_SMILES.txt",
#                                      data_types="gnndrug_exp",
#                                      bottleneck=False,
#                                      pretrain=False,
#                                      n_folds=5,
#                                      max_epochs=5,
#                                      encoder_train=True,
#                                      cv_subset_type='both',
#                                      stratify=True,
#                                      random_morgan=False,
#                                      merge_method='lmf',
#                                      loss_type='weighted_rmse',
#                                      one_hot_drugs=False,
#                                      gnn_drug=True,
#                                      transform=None,
#                                      min_dr_target=None,
#                                      max_dr_target=None,
#                                      omic_standardize=True,
#                                      to_gpu=False,
#                                      data_dir=data_dir
#                                      )
# config = {"drp_first_layer_size": 370, "drp_last_layer_size": 12, "drp_num_layers": 3, "lr": 1e-05, "batch_size": 16, "gnn_hidden_channels": 500, "gnn_out_channels": 400, "gnn_num_layers": 3, "gnn_num_timesteps": 3, "gnn_dropout": 0.10310656849805565, "lmf_output_dim": 863, "lmf_rank": 9}
# cur_trainable = cur_trainable(config=config, logger_creator=MyLoggerCreator)
# cur_trainable = cur_trainable(config=config)

# cur_model, criterion = cur_trainable.cur_model, cur_trainable.criterion
# criterion = cur_trainable.criterion
cur_criterion = RMSELoss(reduction='none')
cur_loss_name = "RMSELoss"

result_dir = local_dir + "/CV_Results/" + experiment_name
print("Reading model from:", result_dir + "/final_model.pt")
cur_model = torch.load(
    result_dir + "/final_model.pt",
    map_location=torch.device(cur_device))
cur_model = cur_model.float()

# cur_train_data, cur_cv_folds = cur_trainable.cur_train_data, cur_trainable.cur_cv_folds


# cur_model.drp_module[1].custom_dense.drp_1_linear.register_forward_hook(lambda m, input, output: print(output))
# cur_model.encoders[0].lin2.register_forward_hook(lambda m, input, output: print(output))
# cur_model.encoders[1].encoder.coder[1].custom_dense.code_layer_exp_linear.register_forward_hook(lambda m, input, output: print(output))


test_loader = DataLoader(cur_train_data,
                         # batch_size=int(args.sample_size),
                         batch_size=32,
                         shuffle=False,
                         num_workers=0, pin_memory=False, drop_last=True)
temp = enumerate(test_loader)
i, cur_samples = next(temp)

cur_model(cur_samples[1], cur_samples[2])

# cur_samples[2][0] = torch.zeros(cur_samples[2][0].shape)
# cur_model(cur_samples[1], cur_samples[2])

epoch_save_folder = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/EpochResults/CrossValidation/HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GNNDrugs_gnndrug_exp"

final_model, \
avg_train_losses, \
avg_valid_losses, \
avg_untrained_losses, \
max_final_epoch = cross_validate(train_data=cur_train_data,
                                 train_function=gnn_drp_train,
                                 cv_folds=cur_cv_folds,
                                 batch_size=config['batch_size'],
                                 cur_model=cur_model,
                                 criterion=criterion,
                                 max_epochs=1,
                                 patience=10,
                                 train_only=True,
                                 final_full_train=True,
                                 learning_rate=config['lr'],
                                 delta=0.001,
                                 NUM_WORKERS=0,
                                 theoretical_loss=False,
                                 omic_standardize=True,
                                 save_epoch_results=True,
                                 epoch_save_folder=epoch_save_folder,
                                 save_model=True,
                                 save_model_frequency=5,
                                 save_model_path=epoch_save_folder,
                                 resume=False,
                                 verbose=True)
