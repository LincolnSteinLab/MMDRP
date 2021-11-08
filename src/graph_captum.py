import argparse
import os
import time

import pandas as pd
import torch
from captum.attr import DeepLift, DeepLiftShap, IntegratedGradients, FeatureAblation
from torch.utils import data
from torch_geometric.data import DataLoader

from DataImportModules import MyGNNData, GenFeatures
from loss_functions import RMSELoss

from DRPPreparation import drp_create_datasets, drp_load_datatypes
# Helper method to print importances and visualize distribution
from TuneTrainables import file_name_dict

data_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
local_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/"
cur_device = "cpu"
to_gpu = False
tag = "ResponseOnly"
experiment_name = "HyperOpt_DRP_" + tag + '_' + "gnndrug_prot" + '_' + "HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GNNDrugs_gnndrug_prot"
result_dir = local_dir + "/CV_Results/" + experiment_name
data_types = "gnndrug_prot"
cur_model = torch.load(
    result_dir + "/final_model.pt",
    map_location=torch.device(cur_device))
cur_model.float()

data_dict, _, key_columns, gpu_locs = drp_load_datatypes(
    # train_file=args.data_set,
    train_file="CTRP_AAC_SMILES.txt",
    # module_list=args.data_types,
    module_list=["gnndrug", "prot"],
    PATH=data_dir,
    file_name_dict=file_name_dict,
    load_autoencoders=False,
    _pretrain=False,
    device='cpu',
    verbose=False)
data_list = list(data_dict.values())

# with infer_mode=True, the Dataset also returns the drug and cell line names with the batch
cur_train_data, cur_cv_folds = drp_create_datasets(data_list=data_list,
                                                   key_columns=key_columns,
                                                   n_folds=5,
                                                   drug_index=0,
                                                   drug_dr_column="area_above_curve",
                                                   class_column_name="primary_disease",
                                                   subset_type="both",
                                                   test_drug_data=None,
                                                   mode=True,
                                                   # infer_mode=False,
                                                   # gnn_mode=True if 'gnndrug' in args.data_types else False,
                                                   gnn_mode=True,
                                                   to_gpu=to_gpu,
                                                   # lds=True,
                                                   # dr_sub_min_target=args.min_dr_target,
                                                   # dr_sub_max_target=args.max_dr_target,
                                                   verbose=True)
cur_train_data.standardize()
omic_types = ['prot']

train_loader = DataLoader(cur_train_data,
                          # batch_size=int(args.sample_size),
                          batch_size=1,
                          shuffle=False,
                          num_workers=0, pin_memory=False, drop_last=True)

train_iter = iter(train_loader)
cur_samples = train_iter.next()

cur_criterion = RMSELoss(reduction='none')


def custom_forward(omic_data, graph_x, graph_edge_attr, graph_edge_index, graph_smiles):
    batch = torch.zeros(graph_x.shape[0], dtype=int)
    cur_graph = MyGNNData(x=graph_x, edge_index=graph_edge_index[0],
                          edge_attr=graph_edge_attr[0], smiles=graph_smiles, batch=batch)
    cur_graph = GenFeatures()(cur_graph)

    # return cur_model([graph_x, graph_edge_index, graph_edge_attr, batch], [omic_data])
    return cur_model(cur_graph, [omic_data])


interpret_method = IntegratedGradients(custom_forward)

# input_mask = torch.ones(cur_samples[1].x.shape[0],
#                         cur_samples[1].x.shape[1]).requires_grad_(True)

zero_dl_attr_train, \
zero_dl_delta_train = interpret_method.attribute(cur_samples[2][0],
                                                 additional_forward_args=(cur_samples[1].x,
                                                                          cur_samples[1].edge_attr,
                                                                          cur_samples[1].edge_index,
                                                                          cur_samples[1].smiles[0]),
                                                 internal_batch_size=1,
                                                 return_convergence_delta=True)

batch = torch.zeros(cur_samples[1].x.shape[0], dtype=int)
cur_graph = MyGNNData(x=input_mask, edge_index=cur_samples[1].edge_index,
                      edge_attr=cur_samples[1].edge_attr, smiles=cur_samples[1].smiles[0], batch=batch)
# cur_omic = cur_samples[2][0]
cur_samples = train_iter.next()

custom_forward(cur_samples[2][0],
               cur_samples[1].x,
               cur_samples[1].edge_attr,
               cur_samples[1].edge_index,
               cur_samples[1].smiles[0])
