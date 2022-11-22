import argparse
import json
import os
import re
import string
import time

from collections import defaultdict
from pathlib import Path

from rdkit.Chem.Draw import rdMolDraw2D, _moltoimg
from IPython.display import SVG

import pandas as pd
import torch
from captum.attr import DeepLift, DeepLiftShap, IntegratedGradients, FeatureAblation
from ray import tune
from rdkit import Chem
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from torch_geometric.data import DataLoader

from CustomFunctions import MyLoggerCreator
from DataImportModules import MyGNNData, GenFeatures
from loss_functions import RMSELoss

from DRPPreparation import drp_create_datasets, drp_load_datatypes
# Helper method to print importances and visualize distribution
from TuneTrainables import file_name_dict, DRPTrainable
from drug_visualization import aggregate_node, aggregate_edge_directions, draw_molecule, drug_interpret_viz
import matplotlib.pyplot as plt

TARGETED_DRUGS = ["Idelalisib", "Olaparib", "Venetoclax", "Crizotinib", "Regorafenib",
                    "Tretinoin", "Bortezomib", "Cabozantinib", "Dasatinib", "Erlotinib",
                    "Sonidegib", "Vandetanib", "Axitinib", "Ibrutinib", "Gefitinib",
                    "Nilotinib", "Tamoxifen", "Bosutinib", "Pazopanib", "Lapatinib",
                    "Dabrafenib", "Bexarotene", "Temsirolimus", "Belinostat",
                    "Sunitinib", "Vorinostat", "Trametinib", "Fulvestrant", "Sorafenib",
                    "Vemurafenib", "Alpelisib"]

# def visualize_importances(feature_names, importances, title="Average Feature Importances", top_n=10, plot=True,
#                           axis_title="Features"):
#     ranks = np.argsort(np.abs(importances))
#     largest_n_indices = ranks[::-1][:top_n]
#     top_importances = importances[largest_n_indices]
#     top_feature_names = np.array(feature_names)[largest_n_indices]
#
#     print(title)
#     for i in range(top_n):
#         print(top_feature_names[i], ": ", '%.6f' % (top_importances[i]))
#     x_pos = (np.arange(len(top_feature_names)))
#
#     if plot:
#         plt.figure(figsize=(12, 6))
#         plt.bar(x_pos, top_importances, align='center')
#         plt.xticks(x_pos, top_feature_names, wrap=True)
#         plt.xlabel(axis_title)
#         plt.title(title)


def interpret(args):
    # bottleneck = True

    if args.machine == "mist":
        data_dir = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/"
        local_dir = "/scratch/l/lstein/ftaj/"
        cur_device = "cuda"
        to_gpu = True
        PATH = "/scratch/l/lstein/ftaj/"
        cv_checkpoint_path = "/scratch/l/lstein/ftaj/EpochResults/CrossValidation/" + args.name_tag +\
                             "/checkpoint_cv_" + args.which_model + ".pt"

    else:
        data_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
        local_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/"
        cur_device = "cpu"
        to_gpu = False
        PATH = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/CV_Results/"
        cv_checkpoint_path = "/Volumes/Farzan_External_Drive/Drug_Response/EpochResults/CrossValidation/" + \
                             args.name_tag + "/checkpoint_cv_" + args.which_model + ".pt"

        # HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_CELL_LINE_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GNNDrugs_gnndrug_metab_rppa/checkpoint_cv_0.pt"

    if bool(int(args.full)) is True:
        tag = "FullModel"
    else:
        tag = "ResponseOnly"

    # Create the results directory and file paths
    experiment_name = "HyperOpt_DRP_" + tag + '_' + "_".join(args.data_types) + '_' + args.name_tag
    # experiment_name = "HyperOpt_DRP_" + tag + '_' + "gnndrug_metab_rppa" + '_' + "HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_CELL_LINE_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GNNDrugs_gnndrug_metab_rppa"
    # name_tag = "HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_CELL_LINE_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GNNDrugs_gnndrug_metab_rppa"
    result_dir = local_dir + "/CV_Results/" + experiment_name
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    print("Result directory is:", result_dir)

    # data_types = "gnndrug_metab_rppa"
    data_types = '_'.join(args.data_types)
    print("Current data types are:", data_types)
    if args.which_model == "final":
        saved_model = torch.load(
            result_dir + "/final_model.pt",
            map_location=torch.device(cur_device))
        saved_state_dict = saved_model.state_dict()

    else:
        saved_model = torch.load(
            cv_checkpoint_path,
            map_location=torch.device(cur_device))
        saved_state_dict = saved_model['model_state_dict']

    if "FullModel" in args.name_tag:
        tag = "FullModel"
    else:
        tag = "ResponseOnly"

    # Only use configs attained from hyper-param optimzation on the CTRPv2 dataset
    name_tag = args.name_tag
    # if "GDSC1" in name_tag or "GDSC2" in name_tag:
    #     import re
    #     print("Changing config source from GDSC1/2 to CTRP")
    #     name_tag = re.sub("GDSC[1,2]", "CTRP", name_tag)

    # file_address = PATH + "HyperOpt_DRP_" + tag + '_' + "_".join(['gnndrug', 'metab', 'rppa']) + '_' + name_tag + '/best_config.json'
    file_address = PATH + "HyperOpt_DRP_" + tag + '_' + "_".join(
        args.data_types) + '_' + name_tag + '/best_config.json'
    print("Best config file should be at:", file_address)

    with open(file_address, 'r') as fp:
        config = json.load(fp)
    print("Found best config:", config)

    if "MergeBySum" in name_tag:
        merge_method = "sum"
    elif "MergeByConcat" in name_tag:
        merge_method = "concat"
    else:
        merge_method = "lmf"
    cur_trainable = tune.with_parameters(DRPTrainable,
                                         # train_file=args.train_file,
                                         train_file="CTRP_AAC_SMILES.txt" if "gnndrug" in args.data_types else "CTRP_AAC_MORGAN_1024.hdf",
                                         # train_file="CTRP_AAC_SMILES.txt",
                                         data_dir=data_dir,
                                         data_types='_'.join(args.data_types),
                                         # data_types="gnndrug_rppa_metab",
                                         # bottleneck=bool(int(args.bottleneck)),
                                         bottleneck=False,
                                         # pretrain=bool(int(args.pretrain)),
                                         pretrain=False,
                                         # n_folds=int(args.n_folds),
                                         n_folds=5,
                                         # max_epochs=int(args.max_num_epochs),
                                         max_epochs=100,
                                         encoder_train=True,
                                         cv_subset_type="both",
                                         stratify=True,
                                         random_morgan=False,
                                         merge_method=merge_method,
                                         loss_type='rmse',
                                         one_hot_drugs=False,
                                         gnn_drug=True if 'gnndrug' in args.data_types else False,
                                         # gnn_drug=True,
                                         to_gpu=to_gpu,
                                         # transform=args.transform,
                                         # min_dr_target=args.min_target if args.min_target != 0 else None,
                                         # min_dr_target=0.65,
                                         # dr_sub_cpd_names=TARGETED_DRUGS if args.targeted_drugs_only else None,
                                         # dr_sub_cpd_names=TARGETED_DRUGS,
                                         # max_dr_target=args.max_dr_target,
                                         omic_standardize=True,
                                         test_mode=True
                                         )

    cur_trainable = cur_trainable(config=config, logger_creator=MyLoggerCreator)

    cur_model, criterion = cur_trainable.cur_model, cur_trainable.criterion
    cur_train_data, cur_cv_folds = cur_trainable.cur_train_data, cur_trainable.cur_cv_folds

    if args.specific_drugs is None and args.specific_cells is None:
        # Get cv fold validation drugs and cell lines
        if args.which_model is not None:
            full_valid_cpd = [cur_train_data.drug_names[i] for i in cur_cv_folds[int(args.which_model)][1]]
            full_valid_ccl = [cur_train_data.drug_data_keys[i] for i in cur_cv_folds[int(args.which_model)][1]]
        else:
            # full_valid_cpd = [cur_train_data.drug_names[i] for i in cur_cv_folds[0][1]]
            # full_valid_ccl = [cur_train_data.drug_data_keys[i] for i in cur_cv_folds[0][1]]
            full_valid_cpd = cur_train_data.drug_names
            full_valid_ccl = cur_train_data.drug_data_keys
    elif args.specific_drugs is None and args.specific_cells is not None:
        full_valid_ccl = args.specific_cells
        full_valid_cpd = [cur_train_data.drug_names[i] for i in cur_cv_folds[int(args.which_model)][1]]
    elif args.specific_drugs is not None and args.specific_cells is None:
        full_valid_ccl = [cur_train_data.drug_data_keys[i] for i in cur_cv_folds[int(args.which_model)][1]]
        full_valid_cpd = args.specific_drugs
    else:
        full_valid_ccl = args.specific_cells
        full_valid_cpd = args.specific_drugs

    cur_valid_dict = dict(zip(full_valid_cpd, full_valid_ccl))

    if args.targeted_drugs_only is True:
        # Find the targeted drugs that are in this validation set
        cur_drug_subset = set(TARGETED_DRUGS).intersection(set(cur_valid_dict.keys()))
    else:
        cur_drug_subset = set(cur_valid_dict.keys())

    cur_cell_subset = set(cur_valid_dict.values())
    print("Current valid drugs:", cur_drug_subset)
    print("Length of current valid drugs:", len(cur_drug_subset))
    print("Current valid cell lines:", cur_cell_subset)
    print("Length of current valid cell lines:", len(cur_cell_subset))

    # cur_train_data.subset(cpd_names=cur_drug_subset, cell_lines=cur_cell_subset)
    cur_train_data.subset_cpds(cpd_names=cur_drug_subset)
    cur_train_data.subset_cells(cell_lines=cur_cell_subset)

    # Subset training data based on DR target values
    if args.min_target is not None:
        cur_train_data.subset(min_target=args.min_target)

    if args.machine != "mist":
        # Must update some layer names due to PyG's naming scheme change
        up_dict = {'att_l': 'att_src', 'att_r': "att_dst", "lin_l": "lin_src", "lin_r": "lin_dst"}
        pattern = r'\b({})\b'.format('|'.join(sorted(re.escape(k) for k in up_dict)))
        for key in list(saved_model['model_state_dict'].keys()):
            new_key = re.sub(pattern, lambda m: up_dict.get(m.group(0)), key)
            saved_state_dict[new_key] = saved_state_dict.pop(key)

        saved_state_dict["encoders.0.atom_convs.0.att_l"] = saved_state_dict.pop("encoders.0.atom_convs.0.att_src")
        saved_state_dict["encoders.0.atom_convs.0.att_r"] = saved_state_dict.pop("encoders.0.atom_convs.0.att_dst")

    # try:
    print("Loading saved model state_dict...")
    cur_model.load_state_dict(saved_state_dict)

    omic_types = args.data_types[1:]
    # omic_types = ['metab', 'rppa']

    # if args.which_model != "final":
    #     cur_valid_sampler = SubsetRandomSampler(cur_cv_folds[int(args.which_model)][1])
    #     # cur_valid_sampler = SubsetRandomSampler(cur_cv_folds[0][1])
    # else:
    #     cur_valid_sampler = None
    train_loader = DataLoader(cur_train_data,
                              # sampler=cur_valid_sampler,
                              # batch_size=int(args.sample_size),
                              batch_size=1,
                              shuffle=False,
                              num_workers=0, pin_memory=False, drop_last=False)

    # Create zero and one baselines (references) for backpropagation
    print("Number of batches to process:", str(len(train_loader)))

    if args.method == "deeplift":
        interpret_method = DeepLift(cur_model, multiply_by_inputs=False)
    elif args.method == "deepliftshap":
        interpret_method = DeepLiftShap(cur_model, multiply_by_inputs=False)
    elif args.method == "integratedgradients":
        interpret_method = IntegratedGradients(cur_model, multiply_by_inputs=False)
    elif args.method == "ablation":
        interpret_method = FeatureAblation(cur_model)

    else:
        Warning("Incorrect interpretation method selected, defaulting to DeepLift")
        interpret_method = DeepLift(cur_model, multiply_by_inputs=False)

    cur_criterion = RMSELoss(reduction='none')

    if to_gpu is True:
        cur_criterion = cur_criterion.cuda()
        cur_model = cur_model.cuda()

    if args.infer_mode is True:
        print("Testing performance: only running on first 1000 batches!")

    # for i in range(len(cur_train_data)):
    all_interpret_results = []
    start_time = time.time()

    def custom_forward(*inputs):
        # omic_data, graph_x, graph_edge_attr, graph_edge_index, omic_length):
        omic_length = inputs[-1]
        omic_data = inputs[0:omic_length]
        graph_x = inputs[-5]
        graph_edge_attr = inputs[-4]
        graph_edge_index = inputs[-3]
        batch = inputs[-2]
        # batch = torch.zeros(graph_x.shape[0], dtype=int)
        # if to_gpu:
        #     batch = batch.cuda()
        #     graph_x = graph_x.cuda()
        #     graph_edge_attr = graph_edge_attr.cuda()
        #     graph_edge_index = graph_edge_index.cuda()

        # cur_graph = MyGNNData(x=graph_x, edge_index=graph_edge_index[0],
        #                       edge_attr=graph_edge_attr[0], smiles=graph_smiles, batch=batch)
        # cur_graph = GenFeatures()(cur_graph)

        return cur_model([graph_x, graph_edge_index, graph_edge_attr, batch], omic_data)

    interpret_method = IntegratedGradients(custom_forward)

    # train_iter = iter(train_loader)
    # cur_samples = train_iter.next()
    # t1 = cur_samples[1]
    # t2 = cur_samples[2]
    # t2[0].shape

    # while cur_dr_data is not None:
    for i, cur_samples in enumerate(train_loader):

        if 'gnndrug' in data_types:
            # PairData() output for GNN in test mode is not Batch(), so must reshape manually
            cur_samples[1][0] = torch.squeeze(cur_samples[1][0], 0)
            cur_samples[1][1] = torch.squeeze(cur_samples[1][1], 0)
            cur_samples[1][2] = torch.squeeze(cur_samples[1][2], 0)
            # Add batch
            if args.machine == "mist":
                cur_samples[1] += [torch.zeros(cur_samples[1][0].shape[0], dtype=int).cuda()]
            else:
                cur_samples[1] += [torch.zeros(cur_samples[1][0].shape[0], dtype=int)]

            cur_output = cur_model(cur_samples[1], cur_samples[2])
            # Do not use LDS during inference
            cur_loss = cur_criterion(cur_output, cur_samples[3])
            cur_loss = cur_loss.tolist()
            cur_loss = [loss[0] for loss in cur_loss]

            cur_targets = cur_samples[3].tolist()
            cur_targets = [target[0] for target in cur_targets]
            cur_preds = cur_output.tolist()
            cur_preds = [pred[0] for pred in cur_preds]
        else:
            # raise NotImplementedError
            # print("cur_samples[1] is:", cur_samples[1])
            # print("len input is:", len(cur_samples[1]))
            cur_output = cur_model(*cur_samples[1])
            # Do not use LDS during inference
            cur_loss = cur_criterion(cur_output, cur_samples[2])
            cur_loss = cur_loss.tolist()
            cur_loss = [loss[0] for loss in cur_loss]

            cur_targets = cur_samples[2].tolist()
            cur_targets = [target[0] for target in cur_targets]
            cur_preds = cur_output.tolist()
            cur_preds = [pred[0] for pred in cur_preds]

        omic_length = len(cur_samples[2])

        zero_dl_attr_train, \
        zero_dl_delta_train = interpret_method.attribute((*cur_samples[2], cur_samples[1][0], cur_samples[1][2]),
                                                         additional_forward_args=(
                                                             # cur_samples[1].x,
                                                             # cur_samples[1].edge_attr,
                                                             cur_samples[1][1],
                                                             cur_samples[1][3],
                                                             omic_length
                                                             # cur_samples[1].smiles[0]
                                                         ),
                                                         internal_batch_size=1,
                                                         return_convergence_delta=True)

        # batch = torch.zeros(cur_samples[1].x.shape[0], dtype=int)
        # cur_graph = MyGNNData(x=input_mask, edge_index=cur_samples[1].edge_index,
        #                       edge_attr=cur_samples[1].edge_attr, smiles=cur_samples[1].smiles[0], batch=batch)
        # custom_forward(cur_samples[2][0], cur_samples[1][0], cur_samples[1][1], cur_samples[1][2],
        #                cur_samples[1][3], 1)
        #
        # cur_samples[1].edge_attr.shape
        # input_mask.shape
        #
        # cur_graph = GenFeatures()(cur_graph)
        # cur_model(cur_graph, cur_samples[2])

        cur_dict = {'cpd_name': cur_samples[0]['drug_name'],
                    'cell_name': cur_samples[0]['cell_line_name'],
                    'target': cur_targets,
                    'predicted': cur_preds,
                    'RMSE_loss': cur_loss,
                    'interpret_delta': zero_dl_delta_train.tolist()
                    }

        # Recreate drug graph to use for interpretation
        cur_graph = MyGNNData(x=cur_samples[1][0], edge_index=cur_samples[1][1],
                              edge_attr=cur_samples[1][2], smiles=cur_samples[0]['smiles'][0])
        cur_graph = GenFeatures()(cur_graph)

        # Create drug plotting directory if it doesn't exist
        Path(result_dir+"/drug_plots/").mkdir(parents=True, exist_ok=True)
        # Plot positive and negative attributions on the drug molecule and save as a PNG plot
        drug_interpret_viz(edge_attr=zero_dl_attr_train[-1], node_attr=zero_dl_attr_train[-2],
                           drug_graph=cur_graph, sample_info=cur_samples[0], target=str(round(cur_targets[0], 3)),
                           plot_address=result_dir+"/drug_plots/")

        for j in range(0, len(zero_dl_attr_train) - 2):  # ignore drug data (for now)
            cur_col_names = cur_train_data.omic_column_names[j]
            for jj in range(len(cur_col_names)):
                cur_dict[omic_types[j] + '_' + cur_col_names[jj]] = zero_dl_attr_train[j][:, jj].tolist()

        all_interpret_results.append(pd.DataFrame.from_dict(cur_dict))

        if args.infer_mode is True:
            if i == 10:
                temp = pd.concat(all_interpret_results)
                temp.to_csv(result_dir + '/testing_results.csv')
                print("Profiling done!")
                break

        if i % 100 == 0:
            print("Current batch:", i, ", elapsed:", str(time.time() - start_time), "seconds")
            # cur_len = len(cur_col_names)
            # cur_dataframe = pd.DataFrame(data={'data_type': [omic_types[i]] * cur_len,
            #                                    'omic_col_name': cur_col_names,
            #                                    'drug_name': [None] * cur_len,
            #                                    # 'IntegratedGradients_Base_0': [None] * cur_len,
            #                                    'DeepLiftSHAP_Base_0': [None] * cur_len,
            #                                    })
            # all_interpret_results.append(cur_dataframe)

        # cur_cpd_cell, cur_dr_data, cur_dr_target = train_iter.next()

    results_df = pd.concat(all_interpret_results)
    if args.which_model is not None:
        ig_dest = result_dir + "/integrated_gradients_results_checkpoint_" + args.which_model + ".csv"
    else:
        ig_dest = result_dir + "/integrated_gradients_results_final_model.csv"

    print("Saving results to:", ig_dest)
    # results_df.to_hdf(result_dir+'/deeplift_results.hdf', key='df', mode='w')
    results_df.to_csv(ig_dest, float_format='%g')
    # Measure attributions with IntegratedGradients ================
    # zero_ig_attr_train = ig.attribute(
    #     inputs=tuple(cur_dr_data),
    #     # target=train_target_tensor,
    #     baselines=tuple(zero_baselines),
    #     method='gausslegendre',
    #     # return_convergence_delta=True,
    #     return_convergence_delta=False,
    #     n_steps=50
    # )
    # one_ig_attr_train = ig.attribute(
    #     inputs=tuple(cur_dr_data),
    #     # target=train_target_tensor,
    #     baselines=tuple(one_baselines),
    #     method='gausslegendre',
    #     # return_convergence_delta=True,
    #     return_convergence_delta=False,
    #     n_steps=50
    # )

    # Measure attributions with DeepLiftSHAP ===========
    # Multiplying by inputs gives global featrue importance
    # one_dls_attr_train = deeplift_shap.attribute(
    #     inputs=tuple(cur_dr_data),
    #     # target=train_target_tensor,
    #     baselines=tuple(one_baselines),
    #     # return_convergence_delta=True,
    #     return_convergence_delta=False,
    # )

    # Create a pandas DataFrame that holds all attributes for each cell-line/drug combo
    # TODO

    # attr, delta = ig.attribute(tuple(train_input_tensor), n_steps=50)

    # attr = attr.detach().numpy()
    # attr_0 = attr[0].detach().numpy()
    # zero_ig_attr_1 = zero_ig_attr_train[1].detach().cpu().numpy()
    # one_ig_attr_1 = one_ig_attr_train[1].detach().cpu().numpy()
    # ig_attr_2 = ig_attr_train[2].detach().numpy()
    # ig_attr_3 = ig_attr_train[3].detach().numpy()
    # ig_attr_4 = ig_attr_train[4].detach().numpy()
    # visualize_importances(list(train_data.cur_pandas[0].columns), np.mean(attr_0, axis=0), axis_title="Drug")
    # visualize_importances(list(train_data.cur_pandas[1].columns), np.mean(ig_attr_1, axis=0), axis_title="Mut",
    #                       plot=False)
    # visualize_importances(list(train_data.cur_pandas[2].columns), np.mean(ig_attr_2, axis=0), axis_title="CNV",
    #                       plot=False)
    # visualize_importances(list(train_data.cur_pandas[3].columns), np.mean(ig_attr_3, axis=0), axis_title="Exp",
    #                       plot=False)
    # visualize_importances(list(train_data.cur_pandas[4].columns), np.mean(ig_attr_4, axis=0), axis_title="Prot",
    #                       plot=False)

    # visualize_importances(list(train_data.cur_pandas[1].columns), np.mean(attr_1, axis=0), axis_title="Mut", plot=False)
    # print("Integrated Gradients Top Attributions (value) per omic input")
    # print("Zero reference -> Max Mut:", np.max(zero_ig_attr_1), ", Sum Mut:", np.sum(zero_ig_attr_1))
    # print("One reference -> Max Mut:", np.max(one_ig_attr_1), ", Sum Mut:", np.sum(one_ig_attr_1))
    # print("CNV:", np.max(ig_attr_2), ", Sum CNV:", np.sum(ig_attr_2))
    # print("Exp:", np.max(ig_attr_3), ", Sum Exp:", np.sum(ig_attr_3))
    # print("Prot:", np.max(ig_attr_4), ", Sum Prot:", np.sum(ig_attr_4))

    # TODO Interpret every sample in CTRPv2 and save as csv

    # zero_dls_attr_1 = zero_dls_attr_train[1].detach().cpu().numpy()
    # one_dls_attr_1 = one_dls_attr_train[1].detach().cpu().numpy()
    # dls_attr_2 = dls_attr_train[2].detach().numpy()
    # dls_attr_3 = dls_attr_train[3].detach().numpy()
    # dls_attr_4 = dls_attr_train[4].detach().numpy()

    # print("DeepLift SHAP Top Attributions (value) per omic input")
    # print("Zero reference -> Mut:", np.max(zero_dls_attr_1), ", Sum Mut:", np.sum(zero_dls_attr_1))
    # print("One reference -> Mut:", np.max(one_dls_attr_1), ", Sum Mut:", np.sum(one_dls_attr_1))
    # print("Saving DeepLiftSHAP results:")

    # print("CNV:", np.max(dls_attr_2), ", Sum CNV:", np.sum(dls_attr_2))
    # print("Exp:", np.max(dls_attr_3), ", Sum Exp:", np.sum(dls_attr_3))
    # print("Prot:", np.max(dls_attr_4), ", Sum Prot:", np.sum(dls_attr_4))

    # GradientSHAP =============
    # gs = GradientShap(cur_model)
    # gs_attr_train, gs_approx_error_train = gs.attribute(tuple(valid_input_tensor), tuple(train_input_tensor),
    #                                                     return_convergence_delta=True)
    # gs_attr_1 = gs_attr_train[1].detach().numpy()
    # gs_attr_2 = gs_attr_train[2].detach().numpy()
    # gs_attr_3 = gs_attr_train[3].detach().numpy()
    # gs_attr_4 = gs_attr_train[4].detach().numpy()
    #
    # print("GradientSHAP Top Attributions (value) per omic input")
    # print("Mut:", np.max(gs_attr_1), ", Sum Mut:", np.sum(gs_attr_1))
    # print("CNV:", np.max(gs_attr_2), ", Sum CNV:", np.sum(gs_attr_2))
    # print("Exp:", np.max(gs_attr_3), ", Sum Exp:", np.sum(gs_attr_3))
    # print("Prot:", np.max(gs_attr_4), ", Sum Prot:", np.sum(gs_attr_4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This programs interprets the DRP model")
    parser.add_argument('--machine', help='Whether code is run on cluster or else', default="mist")
    # parser.add_argument('--sample_size', help='Sample size to use for attribution', default="1000")
    parser.add_argument('--data_types', nargs="+", help='Data types to be used for attribution, should contain drug')
    parser.add_argument('--infer_mode', help='Whether to profile on subset of data using cProfiler',
                        action='store_true')
    parser.add_argument('--method',
                        help='One of deeplift, deepliftshap, ablation or integratedgradients interpretation methods',
                        default='deeplift')
    parser.add_argument('--name_tag',
                        help='A string that will be added to the model checkpoint file name generated by this program',
                        required=True)
    parser.add_argument('--full', help="whether to optimize all modules (full) or just the DRP module itself",
                        default='0')
    parser.add_argument('--min_target', help="Subset DRP data by a minimum AAC", type=float, default=0.0)
    parser.add_argument('--which_model', help="Whether to load the (final) model or a specific CV index checkpoint",
                        default="final")
    parser.add_argument('--targeted_drugs_only',
                        help="Whether to subset to only targeted drugs for interpretation", action="store_true")

    parser.add_argument('--specific_drugs', help="Specific compound names to interpret on", nargs='+')
    parser.add_argument('--specific_cells', help="Specific cell line names to interpret on", nargs='+')

    args = parser.parse_args()

    start_time = time.time()
    interpret(args)

    print("The whole process took", time.time() - start_time, "seconds")

# python3 DRP/src/drp_interpretation.py --machine mist --data_types gnndrug prot --name_tag HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GNNDrugs_gnndrug_prot --full 0 --min_target 0.8
