import argparse
import os
import time

import pandas as pd
import torch
from torch.utils import data
from torch_geometric.data import DataLoader

from loss_functions import RMSLELoss, RMSELoss
from DRPPreparation import drp_create_datasets, drp_load_datatypes
from TuneTrainables import file_name_dict


def infer(args):
    if args.machine == "mist":
        data_dir = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/"
        local_dir = "/scratch/l/lstein/ftaj/"
        cur_device = "cuda"
        to_gpu = True
    else:
        data_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
        local_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/"
        cur_device = "cpu"
        to_gpu = False

    if bool(int(args.full)) is True:
        tag = "FullModel"
    else:
        tag = "ResponseOnly"

    # Create the results directory and file paths
    experiment_name = "HyperOpt_DRP_" + tag + '_' + "_".join(args.data_types) + '_' + args.name_tag
    # experiment_name = "HyperOpt_DRP_ResponseOnly_gnndrug_exp_HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GnnDrugs_gnndrug_exp"

    result_dir = local_dir + "/CV_Results/" + experiment_name
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    print("Result directory is:", result_dir)

    # data_types = "gnndrug exp"

    data_types = '_'.join(args.data_types)
    print("Current data types are:", data_types)

    print("Reading model from:", result_dir + "/final_model.pt")
    cur_model = torch.load(
        result_dir + "/final_model.pt",
        map_location=torch.device(cur_device))
    cur_model = cur_model.float()

    if "_SqrtTransform" in experiment_name:
        transform = "sqrt"
    elif "_LogTransform" in experiment_name:
        transform = "log"
    else:
        transform = None

    data_dict, _, key_columns, gpu_locs = drp_load_datatypes(
        train_file=args.data_set,
        # train_file="CTRP_AAC_SMILES.txt",
        module_list=args.data_types,
        # module_list=["gnndrug", "exp"],
        PATH=data_dir,
        file_name_dict=file_name_dict,
        load_autoencoders=False,
        _pretrain=False,
        transform=transform,
        device='cpu',
        verbose=True)
    data_list = list(data_dict.values())

    one_hot_drugs = True if 'OneHotDrugs' in experiment_name else False

    # with infer_mode=True, the Dataset also returns the drug and cell line names with the batch
    cur_train_data, cur_cv_folds = drp_create_datasets(data_list=data_list,
                                                       key_columns=key_columns,
                                                       n_folds=5,
                                                       drug_index=0,
                                                       drug_dr_column="area_above_curve",
                                                       class_column_name="primary_disease",
                                                       subset_type="both",
                                                       test_drug_data=None,
                                                       mode='infer',
                                                       # infer_mode=False,
                                                       gnn_mode=True if 'gnndrug' in args.data_types else False,
                                                       # gnn_mode=True,
                                                       one_hot_drugs=one_hot_drugs,
                                                       to_gpu=to_gpu,
                                                       # lds=True,
                                                       # dr_sub_min_target=args.min_dr_target,
                                                       # dr_sub_max_target=args.max_dr_target,
                                                       verbose=True)
    # TODO standardization should be done with the training data's statistics
    cur_train_data.standardize()

    # test_loader = data.DataLoader(cur_train_data,
    #                               # batch_size=int(args.sample_size),
    #                               batch_size=32,
    #                               # sampler=train_sampler,
    #                               shuffle=False,
    #                               num_workers=0, pin_memory=False, drop_last=True)
    # Get LDS weights
    # import numpy as np
    # loss_weights = cur_train_data.loss_weights.numpy()
    # loss_weights = loss_weights.reshape(loss_weights.shape[0])
    # targets = cur_train_data.drug_data_targets.numpy()
    # targets = targets.reshape(targets.shape[0])
    #
    # cur_data = pd.DataFrame({'targets': targets, 'loss_weights': loss_weights})
    # cur_data.to_csv("ctrp_targets_lds_weights.csv")
    # cur_train_data.eff_label_dist
    test_loader = DataLoader(cur_train_data,
                             batch_size=int(args.sample_size),
                             # batch_size=32,
                             shuffle=False,
                             num_workers=0, pin_memory=False, drop_last=True)

    # cur_train_data.drug_feat_dict
    # gnn_iter = iter(test_loader)
    # temp = next(gnn_iter)
    # len(temp)[3]
    # temp[3].shape
    # temp.batch.shape
    # temp.x.shape
    # temp.edge_index.shape
    # temp.edge_attr.shape
    # temp.omic_data
    #
    # MyFunction(*temp)
    # MyFunction(temp)
    # len(temp.omic_data)
    # len(temp.omic_data[0])
    # print("Number of batches to process:", str(len(test_loader)))
    start_time = time.time()

    # Determine Loss type from the experiment name, do not use LDS-weighted loss functions during inference
    cur_criterion = None
    cur_loss_name = ""
    if "_MAELoss" in experiment_name:
        cur_criterion = torch.nn.L1Loss(reduction='none')
        cur_loss_name = "MAELoss"
    elif "_MSELoss" in experiment_name:
        cur_criterion = torch.nn.MSELoss(reduction='none')
        cur_loss_name = "MSELoss"
    elif "_RMSLELoss" in experiment_name:
        cur_criterion = RMSLELoss(reduction='none')
        cur_loss_name = "RMSLELoss"
    elif "_RMSELoss" in experiment_name or "_WeightedRMSELoss" in experiment_name:
        cur_criterion = RMSELoss(reduction='none')
        cur_loss_name = "RMSELoss"
    else:
        exit("Unknown loss function requested")

    if to_gpu is True:
        cur_criterion = cur_criterion.cuda()

    # temp = enumerate(test_loader)
    # i, (cur_cpd_cell, cur_dr_data, cur_dr_target) = next(temp)
    # i, (info, gnns, cur_dr_data, cur_dr_target, target_weights) = next(temp)
    # i, cur_samples = next(temp)
    # drug_1 = cur_samples[1]
    # drug_2 = next(temp)[1][1]
    cur_model = cur_model.eval()
    # cur_model = cur_model.train()
    all_results = []
    with torch.no_grad():
        for i, cur_samples in enumerate(test_loader):

            # Note: Do NOT use star expansion when passing a torch_geometric.data object
            if 'gnndrug' in args.data_types:
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

            cur_dict = {
                'cpd_name': cur_samples[0]['drug_name'],
                'cell_name': cur_samples[0]['cell_line_name'],
                'target': cur_targets,
                'predicted': cur_preds,
                cur_loss_name: cur_loss,
            }
            all_results.append(pd.DataFrame.from_dict(cur_dict))

            if i % 1000 == 0:
                print("Current batch:", i, ", elapsed:", str(time.time() - start_time), "seconds")

    results_df = pd.concat(all_results)
    dest = result_dir + '/' + str.split(args.data_set, '.')[0] + '_inference_results.csv'
    print("Saving results to:", dest)
    # results_df.to_hdf(result_dir+'/deeplift_results.hdf', key='df', mode='w')
    results_df.to_csv(dest, float_format='%g')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This programs interprets the DRP model")
    parser.add_argument('--machine', help='Whether code is run on cluster or else', default="mist")
    parser.add_argument('--sample_size', help='Sample size to use for attribution', default="1000")
    parser.add_argument('--data_types', nargs="+", help='Data types to be used for attribution, should contain drug')
    parser.add_argument('--data_set', help='Name of training data hdf', default="CTRP_AAC_MORGAN_512.hdf")
    parser.add_argument('--infer_mode', help='Whether to profile on subset of data using cProfiler', default='0')
    parser.add_argument('--name_tag',
                        help='A string that will be added to the model checkpoint file name generated by this program',
                        required=True)
    parser.add_argument('--full', help="whether to optimize all modules (full) or just the DRP module itself",
                        default='0')
    parser.add_argument('--min_dr_target', help='Maximum dose-response AAC value', type=float, default=None)
    parser.add_argument('--max_dr_target', help='Minimum dose-response AAC value', type=float, default=None)

    args = parser.parse_args()

    start_time = time.time()
    infer(args)

    print("The whole process took", time.time() - start_time, "seconds")

# TODO Batch submit
# python3 DRP/src/drp_inference.py --machine mist --sample_size 128 --data_types drug exp --data_set "CTRP_AAC_MORGAN_512.hdf" --name_tag "CTRP_Full_Split_Cells" --full 1
# python3 ~/.conda/envs/drp1/DRP/slurm/drp_batch_inference.py --test ~/.conda/envs/drp1/DRP/slurm/drp_inference_grid.csv
