import argparse
import os
import time

import pandas as pd
import torch
from torch.utils import data

from DRPPreparation import drp_create_datasets, drp_load_datatypes
# Helper method to print importances and visualize distribution
from TuneTrainables import file_name_dict


def infer(args):
    if args.machine == "mist":
        data_dir = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/"
        local_dir = "/scratch/l/lstein/ftaj/"
        cur_device = "cuda"
    else:
        data_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
        local_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/"
        cur_device = "cpu"

    if bool(int(args.full)) is True:
        tag = "FullModel"
    else:
        tag = "ResponseOnly"

    # Create the results directory and file paths
    experiment_name = "HyperOpt_DRP_" + tag + '_' + "_".join(args.data_types) + '_' + args.name_tag
    # experiment_name = "HyperOpt_DRP_" + tag + '_' + "drug_prot" + '_' + "CTRP_Full"
    result_dir = local_dir + "/CV_Results/" + experiment_name
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    print("Result directory is:", result_dir)

    data_types = '_'.join(args.data_types)
    print("Current data types are:", data_types)

    print("Reading model from:", result_dir + "/final_model.pt")
    cur_model = torch.load(
        result_dir + "/final_model.pt",
        map_location=torch.device(cur_device))
    cur_model.float()

    data_list, _, key_columns, gpu_locs = drp_load_datatypes(
        train_file=args.data_set,
        # train_file="CTRP_AAC_MORGAN_512.hdf",
         module_list=args.data_types,
         # module_list=["drug", "prot"],
         PATH=data_dir,
         file_name_dict=file_name_dict,
         load_autoencoders=False,
         _pretrain=False,
         device='cpu',
         verbose=False)
    # with test_mode=True, the Dataset also returns the drug and cell line names with the batch
    cur_train_data, cur_cv_folds = drp_create_datasets(data_list,
                                                       key_columns,
                                                       n_folds=2,
                                                       drug_index=0,
                                                       drug_dr_column="area_above_curve",
                                                       class_column_name="primary_disease",
                                                       subset_type="cell_line",
                                                       test_drug_data=None,
                                                       test_mode=True,
                                                       to_gpu=True,
                                                       # to_gpu=False,
                                                       verbose=False)
    test_loader = data.DataLoader(cur_train_data,
                                  batch_size=int(args.sample_size),
                                  # batch_size=32,
                                  # sampler=train_sampler,
                                  shuffle=False,
                                  num_workers=0, pin_memory=False, drop_last=True)

    print("Number of batches to process:", str(len(test_loader)))
    start_time = time.time()
    cur_criterion = torch.nn.L1Loss(reduction='none').cuda()
    cur_model.eval()
    all_results = []
    for i, (cur_cpd_cell, cur_dr_data, cur_dr_target) in enumerate(test_loader):

        cur_output = cur_model(*cur_dr_data)
        cur_loss = cur_criterion(cur_output, cur_dr_target)
        cur_loss = cur_loss.tolist()
        cur_loss = [loss[0] for loss in cur_loss]

        cur_targets = cur_dr_target.tolist()
        cur_targets = [target[0] for target in cur_targets]
        cur_preds = cur_output.tolist()
        cur_preds = [pred[0] for pred in cur_preds]

        cur_dict = {'cpd_name': cur_cpd_cell['drug_name'],
                    'cell_name': cur_cpd_cell['cell_line_name'],
                    'target': cur_targets,
                    'predicted': cur_preds,
                    'MAE_loss': cur_loss,
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
    parser.add_argument('--test_mode', help='Whether to profile on subset of data using cProfiler', default='0')
    parser.add_argument('--name_tag',
                        help='A string that will be added to the model checkpoint file name generated by this program',
                        required=True)
    parser.add_argument('--full', help="whether to optimize all modules (full) or just the DRP module itself",
                        default='0')

    args = parser.parse_args()

    start_time = time.time()
    infer(args)

    print("The whole process took", time.time() - start_time, "seconds")


# python3 DRP/src/drp_inference.py --machine mist --sample_size 128 --data_types drug exp --data_set "CTRP_AAC_MORGAN_512.hdf" --name_tag "CTRP_Full_Split_Cells" --full 1
