import argparse
import os
import time

import pandas as pd
import torch
from captum.attr import DeepLift, DeepLiftShap, IntegratedGradients, FeatureAblation
from torch.utils import data

from DRPPreparation import drp_create_datasets, drp_load_datatypes
# Helper method to print importances and visualize distribution
from TuneTrainables import file_name_dict


# # import matplotlib


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
    bottleneck = True

    if args.machine == "mist":
        data_dir = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/"
        cur_device = "cuda"
    else:
        data_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
        cur_device = "cpu"

    local_dir = "/scratch/l/lstein/ftaj/"
    # local_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/"
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
    cur_model = torch.load(
        result_dir + "/final_model.pt",
        map_location=torch.device(cur_device))
    cur_model.float()

    # Must create a logger that points to a writable directory
    # if bool(int(args.full)) is True:
    #     cur_trainable = FullModelTrainable(config=config, logger_creator=MyLoggerCreator)
    # else:
    #     cur_trainable = DRPTrainable(config=config, logger_creator=MyLoggerCreator)
    # cur_trainable.setup(config=config)

    # cur_train_data = cur_trainable.cur_train_data

    # cur_modules = ['drug', 'mut', 'cnv', 'exp', 'prot']

    # prep_gen = drp_main_prep(module_list=args.data_types, train_file="CTRP_AAC_MORGAN_512.hdf", path=path, device=cur_device)
    data_list, _, key_columns, gpu_locs = drp_load_datatypes(train_file="CTRP_AAC_MORGAN_512.hdf",
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

    omic_types = args.data_types[1:]
    # omic_types = ['prot']

    # i = 0

    # prep_list = next(prep_gen)
    # _, final_address, subset_data, subset_keys, subset_encoders, \
    # data_list, key_columns, required_data_indices = prep_list

    # train_data, cur_cv_folds = drp_create_datasets(data_list, key_columns, drug_index=0, drug_dr_column="area_above_curve",
    #                                            test_drug_data=None)

    train_loader = data.DataLoader(cur_train_data,
                                   batch_size=int(args.sample_size),
                                   # batch_size=32,
                                   # sampler=train_sampler,
                                   shuffle=False,
                                   num_workers=0, pin_memory=False, drop_last=True)
    # load validation data in batches 4 times the size
    # valid_loader = data.DataLoader(cur_train_data, batch_size=int(args.sample_size),
    #                                sampler=valid_sampler,
    #                                num_workers=0, pin_memory=True, drop_last=True)

    # prefetcher = DataPrefetcher(train_loader)
    # First output is 0 just for consistency with other data yield state
    # _, cur_dr_data, cur_dr_target = prefetcher.next()
    # _, cur_dr_data, cur_dr_target = train_loader[0]

    # Create zero and one baselines (references) for backpropagation
    print("Number of batches to process:", str(len(train_loader)))

    train_iter = iter(train_loader)
    _, cur_dr_data, cur_dr_target = train_iter.next()
    zero_baselines = [cur_dr_data[j] * 0.0 for j in range(len(cur_dr_data))]
    del train_iter

    drug_length = cur_dr_data[0].shape[1]
    # one_baselines = [cur_dr_data[i] * 1.0 for i in range(len(cur_dr_data))]
    # for i in range(len(cur_dr_data)):
    #     print(cur_dr_data[i].shape == zero_baselines[i].shape)

    # Setup interpretation classes
    # ig = IntegratedGradients(cur_model)
    # deeplift_shap = DeepLiftShap(cur_model, multiply_by_inputs=False)
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

    cur_criterion = torch.nn.L1Loss(reduction='none').cuda()

    if bool(int(args.test_mode)) is True:
        print("Testing performance: only running on first 1000 batches!")

    # for i in range(len(cur_train_data)):
    all_interpret_results = []
    start_time = time.time()
    # while cur_dr_data is not None:
    for i, (cur_cpd_cell, cur_dr_data, cur_dr_target) in enumerate(train_loader):
        # print("Current batch:", str(i))
        # for i in range(len(cur_dr_data)):
        #     cur_dr_data[i] = cur_dr_data[i].float()
        #     cur_dr_target[i] = cur_dr_target[i].float()
        #     cur_dr_data[i].requires_grad_()

        # Measure loss for the current batch
        cur_output = cur_model(*cur_dr_data)
        cur_loss = cur_criterion(cur_output, cur_dr_target)
        cur_loss = cur_loss.tolist()
        cur_loss = [loss[0] for loss in cur_loss]
        # print("Current average MSE loss for", args.sample_size, "samples:", cur_loss)
        # print("Current average MSE loss for", 1, "samples:", cur_loss)

        zero_dl_attr_train, zero_dl_delta_train = interpret_method.attribute(
            inputs=tuple(cur_dr_data),
            # target=train_target_tensor,
            baselines=tuple(zero_baselines),
            # return_convergence_delta=True,
            return_convergence_delta=True,
        )
        cur_dict = {'cpd_name': cur_cpd_cell['drug_name'],
                    'cell_name': cur_cpd_cell['cell_line_name'],
                    'MAE_loss': cur_loss,
                    'DeepLIFT_delta': zero_dl_delta_train.tolist()
                    }

        for jj in range(drug_length):
            cur_dict['fp_pos'+str(jj)] = zero_dl_attr_train[0][:, jj].tolist()

        for j in range(1, len(zero_dl_attr_train)):  # ignore drug data (for now)
            cur_col_names = cur_train_data.omic_column_names[j-1]
            for jj in range(len(cur_col_names)):
                cur_dict[omic_types[j-1] + '_' + cur_col_names[jj]] = zero_dl_attr_train[j][:, jj].tolist()

        all_interpret_results.append(pd.DataFrame.from_dict(cur_dict))

        if bool(int(args.test_mode)) is True:
            if i == 1000:
                temp = pd.concat(all_interpret_results)
                temp.to_csv(result_dir+'/testing_results.csv')
                print("Profiling done!")
                exit(0)

        if i % 1000 == 0:
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
    print("Saving results to:", result_dir+'/deeplift_results.hdf')
    # results_df.to_hdf(result_dir+'/deeplift_results.hdf', key='df', mode='w')
    results_df.to_csv(result_dir+'/deeplift_results.csv', float_format='%g')
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
    parser.add_argument('--sample_size', help='Sample size to use for attribution', default="1000")
    parser.add_argument('--data_types', nargs="+", help='Data types to be used for attribution, should contain drug')
    parser.add_argument('--test_mode', help='Whether to profile on subset of data using cProfiler', default='0')
    parser.add_argument('--method', help='One of deeplift, deepliftshap, ablation or integratedgradients interpretation methods',
                        default='deeplift')
    parser.add_argument('--name_tag',
                        help='A string that will be added to the model checkpoint file name generated by this program',
                        required=True)
    parser.add_argument('--full', help="whether to optimize all modules (full) or just the DRP module itself",
                        default='0')

    args = parser.parse_args()

    start_time = time.time()
    interpret(args)

    print("The whole process took", time.time() - start_time, "seconds")


# python3 DRP/src/drp_interpretation.py --machine mist --sample_size 256 --data_types drug prot --name_tag CTRP_Full_Split_Cells --full 1
