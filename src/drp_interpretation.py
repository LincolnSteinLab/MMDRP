import argparse

import numpy as np
from torch.utils import data

from DRPPreparation import drp_main_prep, drp_create_datasets
import torch

from captum.attr import IntegratedGradients, DeepLiftShap, GradientShap

# import matplotlib
import matplotlib.pyplot as plt

cur_modules_combos = [['drug', 'mut'],
                      ['drug', 'mut', 'cnv'],
                      ['drug', 'mut', 'cnv', 'exp'],
                      ['drug', 'mut', 'cnv', 'exp', 'prot'],
                      ['drug', 'mut', 'cnv', 'prot'],
                      ['drug', 'mut', 'exp'],
                      ['drug', 'mut', 'exp', 'prot'],
                      ['drug', 'mut', 'prot'],
                      ['drug', 'cnv'],
                      ['drug', 'cnv', 'exp'],
                      ['drug', 'cnv', 'exp', 'prot'],
                      ['drug', 'cnv', 'prot'],
                      ['drug', 'exp'],
                      ['drug', 'exp', 'prot'],
                      ['drug', 'prot']]

import itertools
path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
cur_device = "cpu"
cur_modules = ['drug', 'mut', 'cnv', 'exp', 'prot']

stuff = ["mut", "cnv", "exp", "prot"]
subset = ["mut", "cnv", "exp", "prot"]
bottleneck = False
for L in range(0, len(stuff)+1):
    for subset in itertools.combinations(stuff, L):
        print(subset)
        if subset == ():
            continue
        prep_list = drp_main_prep(module_list=['drug']+list(subset), train_file="CTRP_AAC_MORGAN_512.hdf", path=path, device=cur_device)
        _, _, subset_data, subset_keys, subset_encoders, \
            data_list, key_columns = prep_list
        train_data, cv_folds = drp_create_datasets(data_list, key_columns, drug_index=0, drug_dr_column="area_above_curve",
                                                   class_column_name="primary_disease",
                                                   test_drug_data=None, n_folds=10, subset_type="cell_line", verbose=True)
        print("Train len:", len(train_data))
        data_list[0].full_train

# Helper method to print importances and visualize distribution
def visualize_importances(feature_names, importances, title="Average Feature Importances", top_n=10, plot=True,
                          axis_title="Features"):
    ranks = np.argsort(np.abs(importances))
    largest_n_indices = ranks[::-1][:top_n]
    top_importances = importances[largest_n_indices]
    top_feature_names = np.array(feature_names)[largest_n_indices]

    print(title)
    for i in range(top_n):
        print(top_feature_names[i], ": ", '%.6f' % (top_importances[i]))
    x_pos = (np.arange(len(top_feature_names)))

    if plot:
        plt.figure(figsize=(12, 6))
        plt.bar(x_pos, top_importances, align='center')
        plt.xticks(x_pos, top_feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)


def interpret(args):
    bottleneck = True

    if args.machine == "mist":
        path = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/"
        cur_device = "cuda"
    else:
        path = "/Data/DRP_Training_Data/"
        cur_device = "cpu"

    data_types = '_'.join(args.data_types)
    cur_model = torch.load(
        path + "optimal_autoencoders/" + data_types + "_CTRP_Complete_BottleNeck_NoEncoderTrain_DRP_Checkpoint.pt",
        map_location=torch.device(cur_device))
    cur_model.float()

    cur_modules = ['drug', 'mut', 'cnv', 'exp', 'prot']

    prep_gen = drp_main_prep(module_list=cur_modules, train_file="CTRP_AAC_MORGAN_512.hdf", path=path, device=cur_device)
    prep_list = next(prep_gen)
    _, final_address, subset_data, subset_keys, subset_encoders, \
        data_list, key_columns, required_data_indices = prep_list

    train_data, train_sampler, valid_sampler, \
        train_idx, valid_idx = drp_create_datasets(data_list, key_columns, drug_index=0, drug_dr_column="area_above_curve",
                                                   test_drug_data=None, bottleneck=bottleneck,
                                                   required_data_indices=[0, 1, 2, 3, 4])


    train_loader = data.DataLoader(train_data, batch_size=int(args.sample_size),
                                   sampler=train_sampler,
                                   num_workers=0, pin_memory=True, drop_last=True)
    # load validation data in batches 4 times the size
    valid_loader = data.DataLoader(train_data, batch_size=int(args.sample_size),
                                   sampler=valid_sampler,
                                   num_workers=0, pin_memory=True, drop_last=True)

    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)
    # First output is 0 just for consistency with other data yield state
    _, train_input_tensor, train_target_tensor = next(train_iter)
    _, valid_input_tensor, valid_target_tensor = next(valid_iter)

    for i in range(len(train_input_tensor)):
        train_input_tensor[i] = train_input_tensor[i].float()
        train_target_tensor[i] = train_target_tensor[i].float()
        train_input_tensor[i].requires_grad_()

    for i in range(len(valid_input_tensor)):
        valid_input_tensor[i] = valid_input_tensor[i].float()

    # baselines = tuple([torch.zeros([1, train_input_tensor[0][0].shape[1]]),
    #                  torch.zeros([1, train_input_tensor[0][1].shape[1]]),
    #                  torch.zeros([1, train_input_tensor[0][2].shape[1]]),
    #                  torch.zeros([1, train_input_tensor[0][3].shape[1]]),
    #                  torch.zeros([1, train_input_tensor[0][4].shape[1]]),
    #                  ])

    # select a set of background examples to take an expectation over
    # random_selection = np.random.choice(len(train_data), 1000, replace=False)
    # background = [train_data[i][1] for i in random_selection]
    # Should have a tuple of all batches
    # explain predictions of the model on four images
    # train_input_tensor[0].shape
    #
    # e = shap.GradientExplainer(
    #     cur_model, tuple(train_input_tensor[0]), local_smoothing=0)
    # shap_values, indexes = e.shap_values(valid_input_tensor[0], ranked_outputs=10)
    #
    # # ...or pass tensors directly
    # # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    # type(valid_input_tensor[0])
    # valid_input_tensor[0][0]
    # shap_values = e.shap_values(valid_input_tensor[0][0])

    baselines = [train_input_tensor[i] * 0.0 for i in range(len(train_input_tensor))]
    for i in range(len(train_input_tensor)):
        print(train_input_tensor[i].shape == baselines[i].shape)

    ig = IntegratedGradients(cur_model)
    ig_attr_train, ig_approx_error_train = ig.attribute(
        inputs=tuple(train_input_tensor),
        # target=train_target_tensor,
        baselines=tuple(baselines),
        method='gausslegendre',
        return_convergence_delta=True,
        n_steps=50
    )

    # attr, delta = ig.attribute(tuple(train_input_tensor), n_steps=50)

    # attr = attr.detach().numpy()
    # attr_0 = attr[0].detach().numpy()
    ig_attr_1 = ig_attr_train[1].detach().numpy()
    ig_attr_2 = ig_attr_train[2].detach().numpy()
    ig_attr_3 = ig_attr_train[3].detach().numpy()
    ig_attr_4 = ig_attr_train[4].detach().numpy()
    # visualize_importances(list(train_data.cur_pandas[0].columns), np.mean(attr_0, axis=0), axis_title="Drug")
    visualize_importances(list(train_data.cur_pandas[1].columns), np.mean(ig_attr_1, axis=0), axis_title="Mut",
                          plot=False)
    visualize_importances(list(train_data.cur_pandas[2].columns), np.mean(ig_attr_2, axis=0), axis_title="CNV",
                          plot=False)
    visualize_importances(list(train_data.cur_pandas[3].columns), np.mean(ig_attr_3, axis=0), axis_title="Exp",
                          plot=False)
    visualize_importances(list(train_data.cur_pandas[4].columns), np.mean(ig_attr_4, axis=0), axis_title="Prot",
                          plot=False)

    # visualize_importances(list(train_data.cur_pandas[1].columns), np.mean(attr_1, axis=0), axis_title="Mut", plot=False)
    print("Integrated Gradients Top Attributions (value) per omic input")
    print("Max Mut:", np.max(ig_attr_1), ", Sum Mut:", np.sum(ig_attr_1))
    print("CNV:", np.max(ig_attr_2), ", Sum CNV:", np.sum(ig_attr_2))
    print("Exp:", np.max(ig_attr_3), ", Sum Exp:", np.sum(ig_attr_3))
    print("Prot:", np.max(ig_attr_4), ", Sum Prot:", np.sum(ig_attr_4))

    # DeepLiftSHAP ===========
    # Multiplying by inputs gives global featrue importance
    deeplift_shap = DeepLiftShap(cur_model, multiply_by_inputs=False)
    dls_attr_train, dls_approx_error_train = deeplift_shap.attribute(
        inputs=tuple(train_input_tensor),
        # target=train_target_tensor,
        baselines=tuple(baselines),
        return_convergence_delta=True,
    )

    dls_attr_1 = dls_attr_train[1].detach().numpy()
    dls_attr_2 = dls_attr_train[2].detach().numpy()
    dls_attr_3 = dls_attr_train[3].detach().numpy()
    dls_attr_4 = dls_attr_train[4].detach().numpy()

    print("DeepLift SHAP Top Attributions (value) per omic input")
    print("Mut:", np.max(dls_attr_1), ", Sum Mut:", np.sum(dls_attr_1))
    print("CNV:", np.max(dls_attr_2), ", Sum CNV:", np.sum(dls_attr_2))
    print("Exp:", np.max(dls_attr_3), ", Sum Exp:", np.sum(dls_attr_3))
    print("Prot:", np.max(dls_attr_4), ", Sum Prot:", np.sum(dls_attr_4))

    # GradientSHAP =============
    gs = GradientShap(cur_model)
    gs_attr_train, gs_approx_error_train = gs.attribute(tuple(valid_input_tensor), tuple(train_input_tensor),
                                                        return_convergence_delta=True)
    gs_attr_1 = gs_attr_train[1].detach().numpy()
    gs_attr_2 = gs_attr_train[2].detach().numpy()
    gs_attr_3 = gs_attr_train[3].detach().numpy()
    gs_attr_4 = gs_attr_train[4].detach().numpy()

    print("GradientSHAP Top Attributions (value) per omic input")
    print("Mut:", np.max(gs_attr_1), ", Sum Mut:", np.sum(gs_attr_1))
    print("CNV:", np.max(gs_attr_2), ", Sum CNV:", np.sum(gs_attr_2))
    print("Exp:", np.max(gs_attr_3), ", Sum Exp:", np.sum(gs_attr_3))
    print("Prot:", np.max(gs_attr_4), ", Sum Prot:", np.sum(gs_attr_4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This programs interprets the DRP model")
    parser.add_argument('--machine', help='Whether code is run on cluster or else', default="mist")
    parser.add_argument('--sample_size', help='Sample size to use for attribution', default="1000")
    parser.add_argument('--data_types', nargs="+", help='Data types to be used for attribution, should contain drug')

    args = parser.parse_args()

    interpret(args)
