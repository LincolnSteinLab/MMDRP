from collections import Counter

import pandas as pd
from scipy.ndimage import convolve1d
import os
import shutil
import torch
import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import pandas as pd

import torch
import torch.nn.functional as F


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def _prepare_weights(self, reweight, max_target=121, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    assert reweight in {'none', 'inverse', 'sqrt_inv'}
    assert reweight != 'none' if lds else True, \
        "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

    value_dict = {x: 0 for x in range(max_target)}
    labels = self.df['age'].values
    for label in labels:
        value_dict[min(max_target - 1, int(label))] += 1
    if reweight == 'sqrt_inv':
        value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
    elif reweight == 'inverse':
        value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
    num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
    if not len(num_per_label) or reweight == 'none':
        return None
    print(f"Using re-weighting: [{reweight.upper()}]")

    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
        num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

    weights = [np.float32(1 / x) for x in num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights


def get_bin_idx(label):
    # For AAC scores that range from 0-1, make 100 bins
    return int(round(label, 2) * 100)


# exp_infer = pd.read_csv("Data/CV_Results/HyperOpt_DRP_ResponseOnly_drug_exp_HyperOpt_DRP_CTRP_1024_ResponseOnly_EncoderTrain_Split_DRUG_NoBottleNeck_NoTCGAPretrain_MergeBySum_RMSELoss_MorganDrugs_drug_exp/CTRP_AAC_MORGAN_1024_inference_results.csv")
# preds, labels = exp_infer['predicted'], exp_infer['target']
# # preds, labels: [Ns,], "Ns" is the number of total samples
# # preds, labels = ..., ...
# # assign each label to its corresponding bin (start from 0)
# # with your defined get_bin_idx(), return bin_index_per_label: [Ns,]
# bin_index_per_label = [get_bin_idx(label) for label in labels]
#
# # calculate empirical (original) label distribution: [Nb,]
# # "Nb" is the number of bins
# Nb = max(bin_index_per_label) + 1
# num_samples_of_bins = dict(Counter(bin_index_per_label))
# emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
#
# # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
# lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
# # calculate effective label distribution: [Nb,]
# eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
#
# import matplotlib
# import matplotlib.pyplot
#
# matplotlib.pyplot.bar(x=list(range(0, len(emp_label_dist))), height=emp_label_dist)
# matplotlib.pyplot.bar(x=list(range(0, len(eff_label_dist))), height=eff_label_dist)
