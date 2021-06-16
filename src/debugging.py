# cur_modules_combos = [['drug', 'mut'],
#                       ['drug', 'mut', 'cnv'],
#                       ['drug', 'mut', 'cnv', 'exp'],
#                       ['drug', 'mut', 'cnv', 'exp', 'prot'],
#                       ['drug', 'mut', 'cnv', 'prot'],
#                       ['drug', 'mut', 'exp'],
#                       ['drug', 'mut', 'exp', 'prot'],
#                       ['drug', 'mut', 'prot'],
#                       ['drug', 'cnv'],
#                       ['drug', 'cnv', 'exp'],
#                       ['drug', 'cnv', 'exp', 'prot'],
#                       ['drug', 'cnv', 'prot'],
#                       ['drug', 'exp'],
#                       ['drug', 'exp', 'prot'],
#                       ['drug', 'prot']]

import itertools

from DRPPreparation import drp_main_prep, drp_create_datasets
import numpy as np

path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"

# tcga_exp = OmicData(path=path, omic_file_name="TCGA_PreTraining_Expression.hdf")
# tcga_exp[0]
# temp = tcga_exp.full_train
# temp['cancer_type']
# temp.columns
# tcga_exp[0]
# temp.shape
# temp.iloc[1, :].drop(["stripped_cell_line_name", "tcga_sample_id",
#                                                           "DepMap_ID", "cancer_type"], errors='ignore').to_numpy(dtype=float)
# temp.iloc[1, :].drop(["stripped_cell_line_name", "tcga_sample_id",
#                                                           "DepMap_ID", "cancer_type"], errors='ignore').shape[1]
cur_device = "cpu"
cur_modules = ['drug', 'mut', 'cnv', 'exp', 'prot']

stuff = ["mut", "cnv", "exp", "prot"]
subset = ["mut", "cnv", "exp", "prot"]
# subset = ["mut", "cnv", "exp"]
# subset = ["exp"]
bottleneck = False
for L in range(0, len(stuff)+1):
    for subset in itertools.combinations(stuff, L):
        print(subset)
        if subset == ():
            continue
        prep_list = drp_main_prep(module_list=['drug']+list(subset), train_file="CTRP_AAC_MORGAN_512.hdf",
                                  path=path, device=cur_device)
        _, _, subset_data, subset_keys, subset_encoders, \
            data_list, key_columns = prep_list
        train_data, cv_folds = drp_create_datasets(data_list, key_columns, drug_index=0, drug_dr_column="area_above_curve",
                                                   class_column_name="primary_disease",
                                                   test_drug_data=None, n_folds=5, subset_type="cell_line", verbose=True,
                                                   to_gpu=False)
        # There should be no overlap between training and validation indices
        np.intersect1d(cv_folds[0][0], cv_folds[0][1])
        print("Train len:", len(train_data))
        _, input, target = train_data[1]
        input[0].shape
        input[1].shape
        input[2].shape
        input[3].shape
        input[4].shape
        # All omics: 130K
        # Mut + CNV + EXP: 301K
        # Exp: 303,724
        data_list[0].full_train
