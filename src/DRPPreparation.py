import sys
import random

import pandas
from sklearn.model_selection import StratifiedKFold, KFold
from itertools import cycle, islice
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Tuple
import networkx as nx
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse import csr_matrix
from matplotlib.pyplot import figure

from CustomFunctions import cell_drug_match
from DataImportModules import OmicData, DRCurveData, PairData
from ModuleLoader import ExtractEncoder

file_name_dict = {"drug_file_name": "CTRP_AAC_MORGAN.hdf",
                  "mut_file_name": "DepMap_20Q2_CGC_Mutations_by_Cell.hdf",
                  "cnv_file_name": "DepMap_20Q2_CopyNumber.hdf",
                  "exp_file_name": "DepMap_20Q2_Expression.hdf",
                  "prot_file_name": "DepMap_20Q2_No_NA_ProteinQuant.hdf",
                  "tum_file_name": "DepMap_20Q2_Line_Info.csv",
                  "gdsc1_file_name": "GDSC1_AAC_MORGAN.hdf",
                  "gdsc2_file_name": "GDSC2_AAC_MORGAN.hdf",
                  "mut_embed_file_name": "optimal_autoencoders/MUT_Omic_AutoEncoder_Checkpoint.pt",
                  "cnv_embed_file_name": "optimal_autoencoders/CNV_Omic_AutoEncoder_Checkpoint.pt",
                  "exp_embed_file_name": "optimal_autoencoders/EXP_Omic_AutoEncoder_Checkpoint.pt",
                  "prot_embed_file_name": "optimal_autoencoders/PROT_Omic_AutoEncoder_Checkpoint.pt",
                  "4096_drug_embed_file_name": "optimal_autoencoders/Morgan_4096_AutoEncoder_Checkpoint.pt",
                  "2048_drug_embed_file_name": "optimal_autoencoders/Morgan_2048_AutoEncoder_Checkpoint.pt",
                  "1024_drug_embed_file_name": "optimal_autoencoders/Morgan_1024_AutoEncoder_Checkpoint.pt",
                  "512_drug_embed_file_name": "optimal_autoencoders/Morgan_512_AutoEncoder_Checkpoint.pt"}


def create_cv_folds(train_data, train_attribute_name: str, sample_column_name: str = None, n_folds: int = 10,
                    class_data_index: int = None, class_column_name: str = "primary_disease", subset_type: str = None,
                    stratify: bool = True,
                    seed: int = 42, verbose: bool = False) -> []:
    """
    Creates indices for cross validation, with the options to divide data by cell lines, drugs or both, ensuring that
    they are not shared between the training and validation data, and that each datapoint is used a maximum of once as
    validation data.

    :return: list of tuples of np.arrays in (train_indices, valid_indices) format
    """

    assert subset_type in ["cell_line", "drug", "both", None], "subset_type should be one of: cell_line, drug or both"
    np.random.seed(seed)

    if class_data_index is None:
        # Assume there's only one DataFrame
        class_data = getattr(train_data, train_attribute_name)
    else:
        class_data = getattr(train_data, train_attribute_name)[class_data_index]
    # class_data = train_data.cur_pandas[class_data_index]

    # Stratify folds and maintain class distribution
    skf = StratifiedKFold(n_splits=n_folds)
    all_folds = list(skf.split(X=class_data,
                               y=class_data[class_column_name]))

    all_classes = list(set(class_data[class_column_name].to_list()))

    # Get class proportions and add it to the data frame
    class_proportions = class_data[class_column_name].value_counts(normalize=True)
    if verbose:
        print("Class proportions are:")
        print(class_proportions)
    class_proportions = pandas.DataFrame({class_column_name: class_proportions.index,
                                          'proportions': class_proportions.iloc[:]})

    class_data = class_proportions.merge(class_data, on=class_column_name)

    # Strict or lenient splitting:
    # NOTE: Strict splitting of validation sets most often cannot result in exclusive sets across folds,
    # so the goal here is to minimize the overlap between the validation sets from each fold
    # sample_column_name = "ccl_name"
    if stratify is False:
        # No stratification: Randomly subset 1/n_folds of the sample column name
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        if subset_type == "cell_line" or subset_type == "drug":
            all_groups = list(set(class_data[sample_column_name]))
            all_folds_group_idx = []
            for train_idx, val_idx in kfold.split(X=all_groups):
                all_folds_group_idx.append((train_idx, val_idx))

            # Get indices of_group lines in each fold
            all_folds = []
            for i_fold in range(len(all_folds_group_idx)):
                cur_train_groups = [all_groups[i] for i in all_folds_group_idx[i][0]]
                cur_valid_groups = [all_groups[i] for i in all_folds_group_idx[i][1]]
                all_folds.append((class_data.index[class_data[sample_column_name].isin(cur_train_groups)].to_numpy(),
                                  class_data.index[class_data[sample_column_name].isin(cur_valid_groups)].to_numpy()))
        else:
            match_dict = cell_drug_match(class_data)
            # Divide drugs into n_folds baskets
            match_keys = list(match_dict.keys())
            random.seed(42)
            random.shuffle(match_keys)
            drug_folds = np.array_split(match_keys, n_folds)
            all_folds = []
            for i_fold in range(len(drug_folds)):
                cur_valid_drugs = list(drug_folds[i_fold])
                cur_train_drugs = []
                for j in range(len(drug_folds)):
                    if j != i_fold:
                        cur_train_drugs.append(drug_folds[j])
                cur_train_drugs = [list(cur_train_drugs[i]) for i in range(len(cur_train_drugs))]
                cur_train_drugs = [drug for drug_list in cur_train_drugs for drug in drug_list]

                cur_valid_cells = [match_dict[cur_valid_drug] for cur_valid_drug in cur_valid_drugs]
                cur_valid_cells = [cell for cell_list in cur_valid_cells for cell in cell_list]

                cur_train_cells = [match_dict[cur_train_drug] for cur_train_drug in cur_train_drugs]
                cur_train_cells = [cell for cell_list in cur_train_cells for cell in cell_list]

                all_folds.append((class_data.index[
                                      class_data['ccl_name'].isin(cur_train_cells) &
                                      class_data['cpd_name'].isin(cur_train_drugs)].to_numpy(),
                                  class_data.index[
                                      class_data['ccl_name'].isin(cur_valid_cells) &
                                      class_data['cpd_name'].isin(cur_valid_drugs)].to_numpy()))
                if verbose:
                    print("Train fold:", all_folds[i_fold][0], ", Length:", len(all_folds[i_fold][0]),
                          "\nValid fold:", all_folds[i_fold][1], ", Length:", len(all_folds[i_fold][1]),
                          "\nTotal datapoints:", len(all_folds[i_fold][0]) + len(all_folds[i_fold][1]),
                          "\nValidation/Training Ratio:", len(all_folds[i_fold][1]) / len(all_folds[i_fold][0]),
                          "\nTarget Ratio:", 1 / n_folds)

            # Both cell lines and drugs must be exclusive across folds
            # Assuming cell lines are in ccl_name and drugs are in cpd_name columns
            # May waste some of the data
            # all_cells = list(set(class_data['ccl_name']))
            # remaining_drugs = list(set(class_data['cpd_name']))
            #
            # all_folds_group_idx = []
            # for (cell_train_idx, cell_val_idx), (drug_train_idx, drug_val_idx) in zip(kfold.split(X=all_cells),
            #                                                                           kfold.split(X=remaining_drugs)):
            #     all_folds_group_idx.append([(cell_train_idx, cell_val_idx), (drug_train_idx, drug_val_idx)])
            #
            # all_folds = []
            # for i_fold in range(len(all_folds_group_idx)):
            #     cur_train_cells = [all_cells[i] for i in all_folds_group_idx[i_fold][0][0]]
            #     cur_valid_cells = [all_cells[i] for i in all_folds_group_idx[i_fold][0][1]]
            #     cur_train_drugs = [remaining_drugs[i] for i in all_folds_group_idx[i_fold][1][0]]
            #     cur_valid_drugs = [remaining_drugs[i] for i in all_folds_group_idx[i_fold][1][1]]
            #     all_folds.append((class_data.index[
            #                           class_data['ccl_name'].isin(cur_train_cells) & class_data['cpd_name'].isin(
            #                               cur_train_drugs)].to_numpy(),
            #                       class_data.index[
            #                           class_data['ccl_name'].isin(cur_valid_cells) & class_data['cpd_name'].isin(
            #                               cur_valid_drugs)].to_numpy()))
            #     if verbose:
            #         print("Train fold:", all_folds[i_fold][0], ", Length:", len(all_folds[i_fold][0]),
            #               "\nValid fold:", all_folds[i_fold][1], ", Length:", len(all_folds[i_fold][1]),
            #               "\nTotal datapoints:", len(all_folds[i_fold][0]) + len(all_folds[i_fold][1]),
            #               "\nValidation/Training Ratio:", len(all_folds[i_fold][1]) / len(all_folds[i_fold][0]),
            #               "\nTarget Ratio:", 1 / n_folds)
            if verbose:
                ratio_sum = 0
                for i_fold in range(n_folds):
                    ratio_sum += (len(all_folds[i_fold][1]) / len(all_folds[i_fold][0]))
                print("\nSum Target Ratio:", ratio_sum)

            if verbose:
                all_valid_folds = [fold[1] for fold in all_folds]
                for i_fold in range(len(all_valid_folds) - 1):
                    for j in range(i_fold + 1, len(all_valid_folds)):
                        print(i_fold, j)
                        i_set = set(class_data['ccl_name'].iloc[all_valid_folds[i_fold]])
                        j_set = set(class_data['ccl_name'].iloc[all_valid_folds[j]])
                        cur_intsec = set.intersection(i_set, j_set)
                        print("Length of current validation set overlap:", len(cur_intsec))
                for i_fold in range(len(all_valid_folds) - 1):
                    for j in range(i_fold + 1, len(all_valid_folds)):
                        print(i_fold, j)
                        i_set = set(class_data['cpd_name'].iloc[all_valid_folds[i_fold]])
                        j_set = set(class_data['cpd_name'].iloc[all_valid_folds[j]])
                        cur_intsec = set.intersection(i_set, j_set)
                        print("Length of current validation set overlap:", len(cur_intsec))
                # Calculate total number of samples in each fold setup
                for i_fold in range(len(all_folds)):
                    print("Number of samples in current fold:", len(all_folds[i_fold][0]) + len(all_folds[i_fold][1]))
                print("Total original data:", class_data.shape[0])

    else:
        if subset_type == "cell_line":
            if verbose:
                print(
                    "Strictly splitting training/validation data based on cell lines while maintaining class distributions")

            # Remove lineages that have less than n_folds cell lines, as these would result in data leakage
            class_cell_count = class_data[[class_column_name, sample_column_name]].drop_duplicates().groupby(
                [class_column_name]).agg({sample_column_name: "count"})
            valid_classes = class_cell_count[class_cell_count[sample_column_name] >= n_folds].index.tolist()
            # invalid_classes = class_cell_count[class_cell_count[sample_column_name] < n_folds].index.tolist()
            # class_data.index[class_data[class_column_name].isin(invalid_classes)].to_numpy()

            cur_class_data = class_data[class_data[class_column_name].isin(valid_classes)]

            if verbose:
                print("Must remove lineages that have less cell lines than the number of folds!",
                      "\nRemoved", class_data.shape[0] - cur_class_data.shape[0], "datapoints",
                      "\nRemaining datapoints:", cur_class_data.shape[0])

            # Subset (fully remove) an equal set of cell lines from each class
            all_class_cell_cycles = []
            num_cells_per_fold = []
            for cur_class in valid_classes:
                # Get current class cell lines, sample with the number of folds, and separate from train
                cur_class_cells = list(
                    set(list(cur_class_data[cur_class_data[class_column_name] == cur_class][sample_column_name])))
                cur_class_cells.sort(key=str.lower)
                num_cells_in_fold = int(np.floor(len(cur_class_cells) / n_folds))
                cur_class_cells = cycle(cur_class_cells)
                all_class_cell_cycles.append(cur_class_cells)
                num_cells_per_fold.append(num_cells_in_fold)

            for i_fold in range(n_folds):
                # For each fold, get cells from each cycle (cycle has a memory)
                cur_validation_cells = []
                for cyc, num_cells in zip(all_class_cell_cycles, num_cells_per_fold):
                    cur_validation_cells.append(list(islice(cyc, num_cells)))
                # Flatten the list
                cur_validation_cells = [cur_cell for cur_cells in cur_validation_cells for cur_cell in cur_cells]

                # Determine indices of validation and training sets using validation cell lines (and valid lineages)
                all_folds[i_fold] = (class_data.index[~(class_data[sample_column_name].isin(cur_validation_cells)) &
                                                      class_data[class_column_name].isin(valid_classes)].to_numpy(),
                                     class_data.index[
                                         class_data[sample_column_name].isin(cur_validation_cells)].to_numpy())
                if verbose:
                    print("Train fold:", all_folds[i_fold][0], ", Length:", len(all_folds[i_fold][0]),
                          "\nValid fold:", all_folds[i_fold][1], ", Length:", len(all_folds[i_fold][1]),
                          "\nTotal datapoints:", len(all_folds[i_fold][0]) + len(all_folds[i_fold][1]),
                          "\nValidation/Training Ratio:", len(all_folds[i_fold][1]) / len(all_folds[i_fold][0]),
                          "\nTarget Ratio:", 1 / n_folds)

        elif subset_type == 'drug':
            # TODO: Remove classes that have insufficient samples based on the number of folds
            if verbose:
                print(
                    "Strictly splitting training/validation data based on drugs while maintaining class distributions")

            # Subset (fully remove) an equal set of drugs from each class
            # cur_class = all_classes[0]
            all_class_drug_cycles = []
            num_drugs_per_fold = []
            for cur_class in all_classes:
                # Get current class cell lines, sample with the number of folds, and separate from train
                cur_class_drugs = list(
                    set(list(class_data[class_data[class_column_name] == cur_class][sample_column_name])))
                cur_class_drugs.sort(key=str.lower)
                num_drugs_in_fold = int(np.ceil(len(cur_class_drugs) / n_folds))
                cur_class_drugs = cycle(cur_class_drugs)
                all_class_drug_cycles.append(cur_class_drugs)
                num_drugs_per_fold.append(num_drugs_in_fold)

            for i_fold in range(n_folds):
                # For each fold, get cells from each cycle
                cur_validation_drugs = []
                for cyc, num_drugs in zip(all_class_drug_cycles, num_drugs_per_fold):
                    cur_validation_drugs.append(list(islice(cyc, num_drugs)))
                # Flatten the list
                cur_validation_drugs = [cur_drug for cur_drugs in cur_validation_drugs for cur_drug in cur_drugs]

                # Separate validation cell lines from training cells
                before_train_len = len(all_folds[i_fold][0])
                before_valid_len = len(all_folds[i_fold][1])
                # Determine indices of validation and training sets
                all_folds[i_fold] = (
                    class_data.index[~class_data[sample_column_name].isin(cur_validation_drugs)].to_numpy(),
                    class_data.index[class_data[sample_column_name].isin(cur_validation_drugs)].to_numpy())
                if verbose:
                    print("Train data length before:", before_train_len, ", after:", len(all_folds[i_fold][0]),
                          ", Validation data length before:", before_valid_len, ", after:", len(all_folds[i_fold][1]))
                    print("Train fold:", all_folds[i_fold][0])
                    print("Valid fold:", all_folds[i_fold][1])

        if subset_type == "both":
            if verbose:
                print(
                    "Strictly splitting training/validation data based on both cell lines and drugs while maintaining class distributions")

            # Remove lineages that have less than n_folds cell lines, as these would result in data leakage
            class_cell_count = class_data[[class_column_name, 'ccl_name']].drop_duplicates().groupby(
                [class_column_name]).agg({'ccl_name': "count"})
            valid_classes = class_cell_count[class_cell_count['ccl_name'] >= n_folds].index.tolist()
            cur_class_data = class_data[class_data[class_column_name].isin(valid_classes)]

            num_unique_drugs = len(set(cur_class_data['cpd_name']))
            num_unique_cells = len(set(cur_class_data['ccl_name']))
            # Order by drug x cell line completion within each class (ccl and cpd each have their own completion scores)
            ccl_completion = cur_class_data[['ccl_name', 'cpd_name']].drop_duplicates().groupby(
                ['ccl_name']).agg({'cpd_name': 'count'})
            ccl_completion['cpd_name'] = ccl_completion['cpd_name'] / num_unique_drugs
            ccl_completion = ccl_completion.rename(columns={"cpd_name": "ccl_completion"})

            cpd_completion = cur_class_data[['ccl_name', 'cpd_name']].drop_duplicates().groupby(
                ['cpd_name']).agg({'ccl_name': 'count'})
            cpd_completion['ccl_name'] = cpd_completion['ccl_name'] / num_unique_cells
            cpd_completion = cpd_completion.rename(columns={"ccl_name": "cpd_completion"})

            cur_class_data = cur_class_data.merge(ccl_completion, left_on='ccl_name', right_index=True)
            cur_class_data = cur_class_data.merge(cpd_completion, left_on='cpd_name', right_index=True)
            # Order class data by completion
            cur_class_data = cur_class_data.sort_values(['ccl_completion', 'cpd_completion'], ascending=False)

            # Set is an unordered data structure, but we can change the order of the list using dict keys (Python 3.7+)
            # all_valid_classes = cur_class_data.sort_values(['proportions'], ascending=False)[class_column_name].tolist()
            # valid_classes = list(dict.fromkeys(all_valid_classes))

            if verbose:
                print("Must remove classes that have less cell lines than the number of folds!",
                      "\nRemoved", class_data.shape[0] - cur_class_data.shape[0], "datapoints",
                      "\nRemaining datapoints:", cur_class_data.shape[0])

            # Subset (fully remove) an equal set of cell lines from each class
            all_class_cell_cycles = []
            num_cells_per_fold = []
            for cur_class in valid_classes:
                # Get current class cell lines ordered by it's cell line completion score
                cur_class_cells = list(
                    dict.fromkeys(cur_class_data[cur_class_data[class_column_name] == cur_class]['ccl_name'].tolist()))
                # cur_class_cells.sort(key=str.lower)
                num_cells_in_fold = int(np.floor(len(cur_class_cells) / n_folds))
                cur_class_cells_cycle = cycle(cur_class_cells)
                all_class_cell_cycles.append(cur_class_cells_cycle)
                num_cells_per_fold.append(num_cells_in_fold)

            remaining_drugs = list(dict.fromkeys(cur_class_data['cpd_name'].tolist()))
            num_drugs_per_fold = int(np.floor(len(remaining_drugs) / n_folds))

            # remaining_drugs.sort()
            # random.seed(seed)
            # random.shuffle(remaining_drugs)
            # all_drugs_cycle = cycle(remaining_drugs)
            # all_class_drugs = []
            # for i in range(n_folds):
            #     all_class_drugs.append(list(islice(all_drugs_cycle, num_drugs_per_fold)))

            for i_fold in range(n_folds):
                # For each fold, get cells from each cycle (cycle is an infinite iterator and has a memory)
                cur_validation_cells = []
                for cyc, num_cells in zip(all_class_cell_cycles, num_cells_per_fold):
                    cur_validation_cells.append(list(islice(cyc, num_cells)))
                # Flatten the list
                cur_validation_cells = [cur_cell for cur_cells in cur_validation_cells for cur_cell in cur_cells]

                # Subset validation drugs, ensuring it matches with current fold's validation cell lines and isn't used
                # in other folds. First get available drugs for this fold, ordered by cpd_completion score
                cur_avail_drugs = list(
                    dict.fromkeys(cur_class_data[cur_class_data['ccl_name'].isin(cur_validation_cells) &
                                                 cur_class_data['cpd_name'].isin(remaining_drugs)].sort_values(
                        ['cpd_completion'], ascending=True)['cpd_name'].tolist()))
                # Randomly select the appropriate number of drugs
                # cur_validation_drugs = random.sample(cur_avail_drugs, num_drugs_per_fold)
                # Select the drugs with highest drug completion score
                cur_validation_drugs = cur_avail_drugs[0:num_drugs_per_fold]

                # Remove selected drugs from remaining drugs
                for selected_drug in cur_validation_drugs:
                    remaining_drugs.remove(selected_drug)

                # Note: The same approach for drugs can be used for cell lines, instead of cycling through the iterator

                # total_drugs_matched = 0
                # Test for availability in the dataset
                # len(class_data.index[class_data['ccl_name'].isin(cur_validation_cells) &
                #                      class_data['cpd_name'].isin(cur_validation_drugs) &
                #                      class_data[class_column_name].isin(valid_classes)])

                # Determine indices of validation and training sets using validation cell lines (and valid lineages)
                all_folds[i_fold] = (class_data.index[~class_data["ccl_name"].isin(cur_validation_cells) &
                                                      ~class_data['cpd_name'].isin(cur_validation_drugs) &
                                                      class_data[class_column_name].isin(valid_classes)].to_numpy(),
                                     class_data.index[class_data['ccl_name'].isin(cur_validation_cells) &
                                                      class_data['cpd_name'].isin(cur_validation_drugs)].to_numpy())

                if verbose:
                    print("Train fold:", all_folds[i_fold][0], ", Length:", len(all_folds[i_fold][0]),
                          "\nValid fold:", all_folds[i_fold][1], ", Length:", len(all_folds[i_fold][1]),
                          "\nTotal datapoints:", len(all_folds[i_fold][0]) + len(all_folds[i_fold][1]),
                          "\nValidation/Training Ratio:", len(all_folds[i_fold][1]) / len(all_folds[i_fold][0]),
                          "\nTarget Ratio:", 1 / n_folds)
            if verbose:
                ratio_sum = 0
                for i_fold in range(n_folds):
                    ratio_sum += (len(all_folds[i_fold][1]) / len(all_folds[i_fold][0]))
                print("\nSum Target Ratio:", ratio_sum)
    if verbose:
        # Measure overlap among folds. Ideally, validation folds should be exclusive and there shouldn't be any overlap
        # between training and validation sets of the same fold setup
        all_valid_folds = [fold[1] for fold in all_folds]
        for i_fold in range(len(all_valid_folds) - 1):
            for j in range(i_fold + 1, len(all_valid_folds)):
                print(i_fold, j)
                cur_intsec = set.intersection(set(all_valid_folds[i_fold]), set(all_valid_folds[j]))
                print("Length of current validation set overlap:", len(cur_intsec))

        if subset_type != "both":
            # Test for sample_column_name overlap
            for i_fold in range(len(all_valid_folds) - 1):
                for j in range(i_fold + 1, len(all_valid_folds)):
                    print(i_fold, j)
                    i_set = set(class_data[sample_column_name].iloc[all_valid_folds[i_fold]])
                    j_set = set(class_data[sample_column_name].iloc[all_valid_folds[j]])
                    cur_intsec = set.intersection(i_set, j_set)
                    print("Length of current validation set overlap:", len(cur_intsec))
            # Test training and validation overlap
            for i_fold in range(n_folds):
                train_set = set(class_data[sample_column_name].iloc[all_folds[i_fold][0]])
                valid_set = set(class_data[sample_column_name].iloc[all_folds[i_fold][1]])
                cur_intsec = set.intersection(train_set, valid_set)
                print("Training-Validation intersection:", cur_intsec)

        else:
            all_valid_folds = [fold[1] for fold in all_folds]
            # Exclusivity by both drugs and cell lines
            # Check for overlap between cell lines in all validation sets
            for i_fold in range(len(all_valid_folds) - 1):
                for j in range(i_fold + 1, len(all_valid_folds)):
                    print(i_fold, j)
                    i_set = set(class_data['ccl_name'].iloc[all_valid_folds[i_fold]])
                    j_set = set(class_data['ccl_name'].iloc[all_valid_folds[j]])
                    cur_intsec = set.intersection(i_set, j_set)
                    print("Length of cell line validation set overlap:", len(cur_intsec))
            for i_fold in range(len(all_valid_folds) - 1):
                for j in range(i_fold + 1, len(all_valid_folds)):
                    print(i_fold, j)
                    i_set = set(class_data['cpd_name'].iloc[all_valid_folds[i_fold]])
                    j_set = set(class_data['cpd_name'].iloc[all_valid_folds[j]])
                    cur_intsec = set.intersection(i_set, j_set)
                    print("Length of drug validation set overlap:", len(cur_intsec))

            # Test training and validation overlap
            for i_fold in range(n_folds):
                train_set = set(class_data['cpd_name'].iloc[all_folds[i_fold][0]])
                valid_set = set(class_data['cpd_name'].iloc[all_folds[i_fold][1]])
                cur_intsec = set.intersection(train_set, valid_set)
                print("Training-Validation drug intersection:", cur_intsec)

                train_set = set(class_data['ccl_name'].iloc[all_folds[i_fold][0]])
                valid_set = set(class_data['ccl_name'].iloc[all_folds[i_fold][1]])
                cur_intsec = set.intersection(train_set, valid_set)
                print("Training-Validation cell line intersection:", cur_intsec)

    return all_folds


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


def drp_load_datatypes(train_file: str, module_list: [str], PATH: str, file_name_dict: {}, device: str = 'gpu',
                       _pretrain: bool = False, load_autoencoders: bool = False, verbose: bool = False):
    """
    This function loads encoder's data and torch models indicated in the module_list argument. It loads
    drug response data from the train_file at the PATH directory.

    :param train_file: File that contains drug response data e.g. from CTRPv2
    :param module_list: A list of strings containing a combination of drug, mut, cnv, exp and prot
    :param PATH: The full path to the directory that contains the autoencoders and input data
    :param file_name_dict: dictionary containing the name of each file
    :param device: either cpu or gpu, for loading the data
    :return: list of input data, list of auto-encoders, list of key column name for each data type
    """
    data_list = []
    autoencoder_list = []
    key_columns = []
    gpu_locs = []

    if device == "gpu":
        to_gpu = True
    else:
        to_gpu = False

    if _pretrain is True:
        pretrain = "pretrain_"
        print("Loading pretrained auto-encoders for EXP and CNV!")
    else:
        pretrain = ""

    # TODO Make the following shorter (for loop perhaps?)
    # The following if/else statements ensure that the data has a specific order
    if "drug" in module_list:
        # TODO: Implement a way to distinguish training dataset, e.g. CTRP vs GDSC
        # Putting drug first ensures that the drug data and module is in a fixed index, i.e. 0
        if verbose:
            print("Loading drug data...")
        cur_data = DRCurveData(path=PATH, file_name=train_file, key_column="ccl_name", class_column="primary_disease",
                               morgan_column="morgan",
                               target_column="area_above_curve", to_gpu=to_gpu)
        data_list.append(cur_data)
        if verbose:
            print("Drug data width:", cur_data.width())
        if load_autoencoders:
            autoencoder_list.append(torch.load(PATH + file_name_dict[str(cur_data.width()) + "_drug_embed_file_name"],
                                               map_location=torch.device('cpu')))
        key_columns.append("ccl_name")
        if device == "multi_gpu":
            gpu_locs.append(0)
        elif device == "cpu":
            gpu_locs.append(None)
        else:
            gpu_locs.append(0)
    else:
        sys.exit("Drug data must be provided for drug-response prediction")

    if "mut" in module_list:
        if verbose:
            print("Loading mutational data from", file_name_dict["mut_file_name"])
        cur_data = OmicData(path=PATH, omic_file_name=file_name_dict["mut_file_name"], to_gpu=to_gpu)
        data_list.append(cur_data)
        if verbose:
            print("Mut data width:", cur_data.width())
        if load_autoencoders:
            autoencoder_list.append(
                torch.load(PATH + file_name_dict["mut_embed_file_name"], map_location=torch.device('cpu')))
        key_columns.append("stripped_cell_line_name")
        if device == "multi_gpu":
            gpu_locs.append(0)
        elif device == "cpu":
            gpu_locs.append(None)
        else:
            gpu_locs.append(0)

    if "cnv" in module_list:
        if verbose:
            print("Loading copy number variation data from", file_name_dict["cnv_file_name"])
        cur_data = OmicData(path=PATH, omic_file_name=file_name_dict["cnv_file_name"], to_gpu=to_gpu)
        data_list.append(cur_data)
        if verbose:
            print("CNV data width:", cur_data.width())
        if load_autoencoders:
            autoencoder_list.append(
                torch.load(PATH + file_name_dict[pretrain + "cnv_embed_file_name"], map_location=torch.device('cpu')))
        key_columns.append("stripped_cell_line_name")
        if device == "multi_gpu":
            gpu_locs.append(0)
        elif device == "cpu":
            gpu_locs.append(None)
        else:
            gpu_locs.append(0)

    if "exp" in module_list:
        if verbose:
            print("Loading gene expression data from", file_name_dict["exp_file_name"])
        cur_data = OmicData(path=PATH, omic_file_name=file_name_dict["exp_file_name"], to_gpu=to_gpu)
        data_list.append(cur_data)
        if verbose:
            print("Exp data width:", cur_data.width())
        if load_autoencoders:
            autoencoder_list.append(
                torch.load(PATH + file_name_dict[pretrain + "exp_embed_file_name"], map_location=torch.device('cpu')))
        key_columns.append("stripped_cell_line_name")
        if device == "multi_gpu":
            gpu_locs.append(0)
        elif device == "cpu":
            gpu_locs.append(None)
        else:
            gpu_locs.append(0)

    if "prot" in module_list:
        if verbose:
            print("Loading protein quantity data from", file_name_dict["prot_file_name"])
        cur_data = OmicData(path=PATH, omic_file_name=file_name_dict["prot_file_name"], to_gpu=to_gpu)
        data_list.append(cur_data)
        if verbose:
            print("Prot data width:", cur_data.width())
        if load_autoencoders:
            autoencoder_list.append(
                torch.load(PATH + file_name_dict["prot_embed_file_name"], map_location=torch.device('cpu')))

        key_columns.append("stripped_cell_line_name")
        if device == "multi_gpu":
            gpu_locs.append(0)
        elif device == "cpu":
            gpu_locs.append(None)
        else:
            gpu_locs.append(0)

    return data_list, autoencoder_list, key_columns, gpu_locs


def drp_create_datasets(data_list, key_columns, n_folds: int = 10,
                        drug_index: int = 0, drug_dr_column: str = "area_above_curve",
                        class_column_name: str = "primary_disease", subset_type: str = None, stratify: bool = True,
                        test_drug_data=None,
                        bottleneck_keys: [str] = None, test_mode: bool = False, to_gpu: bool = False,
                        verbose: bool = False) -> Tuple[PairData, list]:
    """
    This function takes a list of data for each auto-encoder and passes it to the PairData function. It creates
    a reproducible, lenient validation split of the data and provides the indices as well as the paired data
    """
    if bottleneck_keys is not None:
        assert len(bottleneck_keys) > 0, "Must have at least one key in the bottleneck keys argument"
    if test_drug_data is not None:
        # Only the drug data is different
        data_list[0] = test_drug_data
        if bottleneck_keys is not None:
            test_data = PairData(data_module_list=data_list, key_columns=key_columns, drug_index=drug_index,
                                 drug_dr_column=drug_dr_column, bottleneck_keys=bottleneck_keys,
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
    if bottleneck_keys is not None:
        train_data = PairData(data_module_list=data_list, key_columns=key_columns, key_attribute_names=["data_info"],
                              drug_index=drug_index, drug_dr_column=drug_dr_column, bottleneck_keys=bottleneck_keys,
                              test_mode=test_mode, to_gpu=to_gpu)

    else:
        train_data = PairData(data_module_list=data_list, key_columns=key_columns, key_attribute_names=["data_info"],
                              drug_index=drug_index, drug_dr_column=drug_dr_column, test_mode=test_mode, to_gpu=to_gpu)

    # print("Length of selected data set is:", len(train_data))

    # if (train_idx is not None) and (valid_idx is not None):
    #     pass
    # else:
    # Obtain training indices that will be used for validation
    if subset_type == "cell_line":
        cv_folds = create_cv_folds(train_data=train_data, train_attribute_name="data_infos",
                                   sample_column_name="ccl_name", n_folds=n_folds, class_data_index=drug_index,
                                   subset_type="cell_line", stratify=stratify, class_column_name=class_column_name,
                                   seed=42,
                                   verbose=verbose)
    elif subset_type == "drug":
        cv_folds = create_cv_folds(train_data=train_data, train_attribute_name="data_infos",
                                   sample_column_name="cpd_name", n_folds=n_folds, class_data_index=drug_index,
                                   subset_type="drug", stratify=stratify, class_column_name=class_column_name, seed=42,
                                   verbose=verbose)
    else:
        # Sample column name is ignored
        cv_folds = create_cv_folds(train_data=train_data, train_attribute_name="data_infos",
                                   n_folds=n_folds, class_data_index=drug_index,
                                   subset_type="both", stratify=stratify, class_column_name=class_column_name,
                                   seed=42,
                                   verbose=verbose)

    # num_train = len(train_data)
    # indices = list(range(num_train))
    # np.random.seed(42)
    # np.random.shuffle(indices)
    # split = int(np.floor(valid_size * num_train))
    # train_idx, valid_idx = indices[split:], indices[:split]
    #
    # # define samplers for obtaining training and validation batches
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    return train_data, cv_folds


def drp_main_prep(module_list, train_file, path="/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/",
                  device="gpu", bottleneck: bool = False, bottleneck_file: str = "bottleneck_celllines.csv"):
    """
    This function takes a list of data types and uses them to load data and auto-encoders via the
     drp_load_datatypes function, and also extracts the encoders from the auto-encoders. Furthermore, it
     determines the
    """

    assert "drug" in module_list, "Drug data must be provided for drug-response prediction (training or testing)"
    assert len(module_list) > 1, "Data types to be used must be indicated by: mut, cnv, exp, prot and drug"

    # if bottleneck:
    #     try:
    #         bottleneck_keys = pd.read_csv(bottleneck_file)
    #     except:
    #         exit("Could not read bottleneck keys from indicated file")

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

    data_list, autoencoder_list, key_columns, \
    gpu_locs = drp_load_datatypes(train_file,
                                  module_list=final_address,
                                  PATH=path,
                                  file_name_dict=file_name_dict,
                                  device=device)

    final_address = "_".join(final_address)

    # Convert autoencoders to encoders; assuming fixed encoder paths
    encoder_list = [ExtractEncoder(autoencoder) for autoencoder in autoencoder_list]

    subset_data = data_list
    subset_encoders = encoder_list
    subset_keys = key_columns

    return device, final_address, subset_data, subset_keys, subset_encoders, data_list, key_columns
