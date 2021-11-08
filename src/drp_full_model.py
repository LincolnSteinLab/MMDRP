# This script contains the training pipeline for the DRP module of the FULL DRP MODEL. It can take any of the omics data
# plus the drug auto-encoders, extract the trained encoders and use associated data types to train the DRP module.
import argparse
import csv
import glob
import os
import time
import warnings
from pathlib import Path
from sklearn.linear_model import ElasticNetCV, SGDRegressor
import numpy as np
import json
import ray
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
import torch.nn as nn
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch

from CustomFunctions import AverageMeter, ProgressMeter, MyLoggerCreator, WriterCLIReporter
from DRPPreparation import drp_load_datatypes, drp_create_datasets
from DataImportModules import DRCurveData
from Models import DrugResponsePredictor
from ModuleLoader import ExtractEncoder
from TrainFunctions import drp_train, cross_validate, gnn_drp_train, elasticnet_drp_train
from TuneTrainables import FullModelTrainable, DRPTrainable, file_name_dict

import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
# torch.autograd.set_detect_anomaly(True)
# NUM_CPU = multiprocessing.cpu_count()
# print("# of CPUs:", NUM_CPU)
NUM_CPU = 0
use_cuda = torch.cuda.is_available()

if use_cuda:
    print('CUDA is available. # of GPUs:', torch.cuda.device_count())
print("Current working directory is:", os.getcwd())

activation_dict = nn.ModuleDict({
    'lrelu': nn.LeakyReLU(),
    'prelu': nn.PReLU(),
    'relu': nn.ReLU(),
    'none': nn.Identity()
})

morgan_param_ranges = {
    "morgan_first_layer_size": tune.randint(2 ** 6, 2 ** 9),
    "morgan_code_layer_size": tune.randint(2 ** 5, 2 ** 8),
    "morgan_num_layers": tune.randint(2, 3),
}
mut_param_ranges = {
    "mut_first_layer_size": tune.randint(2 ** 7, 2 ** 10),
    "mut_code_layer_size": tune.randint(2 ** 6, 2 ** 9),
    "mut_num_layers": tune.randint(2, 4)
}
cnv_param_ranges = {
    "cnv_first_layer_size": tune.randint(2 ** 9, 6000),
    "cnv_code_layer_size": tune.randint(2 ** 8, 4000),
    "cnv_num_layers": tune.randint(2, 4)
}
exp_param_ranges = {
    "exp_first_layer_size": tune.randint(2 ** 9, 6000),
    "exp_code_layer_size": tune.randint(2 ** 8, 4000),
    "exp_num_layers": tune.randint(2, 4)
}
prot_param_ranges = {
    "prot_first_layer_size": tune.randint(2 ** 9, 4000),
    "prot_code_layer_size": tune.randint(2 ** 8, 2500),
    "prot_num_layers": tune.randint(2, 4)
}
mirna_param_ranges = {
    "mirna_first_layer_size": tune.randint(2 ** 7, 2 ** 10),
    "mirna_code_layer_size": tune.randint(2 ** 6, 2 ** 9),
    "mirna_num_layers": tune.randint(2, 3)
}
hist_param_ranges = {
    "hist_first_layer_size": tune.randint(2 ** 3, 2 ** 5),
    "hist_code_layer_size": tune.randint(2 ** 6, 2 ** 9),
    "hist_num_layers": tune.randint(2, 3)
}
rppa_param_ranges = {
    "rppa_first_layer_size": tune.randint(2 ** 5, 2 ** 8),
    "rppa_code_layer_size": tune.randint(2 ** 4, 2 ** 7),
    "rppa_num_layers": tune.randint(2, 3)
}
metab_param_ranges = {
    "metab_first_layer_size": tune.randint(2 ** 5, 2 ** 8),
    "metab_code_layer_size": tune.randint(2 ** 4, 2 ** 7),
    "metab_num_layers": tune.randint(2, 3)
}

data_types = ['drug', 'mut', 'cnv', 'exp', 'prot', 'mirna', 'hist', 'rppa', 'metab']
code_layer_size_names = [data_type + '_code_layer_size' for data_type in data_types] + ['gnn_out_channels']

lmf_param_ranges = {
    "lmf_output_dim": tune.randint(2 ** 6, 2 ** 10),
    "lmf_rank": tune.randint(2 ** 2, 2 ** 6),
}

gnn_param_ranges = {
    # "gnn_in_channels": tune.randint(2 ** 6, 2 ** 10),
    "gnn_hidden_channels": tune.randint(2 ** 6, 2 ** 9),
    "gnn_out_channels": tune.randint(2 ** 5, 2 ** 11),
    # "gnn_edge_dim": tune.randint(2 ** 2, 2 ** 4),
    "gnn_num_layers": tune.randint(2, 4),
    "gnn_num_timesteps": tune.randint(2, 4),
    "gnn_dropout": tune.uniform(0.05, 0.3),
}


def test(args):
    """
    There are 15 omic data combinations (based on 4 data types). This function will go through each file and will
    test the model on the requested dataset if the "best_checkpoint" file exists. This function loads auto-encoder inputs
    only once and presents them accordingly at test time.
    """
    num_gpus = int(torch.cuda.device_count())
    assert num_gpus > 1, "More than 1 GPU is needed if both Exp and CNV data types are requested"

    # All checkpoint file names
    # Find all checkpoint files in the relevant directory based on name_tag
    all_checkpoint_files = glob.glob(args.name_tag + "/*best_checkpoint*")
    print("all_checkpoint_files:", all_checkpoint_files)

    if args.machine == "cluster":
        PATH = "/u/ftaj/anaconda3/envs/Python/Data/DRP_Training_Data/"
    elif args.machine == "mist":
        PATH = "/scratch/l/lstein/ftaj/"
    else:
        PATH = "/Data/DRP_Training_Data/"

    if os.path.exists(args.name_tag + '_evaluation/') is not True:
        os.mkdir(args.name_tag + '_evaluation/')

    # Load data
    module_list = ["drug", "mut", "cnv", "exp", "prot"]
    # TODO training data shouldn't be required for this part,
    data_list, autoencoder_list, key_columns, state_paths, gpu_locs, final_address = drp_load_datatypes(args,
                                                                                                        module_list=module_list,
                                                                                                        PATH=PATH,
                                                                                                        file_name_dict=file_name_dict)
    # Load all encoders
    # Convert autoencoders to encoders; assuming fixed encoder paths
    encoder_list = [ExtractEncoder(autoencoder, state_path) for autoencoder, state_path in
                    zip(autoencoder_list, state_paths)]

    test_data = DRCurveData(path=PATH, file_name=args.test, key_column="ccl_name",
                            morgan_column="morgan",
                            target_column="area_above_curve")

    # name_tag = args.name_tag
    # data_time = AverageMeter('Data', ':6.3f')

    criterion = nn.L1Loss().cuda()
    # file_name = "drp_drug_mut_cnv_CTRP_CompleteBottleNeck_EncoderTrain_best_checkpoint.pth.tar"
    # cur_data_types = file_name.split(name_tag)[0].replace("drp_", "").split('_')
    # file_name.split("_CompleteBottleNeck_")[0].replace("drp_", "").split('_')
    for file_name in all_checkpoint_files:
        # Extract data type names from the checkpoint file name
        cur_file_name = file_name.split('/')[1]
        cur_data_types = cur_file_name.split(args.name_tag)[0].replace("drp_", "").split('_')
        del cur_data_types[-1]
        print("Current data types are:", cur_data_types)
        # TODO We know that the order of file types is consistent with other parts of the model
        # Subset encoder list based on data types (drug data is always present)
        subset_encoders = [encoder_list[0]]
        subset_gpu_locs = [0]
        subset_data = [data_list[0]]
        subset_keys = ["ccl_name"]
        required_data_indices = [0]
        if "mut" in cur_data_types:
            subset_encoders.append(encoder_list[1])
            subset_gpu_locs.append(gpu_locs[1])
            subset_data.append(data_list[1])
            subset_keys.append(key_columns[1])
            required_data_indices.append(1)
        if "cnv" in cur_data_types:
            subset_encoders.append(encoder_list[2])
            subset_gpu_locs.append(gpu_locs[2])
            subset_data.append(data_list[2])
            subset_keys.append(key_columns[2])
            required_data_indices.append(2)
        if "exp" in cur_data_types:
            subset_encoders.append(encoder_list[3])
            subset_gpu_locs.append(gpu_locs[3])
            subset_data.append(data_list[3])
            subset_keys.append(key_columns[3])
            required_data_indices.append(3)
        if "prot" in cur_data_types:
            subset_encoders.append(encoder_list[4])
            subset_gpu_locs.append(gpu_locs[4])
            subset_data.append(data_list[4])
            subset_keys.append(key_columns[4])
            required_data_indices.append(4)

        final_address = "_".join(cur_data_types)
        print("Length of current encoder list:", len(subset_encoders))

        print("Final address is:", final_address)

        # TODO: Can we have one function call determined by args.bottleneck instead of using and if/else statement?
        if bool(int(args.bottleneck)) is True:
            print("Creating bottle-necked data...")
            test_loader = drp_create_datasets(data_list, key_columns, int(args.batch_size), drug_index=0,
                                              drug_dr_column="area_above_curve", test_drug_data=test_data,
                                              bottleneck=True,
                                              required_data_indices=required_data_indices)
        else:
            # Subset of data and associated keys change for each combination, but not the test drug response curve data
            test_loader = drp_create_datasets(subset_data, subset_keys, int(args.batch_size), drug_index=0,
                                              drug_dr_column="area_above_curve", test_drug_data=test_data)
        # Create the model
        cur_model = DrugResponsePredictor(layer_sizes=[2048, 1024, 512, 1], gpu_locs=subset_gpu_locs,
                                          encoder_requires_grad=args.encoder_freeze)

        # Load the checkpoint
        # try:
        checkpoint = torch.load(file_name)
        cur_model.load_state_dict(checkpoint['cur_model'], strict=True)
        print("Current model's loss on training set:", checkpoint['loss'])
        # except:
        #     print(file_name + " does not exist, moving on to the next file...")
        #     continue

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            len(test_loader),
            [batch_time, losses],
            prefix='Test: ')

        # switch to evaluate mode
        cur_model.eval()

        total_time = time.time()
        print("Starting evaluation...")
        print("Data types:", str(cur_data_types))

        # Evaluate model with the given dataset
        cell_line_names = []
        drug_names = []
        actual_targets = []
        all_outputs = []
        test_losses = []
        with torch.no_grad():
            batch_start = time.time()
            end = time.time()
            for i, local_batch in enumerate(test_loader, 0):
                cur_dr_data = local_batch[1]  # This is a tuple of torch.Tensor
                cur_dr_target = local_batch[2]  # This is a torch.Tensor

                cur_dr_data = [cur_data.float() for cur_data in cur_dr_data]
                # cur_dr_data = [cur_data.to(device) for cur_data in cur_dr_data]
                cur_dr_target = cur_dr_target.float()
                cur_dr_target = cur_dr_target.cuda()

                # forward + loss measurement
                output = cur_model(cur_dr_data)
                test_loss = criterion(output, cur_dr_target)

                # Append to above-defined lists
                cell_line_names.append(local_batch[0]["cell_line_name"])
                drug_names.append(local_batch[0]["drug_name"])
                actual_targets.append(cur_dr_target.detach().cpu().numpy())
                all_outputs.append(output.detach().cpu().numpy())
                test_losses.append(test_loss.item())

                # Measure R-squared and record loss
                losses.update(test_loss.item(), cur_dr_target.shape[0])

                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 100 == 0:  # print every 100 mini-batches, average loss
                    progress.display(i)
                    print('[%d] loss: %.6f in %3d seconds' %
                          (i, test_loss.item(), time.time() - batch_start))
                    batch_start = time.time()
            duration = time.time() - total_time
            print('Finished evaluation in', str(duration), "seconds")
            print("Total Average MAE Loss on", len(test_loader), "batches:", np.asarray(test_losses).mean())

            # Flatten the list of numpy arrays into a single list
            cell_line_names = np.concatenate(cell_line_names).ravel().tolist()
            drug_names = np.concatenate(drug_names).ravel().tolist()
            all_outputs = np.concatenate(all_outputs).ravel().tolist()
            actual_targets = np.concatenate(actual_targets).ravel().tolist()
            # test_losses = np.concatenate(test_losses).ravel().tolist()

            print("Length of cell_line_names:", len(cell_line_names))
            print("Length of drug_names:", len(drug_names))
            print("Length of all_outputs:", len(all_outputs))
            print("Length of actual_targets:", len(actual_targets))
            # print("Length of test_losses:", len(test_losses))

            # TODO: Move the saving part to outside of function
            print("Final file name:",
                  args.name_tag + '_evaluation/drp_' + str(final_address) + "_" + args.name_tag + '_' +
                  file_name.split(args.test)[0] + '_evaluation.csv')
            # Create a dataframe of actual + predicted + loss score for each cell line x drug combination
            evaluation = pd.DataFrame({"cell_line_name": cell_line_names,
                                       "drug_name": drug_names,
                                       "actual_target": actual_targets,
                                       "predicted_target": all_outputs})
            # "loss_score": test_losses})
            # Save
            evaluation.to_csv(args.name_tag + '_evaluation/drp_' + str(final_address) + '_' + args.name_tag +
                              '_' + args.test.split('_')[0] + '_evaluation.csv', sep="\t")


def train_full_model(args, config):
    """
    Takes the data combo and associated model from the main_worker function and trains the model for specified epochs.
    :param args:
    :return:
    """
    max_epochs = int(args.max_num_epochs)

    local_dir = "/scratch/l/lstein/ftaj/"
    if bool(int(args.full)) is True:
        tag = "FullModel"
    else:
        tag = "ResponseOnly"

    # Create the results directory and file paths
    experiment_name = "HyperOpt_DRP_" + tag + '_' + "_".join(args.data_types) + '_' + args.name_tag
    result_dir = local_dir + "/CV_Results/" + experiment_name
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    print("Result directory is:", result_dir)

    # Must create a logger that points to a writable directory
    # if bool(int(args.cross_validate)) is True:
    if bool(int(args.full)) is True:
        # cur_trainable = FullModelTrainable(config=config, logger_creator=MyLoggerCreator)
        cur_trainable = tune.with_parameters(FullModelTrainable,
                                             train_file=args.train_file,
                                             data_types='_'.join(args.data_types),
                                             bottleneck=bool(int(args.bottleneck)),
                                             n_folds=int(args.n_folds),
                                             max_epochs=int(args.max_num_epochs),
                                             encoder_train=args.encoder_freeze,
                                             cv_subset_type=args.cv_subset_type,
                                             stratify=bool(int(args.stratify)),
                                             random_morgan=bool(int(args.random_morgan)),
                                             merge_method=args.merge_method,
                                             loss_type=args.loss_type,
                                             one_hot_drugs=bool(int(args.one_hot_drugs)),
                                             transform=args.transform,
                                             min_dr_target=args.min_dr_target,
                                             max_dr_target=args.max_dr_target,
                                             to_gpu=False if args.baseline_only else True
                                             )
        cur_trainable = cur_trainable(config=config, logger_creator=MyLoggerCreator)
    else:
        # cur_trainable = DRPTrainable(config=config, logger_creator=MyLoggerCreator)
        cur_trainable = tune.with_parameters(DRPTrainable,
                                             train_file=args.train_file,
                                             data_types='_'.join(args.data_types),
                                             bottleneck=bool(int(args.bottleneck)),
                                             pretrain=bool(int(args.pretrain)),
                                             n_folds=int(args.n_folds),
                                             max_epochs=int(args.max_num_epochs),
                                             encoder_train=args.encoder_freeze,
                                             cv_subset_type=args.cv_subset_type,
                                             stratify=bool(int(args.stratify)),
                                             random_morgan=bool(int(args.random_morgan)),
                                             merge_method=args.merge_method,
                                             loss_type=args.loss_type,
                                             one_hot_drugs=bool(int(args.one_hot_drugs)),
                                             gnn_drug=True if 'gnndrug' in args.data_types else False,
                                             transform=args.transform,
                                             min_dr_target=args.min_dr_target,
                                             max_dr_target=args.max_dr_target,
                                             omic_standardize=args.omic_standardize,
                                             to_gpu=False if args.baseline_only else True
                                             )

        cur_trainable = cur_trainable(config=config, logger_creator=MyLoggerCreator)

    cur_model, criterion = cur_trainable.cur_model, cur_trainable.criterion
    cur_train_data, cur_cv_folds = cur_trainable.cur_train_data, cur_trainable.cur_cv_folds

    print("Total samples for training:", str(len(cur_train_data)))

    if args.train_only:
        # cur_cv_folds = [cur_cv_folds[0]]
        print("Skipping cross-validation...")

    name_tag = args.name_tag
    if args.baseline_only:
        assert '_'.join(args.data_types) == "drug_exp",\
            "Baseline model only available for drug and exp data types"
        del cur_model
        del criterion

        cur_model = SGDRegressor(penalty='elasticnet')
        criterion = None
        cur_train_function = elasticnet_drp_train
        name_tag = "Baseline_ElasticNet"
    else:
        cur_train_function = gnn_drp_train if 'gnndrug' in args.data_types else drp_train

    epoch_save_folder = "/scratch/l/lstein/ftaj/EpochResults/CrossValidation/" + name_tag
    Path(epoch_save_folder).mkdir(parents=True, exist_ok=True)
    print("Saving epoch results to:", epoch_save_folder)

    # Perform cross validation using the model with the given config
    start_time = time.time()
    final_model, \
    avg_train_losses, \
    avg_valid_losses, \
    avg_untrained_losses, \
    max_final_epoch = cross_validate(train_data=cur_train_data,
                                     train_function=cur_train_function,
                                     cv_folds=cur_cv_folds,
                                     batch_size=config['batch_size'],
                                     cur_model=cur_model,
                                     criterion=criterion,
                                     max_epochs=max_epochs,
                                     patience=10,
                                     train_only=args.train_only,
                                     final_full_train=args.final_train,
                                     learning_rate=config['lr'],
                                     delta=0.001,
                                     NUM_WORKERS=0,
                                     theoretical_loss=False,
                                     omic_standardize=args.omic_standardize,
                                     save_epoch_results=False,
                                     redo_validation=args.redo_validation,
                                     epoch_save_folder=epoch_save_folder,
                                     save_model=False if args.baseline_only else True,
                                     save_model_frequency=5,
                                     save_model_path=epoch_save_folder,
                                     resume=args.resume,
                                     to_gpu=False if args.baseline_only else True,
                                     verbose=True)

    duration = time.time() - start_time

    print_msg = (f"avg_cv_train_loss: {avg_train_losses:.5f}",
                 f"avg_cv_valid_loss: {avg_valid_losses:.5f}",
                 # f"avg_cv_untrained_loss: {avg_untrained_losses:.5f}",
                 f"max_final_epoch: {max_final_epoch}",
                 f"num_samples: {len(cur_train_data)}",
                 f"time_this_iter_s: {duration}")

    print(print_msg)

    if not args.train_only:
        results = {
            "avg_cv_train_loss": avg_train_losses,
            "avg_cv_valid_loss": avg_valid_losses,
            # "avg_cv_untrained_loss": avg_untrained_losses,
            "max_final_epoch": max_final_epoch,
            "num_samples": len(cur_train_data),
            "time_this_iter_s": duration
        }
        # Save CV results as CSV
        print("Saving results to:", result_dir + '/CV_results.csv')
        with open(result_dir + '/CV_results.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in results.items():
                writer.writerow([key, value])

    # Save config to CSV
    print("Saving config csv to:", result_dir + '/config.csv')
    with open(result_dir + '/config.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in config.items():
            writer.writerow([key, value])

    print("Saving config json to:", result_dir + '/best_config.json')
    with open(result_dir + '/best_config.json', 'w') as fp:
        json.dump(config, fp)

    # Save the whole model
    print("Saving the final model to:", result_dir + "/final_model.pt")
    torch.save(final_model, result_dir + "/final_model.pt")


def hypopt(num_samples: int = 10, gpus_per_trial: float = 1.0, cpus_per_trial: float = 6.0):
    if args.machine == "cluster":
        local_dir = "/.mounts/labs/steinlab/scratch/ftaj/"
    elif args.machine == "mist":
        local_dir = "/scratch/l/lstein/ftaj/"
    else:
        local_dir = "//"

    if args.address == "":
        ray.init(num_cpus=int(args.init_cpus), num_gpus=int(args.init_gpus))
    else:
        # When connecting to an existing cluster, _lru_evict, num_cpus and num_gpus must not be provided.
        ray.init(address=args.address)

    # Full DRP
    config = {
        "drp_first_layer_size": tune.randint(2 ** 6, 2 ** 10),
        "drp_last_layer_size": tune.randint(2 ** 3, 2 ** 5),
        "drp_num_layers": tune.randint(2, 5),
        # "lr": tune.loguniform(1e-7, 1e-4),
        "lr": tune.choice([1e-5]),
        "batch_size": tune.choice([4, 8, 16, 32, 64, 128, 256])
    }

    if "gnndrug" in args.data_types:
        config = {**config, **gnn_param_ranges}
    if "cnv" in args.data_types:
        config['batch_size'] = tune.choice([12, 16, 20])
    if "exp" in args.data_types:
        config['batch_size'] = tune.choice([12, 16, 20, 24, 32, 64, 128, 256])

    if bool(int(args.full)) is True:
        if args.encoder_freeze is False:
            exit(
                "Full training instantiates encoders from random parameters; freezing their weights doesn't make sense")

        # Merge parameter ranges with config depending on input args
        if "drug" in args.data_types:
            config = {**config, **morgan_param_ranges}
        if "mut" in args.data_types:
            config = {**config, **mut_param_ranges}
        if "cnv" in args.data_types:
            config = {**config, **cnv_param_ranges}
        if "exp" in args.data_types:
            config = {**config, **exp_param_ranges}
        if "prot" in args.data_types:
            config = {**config, **prot_param_ranges}
        if "mirna" in args.data_types:
            config = {**config, **mirna_param_ranges}
        if "hist" in args.data_types:
            config = {**config, **hist_param_ranges}
        if "rppa" in args.data_types:
            config = {**config, **rppa_param_ranges}
        if "metab" in args.data_types:
            config = {**config, **metab_param_ranges}

    # Check if a global code layer size should be used
    if args.merge_method == 'sum':
        # remove all data type specific code layer sizes from the config
        updated_config = {key: config[key] for key in config.keys() - set(code_layer_size_names)}
        config = updated_config
        config = {**config, **{'global_code_size': int(args.global_code_size)}}

    elif args.merge_method == 'lmf':
        # Add LMF related configs
        config = {**config, **lmf_param_ranges}

    # Must NOT use an early-stopping TrialScheduler when performing cross-validation
    # if int(args.n_folds) == 1:
    # print("Will NOT perform cross-validation")
    scheduler = ASHAScheduler(
        grace_period=3,
        reduction_factor=3,
        brackets=3)
    search_algo = HyperOptSearch(random_state_seed=42)

    max_conc = int(int(args.init_gpus) / gpus_per_trial)

    search_algo = ConcurrencyLimiter(search_algo, max_concurrent=max_conc)

    print("Max concurrent:", str(max_conc))
    print("GPUs per trial:", str(gpus_per_trial),
          "\nCPUs per trial:", str(cpus_per_trial))
    # else:
    #     # print("Will perform cross-validation with Ray's Repeater Class")
    #     print("Will perform manual cross-validation")
    #     scheduler = None
    #     search_algo = Repeater(HyperOptSearch(points_to_evaluate=current_best_parameters),
    #                            repeat=int(args.n_folds), set_index=True)

    if bool(int(args.full)) is True:
        tag = "FullModel"
        cur_trainable = tune.with_parameters(FullModelTrainable,
                                             train_file=args.train_file,
                                             data_types='_'.join(args.data_types),
                                             bottleneck=bool(int(args.bottleneck)),
                                             n_folds=int(args.n_folds),
                                             max_epochs=int(args.max_num_epochs),
                                             encoder_train=args.encoder_freeze,
                                             cv_subset_type=args.cv_subset_type,
                                             stratify=bool(int(args.stratify)),
                                             random_morgan=bool(int(args.random_morgan)),
                                             merge_method=args.merge_method,
                                             loss_type=args.loss_type,
                                             one_hot_drugs=bool(int(args.one_hot_drugs)),
                                             transform=args.transform,
                                             min_dr_target=args.min_dr_target,
                                             max_dr_target=args.max_dr_target,
                                             name_tag=args.name_tag
                                             )
        # cur_trainable = tune.with_parameters(FullModelTrainable,
        #                                      train_file="CTRP_AAC_MORGAN_512.hdf",
        #                                      data_types='_'.join(['drug', 'exp']),
        #                                      bottleneck=bool(int('0')),
        #                                      n_folds=int('10'),
        #                                      max_epochs=int('50'),
        #                                      encoder_train=bool(int('1')),
        #                                      cv_subset_type='cell_line',
        #                                      stratify=bool(int('1')))
    else:
        tag = "ResponseOnly"
        cur_trainable = tune.with_parameters(DRPTrainable,
                                             train_file=args.train_file,
                                             data_types='_'.join(args.data_types),
                                             bottleneck=bool(int(args.bottleneck)),
                                             pretrain=bool(int(args.pretrain)),
                                             n_folds=int(args.n_folds),
                                             max_epochs=int(args.max_num_epochs),
                                             encoder_train=args.encoder_freeze,
                                             cv_subset_type=args.cv_subset_type,
                                             stratify=bool(int(args.stratify)),
                                             random_morgan=bool(int(args.random_morgan)),
                                             merge_method=args.merge_method,
                                             loss_type=args.loss_type,
                                             one_hot_drugs=bool(int(args.one_hot_drugs)),
                                             transform=args.transform,
                                             min_dr_target=args.min_dr_target,
                                             max_dr_target=args.max_dr_target,
                                             gnn_drug=("gnndrug" in args.data_types),
                                             omic_standardize=args.omic_standardize,
                                             name_tag=args.name_tag
                                             )

    checkpoint_dir = "HyperOpt_DRP_" + tag + '_' + "_".join(args.data_types) + '_' + args.name_tag
    print("Checkpoint directory:", checkpoint_dir)

    # reporter = CLIReporter(
    #     metric_columns=["avg_cv_train_loss", "avg_cv_valid_loss", "training_iteration",
    #                     "num_samples", "max_final_epoch", "time_this_iter_s"])
    reporter = WriterCLIReporter(
        checkpoint_dir=checkpoint_dir,
        metric_columns=["avg_cv_train_loss", "avg_cv_valid_loss", "training_iteration",
                        "num_samples", "max_final_epoch", "time_this_iter_s"])

    # if args.resume in ['0', '1']:
    #     resume = bool(int(args.resume))
    # elif args.resume == "ERRORED_ONLY":
    #     resume = "ERRORED_ONLY"
    # else:
    #     resume = False
    #     Warning("Invalid --resume arg, defaulting to False")

    result = tune.run(
        # partial(train_auto_encoder, checkpoint_dir="/.mounts/labs/steinlab/scratch/ftaj/", data_dir=data_dir),
        cur_trainable,
        name=checkpoint_dir,
        verbose=1,
        resume=args.resume if not args.errored_only else "ERRORED_ONLY",
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},  # "memory": 8 * 10 ** 9},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=search_algo,
        progress_reporter=reporter,
        metric='avg_cv_valid_loss',
        mode='min',
        keep_checkpoints_num=1,
        stop={"training_iteration": 1},
        checkpoint_at_end=True,
        # checkpoint_score_attr="min-loss",
        reuse_actors=True, local_dir=local_dir)

    # result = tune.run(
    #     # partial(train_auto_encoder, checkpoint_dir="/.mounts/labs/steinlab/scratch/ftaj/", data_dir=data_dir),
    #     cur_trainable,
    #     name=checkpoint_dir,
    #     verbose=1,
    #     resume="ERRORED_ONLY",
    #     resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},  # "memory": 8 * 10 ** 9},
    #     config=config,
    #     num_samples=num_samples,
    #     scheduler=scheduler,
    #     search_alg=search_algo,
    #     progress_reporter=reporter,
    #     metric='avg_cv_valid_loss',
    #     mode='min',
    #     keep_checkpoints_num=1,
    #     stop={"training_iteration": 1},
    #     checkpoint_at_end=True,
    #     # checkpoint_score_attr="min-loss",
    #     reuse_actors=True, local_dir=local_dir)

    best_trial = result.get_best_trial(metric="avg_cv_valid_loss", mode="min", scope="all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["avg_cv_valid_loss"]))
    print("Best trial number of epochs:", best_trial.last_result["max_final_epoch"])

    # Save best config
    best_config = best_trial.config

    print("Saving best config as JSON to", local_dir + checkpoint_dir + '/best_config.json')
    with open(local_dir + checkpoint_dir + '/best_config.json', 'w') as fp:
        json.dump(best_config, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This programs trains the DRP module by using the encoder modules of other data types.")
    parser.add_argument('--machine', help='Whether code is run on cluster or else', default="cluster")

    # Tuning parameters ###
    parser.add_argument("--address", help="ray head node address", default="")
    parser.add_argument('--optimize', help='Whether to just train the model (0) or optimize via Raytune (1)')
    parser.add_argument('--max_num_epochs', help='Number of epochs to run', required=False)
    # parser.add_argument('--max_seconds', help='Number of seconds to run')
    parser.add_argument('--init_cpus', help='Number of Total CPUs to use for ray.init', required=False)
    parser.add_argument('--init_gpus', help='Number of Total GPUs to use for ray.init', required=False)
    parser.add_argument('--gpus_per_trial', help="Can be a float between 0 and 1, or higher", default="1.0")
    parser.add_argument('--n_folds', help="Number of folds to use in cross validation", default="10")
    parser.add_argument('--cv_subset_type', help="Whether to subset by cell line, drug or both", default="cell_line")
    parser.add_argument('--stratify', help="Whether to stratify by cancer type during cross-validation", default="1")
    parser.add_argument('--num_samples',
                        help="Number of samples drawn from parameter combinations", required=False)
    # parser.add_argument('--fp_width', help="Width of the drug fingerprints used")
    parser.add_argument('--full', help="whether to optimize all modules (full) or just the DRP module itself",
                        default='0')

    parser.add_argument('--data_types', nargs="+", help='List of data types to be used in training this DRP model')
    parser.add_argument('--bottleneck',
                        help='Whether to restrict the training data to complete samples that have all 4 omic data types',
                        default="0")

    parser.add_argument('--pretrain', help='0/1 whether to use TCGA-pretrained encoders (for EXP and CNV only)',
                        default="0")

    parser.add_argument('--random_morgan', help='Create and use random morgan fingerprints for DRCurve data',
                        default='0')
    parser.add_argument('--merge_method', help='How encoder outputs should be merged', default='concat')
    parser.add_argument('--global_code_size',
                        help='If code layers are to be summed, what global code layer should be used?',
                        default='512')
    parser.add_argument('--loss_type', help='Loss function used, choices are: mae, mse, rmse, rmsle', default='mae')
    parser.add_argument('--one_hot_drugs', help='Use one-hot encoding for drugs instead of fingerprints', default='0')
    parser.add_argument('--transform', help='Whether to transform DR data; available methods are: log, sqrt',
                        default=None)
    parser.add_argument('--min_dr_target', help='Maximum dose-response AAC value', type=float, default=None)
    parser.add_argument('--max_dr_target', help='Minimum dose-response AAC value', type=float, default=None)
    parser.add_argument('--omic_standardize', help='Whether to standardize omics data', action='store_true')
    parser.add_argument('--errored_only', help='Whether to rerun errored only RayTune Trials', action='store_true')

    # Training parameters ###
    parser.add_argument('--train_file', help='Name of file used for training model, e.g. CTRP AAC data.')
    parser.add_argument('--train', help="Whether to train the model, or optimize hyperparameters",
                        action='store_true')
    parser.add_argument('--num_epochs', help='Number of epochs to run', required=False)
    parser.add_argument('--batch_size', help='Size of each training batch', required=False)
    parser.add_argument('--resume',
                        help='Whether to continue training from saved model at epoch, or start from scratch',
                        action='store_true')
    parser.add_argument('--test', help='Name of the testing data file if testing is requested')
    parser.add_argument('--interpret', help='Whether a fully trained model should be interpreted using SHAP',
                        default="0")
    parser.add_argument('--encoder_freeze', help="Whether to freeze or unfreeze encoder modules",
                        action='store_false')
    parser.add_argument('--name_tag',
                        help='A string that will be added to the model checkpoint file name generated by this program',
                        required=True)
    parser.add_argument('--dir_tag', help='The name of the subdirectory where checkpoints will be saved',
                        required=False)
    # parser.add_argument('--cross_validate',
    #                     help="Whether to cross-validate with the given config before training the model",
    #                     default='1')
    parser.add_argument("--train_only",
                        help="Whether to only train the model using the given config, or cross-validate too",
                        action='store_true')
    parser.add_argument("--final_train", help="Whether to train the model on all data at the end of cross-validation",
                        action='store_true')
    parser.add_argument("--redo_validation",
                        help="Whether to generate validation results using each CV model checkpoint",
                        action='store_true')
    parser.add_argument("--baseline_only",
                        help="Only run CV using ElasticNet (only available for drug + exp)",
                        action='store_true')

    parser.add_argument('--benchmark',
                        help="Whether the code should be run in benchmark mode for torch.utils.bottleneck")
    parser.add_argument('--force', help="Whether to force a re-run of training if a checkpoint already exists")

    parser.add_argument('--drp_first_layer_size')
    parser.add_argument('--drp_last_layer_size')
    parser.add_argument('--drp_num_layers')
    parser.add_argument('--lr')
    parser.add_argument('--batchnorm_list')
    parser.add_argument('--act_fun_list')
    parser.add_argument('--dropout_list')

    # parser.add_argument("--drug_file_name", help="Address of the drug SMILES + DR targets used for model evaluation")

    args = parser.parse_args()

    # if args.train_file is not None:
    #     assert os.path.isfile(
    #         "Data/DRP_Training_Data/" + args.train_file), "Given train file could not be found in main directory"

    # Create a checkpoint directory based on given name_tag if it doesn't exist
    # if not os.path.exists(args.name_tag):
    #     print("Creating checkpoint directory:", args.name_tag)
    #     os.makedirs(args.name_tag)

    # Path("/" + args.name_tag).mkdir(exist_ok=True)

    # if bool(int(args.interpret)) is True:
    #     assert bool(int(args.resume)) is True, "Must have a fully trained model for interpretation, with resume mode on"

    # Turning off benchmarking makes highly dynamic models faster
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    if args.train:
        PATH = "/scratch/l/lstein/ftaj/"
        if bool(int(args.full)) is True:
            tag = "FullModel"
        else:
            tag = "ResponseOnly"

        # Only use configs attained from hyper-param optimzation on the CTRPv2 dataset
        name_tag = args.name_tag
        if "GDSC1" in name_tag or "GDSC2" in name_tag:
            import re

            print("Changing config source from GDSC1/2 to CTRP")
            name_tag = re.sub("GDSC[1,2]", "CTRP", name_tag)
        # Ignore configs from cell line or drug only splitting, only use "BOTH"
        if "Split_CELL_LINE" in name_tag or "Split_DRUG" in name_tag:
            import re

            print("Changing split type from cell_line/drug to both")
            name_tag = re.sub(r'Split_CELL_LINE', "Split_BOTH", name_tag)
            name_tag = re.sub(r'Split_DRUG', "Split_BOTH", name_tag)

        file_address = PATH + "HyperOpt_DRP_" + tag + '_' + "_".join(
            args.data_types) + '_' + name_tag + '/best_config.json'
        print("Best config file should be at:", file_address)

        try:
            with open(file_address, 'r') as fp:
                best_config = json.load(fp)
            print("Found best config:", best_config)
            # analysis = Analysis(experiment_dir=PATH + "HyperOpt_DRP_" + tag + '_' + "_".join(args.data_types) + '_' + args.name_tag + "/",
            #                     default_metric="avg_cv_valid_loss", default_mode="min")
            # best_config = analysis.get_best_config()
        except:
            exit("Could not get best config from ray tune trial logdir")

        train_full_model(args, best_config)
        exit()

    if bool(int(args.optimize)) is True:
        hypopt(num_samples=int(args.num_samples),
               gpus_per_trial=float(args.gpus_per_trial),
               cpus_per_trial=np.floor(int(int(args.init_cpus) / int(args.init_gpus)) * float(args.gpus_per_trial))
               )

        exit()
