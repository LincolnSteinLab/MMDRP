import argparse
import csv
import json
import os
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
import torch.multiprocessing
from ray import tune
from torch.backends import cudnn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader

from CustomFunctions import MyLoggerCreator
from TrainFunctions import drp_train, cross_validate, gnn_drp_train
from TuneTrainables import FullModelTrainable, DRPTrainable

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


def train_viz(args, config):
    """
    Takes the data combo and associated model from the main_worker function and trains the model for specified epochs.
    :param args:
    :return:
    """
    max_epochs = int(args.max_num_epochs)

    if args.machine == "mist":
        data_dir = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/"
        local_dir = "/scratch/l/lstein/ftaj/"
        tensorboard_logdir = "/scratch/l/lstein/ftaj/TensorBoard/"
        cur_device = "cuda"
        to_gpu = True
    else:
        data_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
        local_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/"
        tensorboard_logdir = local_dir + "TensorBoard/"
        cur_device = "cpu"
        to_gpu = False

    if bool(int(args.full)) is True:
        tag = "FullModel"
    else:
        tag = "ResponseOnly"

    # Create the results directory and file paths
    experiment_name = "HyperOpt_DRP_" + tag + '_' + "_".join(args.data_types) + '_' + args.name_tag
    # experiment_name = "HyperOpt_DRP_ResponseOnly_gnndrug_exp_HyperOpt_DRP_CTRP_FullModel_EncoderTrain_Split_DRUG_WithBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GNNDrugs_gnndrug_exp_test"

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
                                             max_dr_target=args.max_dr_target
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
                                             transform=args.transform,
                                             min_dr_target=args.min_dr_target,
                                             max_dr_target=args.max_dr_target,
                                             gnn_drug=("gnndrug" in args.data_types),
                                             omic_standardize=args.omic_standardize,
                                             )

        cur_trainable = cur_trainable(config=config, logger_creator=MyLoggerCreator)

    cur_model, criterion = cur_trainable.cur_model, cur_trainable.criterion
    cur_train_data, cur_cv_folds = cur_trainable.cur_train_data, cur_trainable.cur_cv_folds

    print("Total samples for training:", str(len(cur_train_data)))

    if args.train_only:
        # cur_cv_folds = [cur_cv_folds[0]]
        print("Skipping cross-validation...")

    # Choose appropriate training function and data loaders depending on GNN use
    cur_train_function = gnn_drp_train if 'gnndrug' in args.data_types else drp_train
    cur_data_loader = DataLoader if 'gnndrug' in args.data_types else data.DataLoader

    train_loader = cur_data_loader(cur_train_data, batch_size=4,
                                   pin_memory=False, drop_last=True)

    # === Model visualization
    # default `log_dir` is "runs" - we'll be more specific here
    sw = SummaryWriter(tensorboard_logdir + "/runs")
    # drug_feats, omic_feats, targets, loss_weights = next(iter(train_loader))
    # net_input = [drug_feats, omic_feats]
    # sw.add_graph(cur_model, *net_input)
    # sw.close()
    # exit()

    cur_dict = {k: config[k] for k in ('drp_first_layer_size', 'drp_last_layer_size', 'gnn_out_channels')}
    epoch_save_folder = "/scratch/l/lstein/ftaj/EpochResults/" + str(cur_dict)
    Path(epoch_save_folder).mkdir(parents=True, exist_ok=True)

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
                                     patience=5,
                                     train_only=args.train_only,
                                     summary_writer=sw,
                                     final_full_train=True,
                                     learning_rate=0.0001,
                                     delta=0.001,
                                     NUM_WORKERS=0,
                                     theoretical_loss=True,
                                     verbose=True,
                                     save_epoch_results=True,
                                     epoch_save_folder=epoch_save_folder)
    # Close SummaryWriter
    sw.close()
    duration = time.time() - start_time

    print_msg = (f"avg_cv_train_loss: {avg_train_losses:.5f}",
                 f"avg_cv_valid_loss: {avg_valid_losses:.5f}",
                 f"avg_cv_untrained_loss: {avg_untrained_losses:.5f}",
                 f"max_final_epoch: {max_final_epoch}",
                 f"num_samples: {len(cur_train_data)}",
                 f"time_this_iter_s: {duration}")

    print(print_msg)

    if not args.train_only:
        results = {
            "avg_cv_train_loss": avg_train_losses,
            "avg_cv_valid_loss": avg_valid_losses,
            "avg_cv_untrained_loss": avg_untrained_losses,
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
    print("Saving config to:", result_dir + '/config.csv')
    with open(result_dir + '/config.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in config.items():
            writer.writerow([key, value])

    # Save the whole model
    print("Saving the final model to:", result_dir + "/final_model.pt")
    torch.save(cur_model, result_dir + "/final_model.pt")


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
    parser.add_argument('--global_code_size', help='If code layers are to be summed, what global code layer should be used?',
                        default='512')
    parser.add_argument('--loss_type', help='Loss function used, choices are: mae, mse, rmse, rmsle', default='mae')
    parser.add_argument('--one_hot_drugs', help='Use one-hot encoding for drugs instead of fingerprints', default='0')
    parser.add_argument('--transform', help='Whether to transform DR data; available methods are: log, sqrt', default=None)
    parser.add_argument('--min_dr_target', help='Maximum dose-response AAC value', type=float, default=None)
    parser.add_argument('--max_dr_target', help='Minimum dose-response AAC value', type=float, default=None)
    parser.add_argument('--omic_standardize', help='Whether to standardize _all_ data', action='store_true')

    # Training parameters ###
    parser.add_argument('--train_file', help='Name of file used for training model, e.g. CTRP AAC data.')
    parser.add_argument('--train', help="Whether to train the model, or optimize hyperparameters",
                        action='store_true')
    parser.add_argument('--num_epochs', help='Number of epochs to run', required=False)
    parser.add_argument('--batch_size', help='Size of each training batch', required=False)
    parser.add_argument('--resume',
                        help='Whether to continue training from saved model at epoch, or start training from scratch. Required for model interpretation.',
                        default="0")
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
    parser.add_argument("--train_only", help="Whether to only train the model using the given config, or cross-validate too",
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

    args = parser.parse_args()

    cudnn.benchmark = False
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

        train_viz(args, best_config)
        exit()
