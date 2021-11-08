# This script contains an encoder of expression data either for an auto-encoder or to predict
# secondary labels such as tumor types
import copy
import json
import os
import argparse
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from CustomFunctions import MyLoggerCreator, WriterCLIReporter
from loss_functions import RMSLELoss, RMSELoss
from DRPPreparation import create_cv_folds
from DataImportModules import OmicData
from Models import DNNAutoEncoder
from TrainFunctions import omic_train, cross_validate
from TuneTrainables import OmicTrainable, file_name_dict, get_layer_configs, MorganTrainable

import ray
from ray import tune
from ray.tune import CLIReporter, Analysis
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

print("Current working directory is:", os.getcwd())

use_cuda = torch.cuda.is_available()
torch.cuda.device_count()
device = torch.device("cuda:0" if use_cuda else "cpu")
# Turning off benchmarking makes highly dynamic models faster
cudnn.benchmark = True
cudnn.deterministic = True

mut_param_ranges = {
    "first_layer_size": tune.randint(2 ** 7, 2 ** 9),
    "code_layer_size": tune.randint(2 ** 6, 2 ** 8),
    "num_layers": tune.randint(2, 4)
}
cnv_param_ranges = {
    "first_layer_size": tune.randint(2 ** 9, 6000),
    "code_layer_size": tune.randint(2 ** 8, 4000),
    "num_layers": tune.randint(2, 4)
}
exp_param_ranges = {
    "first_layer_size": tune.randint(2 ** 9, 6000),
    "code_layer_size": tune.randint(2 ** 8, 4000),
    "num_layers": tune.randint(2, 4)
}
prot_param_ranges = {
    "first_layer_size": tune.randint(2 ** 9, 4000),
    "code_layer_size": tune.randint(2 ** 8, 2500),
    "num_layers": tune.randint(2, 4)
}
mirna_param_ranges = {
    "first_layer_size": tune.randint(2 ** 7, 2 ** 10),
    "code_layer_size": tune.randint(2 ** 6, 2 ** 9),
    "num_layers": tune.randint(2, 3)
}
hist_param_ranges = {
    "first_layer_size": tune.randint(2 ** 3, 2 ** 5),
    "code_layer_size": tune.randint(2 ** 6, 2 ** 9),
    "num_layers": tune.randint(2, 3)
}
rppa_param_ranges = {
    "first_layer_size": tune.randint(2 ** 5, 2 ** 8),
    "code_layer_size": tune.randint(2 ** 4, 2 ** 7),
    "num_layers": tune.randint(2, 3)
}
metab_param_ranges = {
    "first_layer_size": tune.randint(2 ** 5, 2 ** 8),
    "code_layer_size": tune.randint(2 ** 4, 2 ** 7),
    "num_layers": tune.randint(2, 3)
}


def train_auto_encoder(args, config, local_dir, data_dir):
    # batch_size = int(args.batch_size)
    # resume = bool(int(args.resume))
    checkpoint_dir = local_dir + '/' + args.name_tag

    # Create a checkpoint directory based on given name_tag if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        print("Creating checkpoint directory:", args.name_tag)
        os.makedirs(checkpoint_dir)

    # training_set = OmicData(path=data_dir,
    #                         omic_file_name=file_name_dict[pretrain + args.data_type + "_file_name"])

    # # Create model for validation/max_epoch determination
    # all_act_funcs, all_batch_norm, all_dropout = get_layer_configs(config, model_type="dnn")
    # cur_model = DNNAutoEncoder(input_dim=training_set.width(), first_layer_size=config["first_layer_size"],
    #                            code_layer_size=config["code_layer_size"], num_layers=config["num_layers"],
    #                            batchnorm_list=all_batch_norm, act_fun_list=all_act_funcs,
    #                            dropout_list=all_dropout, name=args.data_type)
    # cur_model = cur_model.float()

    cur_trainable = tune.with_parameters(OmicTrainable,
                                         max_epochs=int(args.max_num_epochs),
                                         pretrain=pretrain,
                                         data_type=args.data_type,
                                         n_folds=int(args.n_folds),
                                         loss_type=args.loss_type,
                                         omic_standardize=args.omic_standardize
                                         )

    if args.global_code_size is not None:
        global_code_tag = "GlobalCodeSize_" + args.global_code_size + "_"
        # Manually ensure that code layer size is the one we actually want
        config['code_layer_size'] = int(args.global_code_size)
    else:
        global_code_tag = ""

    cur_trainable = cur_trainable(config=config, logger_creator=MyLoggerCreator)
    cur_model = copy.deepcopy(cur_trainable.cur_model)
    cur_model = cur_model.float()
    cur_model = cur_model.cuda()

    final_filename = checkpoint_dir + '/' + args.data_type.upper() + '_' + pretrain + \
                     untrained + global_code_tag + "Omic_AutoEncoder_Checkpoint.pt"
    if bool(int(args.untrained_model)) is True:
        print("Saving untrained model (simply instantiated) based on given configuration")
        print("Final file name:", final_filename)
        torch.save(cur_model, final_filename)
        exit(0)

    if args.loss_type == "mae":
        criterion = nn.L1Loss().cuda()
    elif args.loss_type == "mse":
        criterion = nn.MSELoss().cuda()
    elif args.loss_type == "rmsle":
        criterion = RMSLELoss().cuda()
    elif args.loss_type == "rmse":
        criterion = RMSELoss().cuda()
    else:
        exit("Unknown loss function requested")

    if pretrain != "":
        cur_sample_column_name = "tcga_sample_id"
    else:
        cur_sample_column_name = "stripped_cell_line_name"

    # Use 1/5 of the data for validation, ONLY to determine the number of epochs to train before over-fitting
    cur_train_data, cur_cv_folds = cur_trainable.train_data, cur_trainable.cv_folds

    # cv_folds = create_cv_folds(train_data=training_set, train_attribute_name="data_info",
    #                            sample_column_name=cur_sample_column_name, n_folds=5,
    #                            class_data_index=None, subset_type="cell_line",
    #                            class_column_name="primary_disease", seed=42, verbose=False)
    train_valid_fold = [cur_cv_folds[0]]
    print("CV folds:", train_valid_fold)
    # Determine the number of epochs required for training before validation loss doesn't improve
    print("Using validation set to determine performance plateau\n" +
          "Batch Size:", config['batch_size'])
    cur_model, \
    train_loss, \
    valid_loss, \
    untrained_loss, \
    final_epoch = cross_validate(train_data=cur_train_data,
                                 train_function=omic_train,
                                 cv_folds=train_valid_fold,
                                 batch_size=config['batch_size'],
                                 cur_model=cur_model,
                                 criterion=criterion,
                                 patience=15,
                                 delta=0.01,
                                 max_epochs=int(args.max_num_epochs),
                                 learning_rate=config['lr'],
                                 theoretical_loss=False,
                                 final_full_train=True,
                                 omic_standardize=args.omic_standardize,
                                 verbose=True)

    # print("Starting training based on number of epochs before validation loss worsens")
    # cur_model = copy.deepcopy(cur_trainable.cur_model)
    # cur_model = cur_model.float()
    # cur_model = cur_model.cuda()

    # cur_model = DNNAutoEncoder(input_dim=training_set.width(), first_layer_size=config["first_layer_size"],
    #                            code_layer_size=config["code_layer_size"], num_layers=config["num_layers"],
    #                            batchnorm_list=all_batch_norm, act_fun_list=all_act_funcs,
    #                            dropout_list=all_dropout, name=args.data_type)

    # optimizer = optim.Adam(cur_model.parameters(), lr=config["lr"])
    # train_loader = data.DataLoader(cur_train_data, batch_size=config["batch_size"], num_workers=0, shuffle=True,
    #                                pin_memory=True)
    # avg_train_losses = []
    # start = time.time()
    # for epoch in range(final_epoch):
    #     train_losses = omic_train(cur_model, criterion, optimizer, epoch=epoch, train_loader=train_loader, verbose=True)
    #
    #     avg_train_losses.append(train_losses.avg)
    #
    #     duration = time.time() - start
    #     print_msg = (f'[{epoch:>{int(args.max_num_epochs)}}/{final_epoch:>{int(args.max_num_epochs)}}] ' +
    #                  f'train_loss: {train_losses.avg:.5f} ' +
    #                  f'epoch_time: {duration:.4f}')
    #
    #     print(print_msg)
    #     start = time.time()

    # print("Finished epoch in", str(epoch + 1), str(duration), "seconds")
    print("Final file name:", final_filename)
    # Save entire model, not just the state dict
    torch.save(cur_model, final_filename)
    print("Finished training!")


def main(num_samples=10, gpus_per_trial=1.0, cpus_per_trial=6.0):
    if args.machine == "cluster":
        local_dir = "/.mounts/labs/steinlab/scratch/ftaj/"
    elif args.machine == "mist":
        local_dir = "/scratch/l/lstein/ftaj/"
    else:
        local_dir = "//"

    # Ray address for the multi-node case
    if args.address == "":
        ray.init(num_cpus=int(args.init_cpus), num_gpus=int(args.init_gpus))
    else:
        # When connecting to an existing cluster, _lru_evict, num_cpus and num_gpus must not be provided.
        ray.init(address=args.address)

    config = {
        # "act_fun": tune.choice(['none', 'relu', 'lrelu', 'prelu']),
        # "batchnorm": tune.choice([True, False]),
        # "dropout": tune.uniform(0.0, 0.1),
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([4, 8, 16, 32])
    }

    # Merge parameter ranges with config depending on input args
    if "mut" == args.data_type:
        config = {**config, **mut_param_ranges}
    if "cnv" == args.data_type:
        config = {**config, **cnv_param_ranges}
    if "exp" == args.data_type:
        config = {**config, **exp_param_ranges}
    if "prot" == args.data_type:
        config = {**config, **prot_param_ranges}
    if "mirna" == args.data_type:
        config = {**config, **mirna_param_ranges}
    if "hist" == args.data_type:
        config = {**config, **hist_param_ranges}
    if "rppa" == args.data_type:
        config = {**config, **rppa_param_ranges}
    if "metab" == args.data_type:
        config = {**config, **metab_param_ranges}

    # Check if a global code layer size should be used
    if args.global_code_size is not None:
        print("Global code layer size is given, will remove code layer choice ranges from hyper-parameter list")
        # reset code_layer_size to given input
        config['code_layer_size'] = int(args.global_code_size)

    # scheduler = PopulationBasedTraining(
    #     time_attr='training_iteration',
    #     metric="loss",
    #     mode="min",
    #     perturbation_interval=5,
    #     hyperparam_mutations={
    #         # "first_layer_size": tune.sample_from(lambda _: 2 ** np.random.randint(9, 13)),
    #         # "code_layer_size": tune.sample_from(lambda _: 2 ** np.random.randint(8, 12)),
    #         # "num_layers": tune.choice([2, 3, 4, 5, 6]),
    #         "lr": tune.loguniform(1e-4, 1e-1),
    #         "batch_size": [2, 4, 8, 16, 32]
    #     },
    #     # max_t=max_num_epochs,
    #     # grace_period=1,
    #     # reduction_factor=2
    # )
    # scheduler = HyperBandForBOHB(
    #     time_attr="training_iteration", max_t=max_num_epochs, reduction_factor=2)
    # bohb_search = TuneBOHB(
    #     # space=config_space,  # If you want to set the space manually
    #     mode="min",
    #     max_concurrent=4)

    cur_trainable = tune.with_parameters(OmicTrainable,
                                         max_epochs=int(args.max_num_epochs),
                                         pretrain=pretrain,
                                         data_type=args.data_type,
                                         n_folds=int(args.n_folds),
                                         loss_type=args.loss_type,
                                         omic_standardize=args.omic_standardize
                                         )

    # # Get untrained loss
    # data_dir = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/"
    # training_set = OmicData(path=data_dir,
    #                         omic_file_name=file_name_dict[pretrain + args.data_type + "_file_name"])
    #
    # # Create model for validation/max_epoch determination
    # all_act_funcs, all_batch_norm, all_dropout = get_layer_configs(config, model_type="dnn")
    # cur_model = DNNAutoEncoder(input_dim=training_set.width(), first_layer_size=config["first_layer_size"],
    #                            code_layer_size=config["code_layer_size"], num_layers=config["num_layers"],
    #                            batchnorm_list=all_batch_norm, act_fun_list=all_act_funcs,
    #                            dropout_list=all_dropout, name=args.data_type)
    # cur_model = cur_model.float()
    #
    # final_filename = checkpoint_dir + '/' + args.data_type.upper() + '_' + pretrain + untrained + "Omic_AutoEncoder_Checkpoint.pt"
    # if bool(int(args.untrained_model)) is True:
    #     print("Saving untrained model (simply instantiated) based on given configuration")
    #     print("Final file name:", final_filename)
    #     torch.save(cur_model, final_filename)
    #     exit(0)
    #
    # cur_model = cur_model.cuda()
    #
    # criterion = nn.MSELoss().cuda()
    # # optimizer = optim.Adam(cur_model.parameters(), lr=0.001)
    # optimizer = optim.Adam(cur_model.parameters(), lr=config["lr"])

    scheduler = ASHAScheduler(
        grace_period=5,
        reduction_factor=3,
        brackets=3)
    hyperopt_search = HyperOptSearch(random_state_seed=42)
    # hyperopt_search = HyperOptSearch()

    reporter = WriterCLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        checkpoint_dir="HyperOpt_CV_" + pretrain + args.data_type,
        metric_columns=["avg_cv_train_loss", "avg_cv_valid_loss", "training_iteration", "max_final_epoch",
                        "time_this_iter_s"])

    if args.resume in ['0', '1']:
        resume = bool(int(args.resume))
    elif args.resume == "ERRORED_ONLY":
        resume = "ERRORED_ONLY"
    else:
        resume = False
        Warning("Invalid --resume arg, defaulting to False")

    result = tune.run(
        # partial(train_auto_encoder, checkpoint_dir="/.mounts/labs/steinlab/scratch/ftaj/", data_dir=data_dir),
        cur_trainable,
        name="HyperOpt_CV_" + pretrain + args.data_type,
        verbose=1, resume=resume,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},  # "memory": 8 * 10 ** 9},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=hyperopt_search,
        progress_reporter=reporter,
        # TODO for CV, only 1 step does all the work without checkpointing in between
        stop={"training_iteration": 1},
        # time_budget_s=int(args.max_seconds),
        metric="avg_cv_valid_loss",
        mode="min",
        keep_checkpoints_num=1,
        checkpoint_at_end=True,
        # checkpoint_score_attr="min-loss",
        reuse_actors=True, local_dir=local_dir)

    best_trial = result.get_best_trial(metric="avg_cv_valid_loss", mode="min", scope="all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final sum validation loss: {}".format(
        best_trial.last_result["avg_cv_valid_loss"]))
    print("Best trial number of epochs:", best_trial.last_result["training_iteration"])

    # Save best config
    best_config = best_trial.config

    print("Saving best config as JSON to", local_dir + "HyperOpt_CV_" + pretrain + args.data_type + '/best_config.json')
    with open(local_dir + "HyperOpt_CV_" + pretrain + args.data_type + '/best_config.json', 'w') as fp:
        json.dump(best_config, fp)

    # print("Best trial final validation accuracy: {}".format(
    #     best_trial.last_result["accuracy"]))

    # best_trained_model = ExpressionAutoEncoder(best_trial.config["l1"], best_trial.config["l2"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = NeuralNets.DataParallel(best_trained_model)
    # best_trained_model.to(device)
    #
    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(
    #     best_checkpoint_dir, "checkpoint"))
    # best_trained_model.load_state_dict(model_state)
    #
    # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This programs trains an auto-encoder module of specified data types")

    parser.add_argument('--machine', help='Whether code is run on cluster or else', default="cluster")
    # Tune parameters
    parser.add_argument("--address", help="ray head node address", default="")
    parser.add_argument('--max_num_epochs', help='Number of epochs to run', required=False)
    # parser.add_argument('--max_seconds', help='Number of seconds to run', default="3600000")
    parser.add_argument('--init_cpus', help='Number of Total CPUs to use for ray.init', required=False)
    parser.add_argument('--init_gpus', help='Number of Total GPUs to use for ray.init', required=False)
    parser.add_argument('--gpus_per_trial', help="Can be a float between 0 and 1, or higher", default="1.0")
    parser.add_argument('--num_samples',
                        help="Number of samples drawn from parameter combinations, i.e. PBT's population",
                        required=False)
    parser.add_argument('--resume',
                        help='Whether to continue training from saved model at epoch, or start training from scratch. Required for model interpretation.',
                        default="0")
    parser.add_argument('--data_type', help='Data type to be used in training this auto-encoder', required=True)
    parser.add_argument('--n_folds', help='Number of folds for cross-validation', default="5")
    parser.add_argument('--name_tag', help='A string that will be added to the CSV file name generated by this program',
                        required=False)

    parser.add_argument('--pretrain', default='0', help="Whether to use pre-training data for HyperOpt or Training")
    parser.add_argument('--untrained_model', default='0',
                        help="Only instantiates the model and saves it without training")

    parser.add_argument('--global_code_size',
                        help='If code layers are to be summed, what global code layer should be used?',
                        required=False)
    parser.add_argument('--loss_type', help='Loss function used, choices are: mae, mse, rmse, rmsle', default='mae')
    parser.add_argument('--omic_standardize', help='Whether to standardize omics data', action='store_true')

    # Training/model parameters if not running optimization
    parser.add_argument('--train', default='0')
    # parser.add_argument('--num_epochs', default='100')
    parser.add_argument('--first_layer_size', required=False)
    parser.add_argument('--code_layer_size', required=False)
    parser.add_argument('--num_layers', required=False)
    parser.add_argument('--batchnorm_list', required=False)
    parser.add_argument('--act_fun_list', required=False)
    parser.add_argument('--batch_size', required=False)
    parser.add_argument('--lr', required=False)

    # # TODO
    # parser.add_argument('--force', help="Whether to force a re-run of training if a checkpoint already exists")
    #
    args = parser.parse_args()

    if bool(int(args.pretrain)) is True and bool(int(args.untrained_model)) is True:
        print("Will create untrained model based on configuration from pretraining trials!")
    # Create a checkpoint directory based on given name_tag if it doesn't exist
    # if not os.path.exists(args.name_tag):
    #     print("Creating checkpoint directory:", args.name_tag)
    #     os.makedirs(args.name_tag)

    # Turning off benchmarking makes highly dynamic models faster
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    if bool(int(args.pretrain)) is True:
        pretrain = "pretrain_"
    else:
        pretrain = ""

    if bool(int(args.untrained_model)) is True:
        untrained = "untrained_"
    else:
        untrained = ""

    if bool(int(args.train)) is True:
        PATH = "/scratch/l/lstein/ftaj/"
        file_address = PATH + "HyperOpt_CV_" + pretrain + args.data_type + '/best_config.json'
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

        # analysis = Analysis(experiment_dir=PATH + "HyperOpt_CV_" + pretrain + args.data_type + "/",
        #                     default_metric="avg_cv_valid_loss", default_mode="min")
        # best_config = analysis.get_best_config()

        train_auto_encoder(args, best_config, local_dir="/scratch/l/lstein/ftaj/",
                           data_dir='~/.conda/envs/drp1/Data/DRP_Training_Data/')
        exit()
    # You can change the number of GPUs per trial here:
    main(num_samples=int(args.num_samples),
         gpus_per_trial=float(args.gpus_per_trial),
         cpus_per_trial=int(int(args.init_cpus) / int(args.init_gpus)) * float(args.gpus_per_trial))
