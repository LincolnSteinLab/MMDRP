# This script contains an encoder of expression data either for an auto-encoder or to predict
# secondary labels such as tumor types
import os
import argparse
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from DRPPreparation import create_cv_folds
from DataImportModules import OmicData
from Models import DNNAutoEncoder
from TrainFunctions import omic_train, cross_validate
from TuneTrainables import OmicTrainable, file_name_dict, get_layer_configs

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


def train_auto_encoder(args, config, local_dir, data_dir):
    # batch_size = int(args.batch_size)
    # resume = bool(int(args.resume))
    num_epochs = int(args.num_epochs)
    checkpoint_dir = local_dir + '/' + args.name_tag

    # Create a checkpoint directory based on given name_tag if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        print("Creating checkpoint directory:", args.name_tag)
        os.makedirs(checkpoint_dir)

    training_set = OmicData(path=data_dir,
                            omic_file_name=file_name_dict[pretrain + args.data_type + "_file_name"])

    # Create model for validation/max_epoch determination
    all_act_funcs, all_batch_norm, all_dropout = get_layer_configs(config, model_type="dnn")
    cur_model = DNNAutoEncoder(input_dim=training_set.width(), first_layer_size=config["first_layer_size"],
                               code_layer_size=config["code_layer_size"], num_layers=config["num_layers"],
                               batchnorm_list=all_batch_norm, act_fun_list=all_act_funcs,
                               dropout_list=all_dropout, name=args.data_type)
    cur_model = cur_model.float()

    final_filename = checkpoint_dir + '/' + args.data_type.upper() + '_' + pretrain + untrained + "Omic_AutoEncoder_Checkpoint.pt"
    if bool(int(args.untrained_model)) is True:
        print("Saving untrained model (simply instantiated) based on given configuration")
        print("Final file name:", final_filename)
        torch.save(cur_model, final_filename)
        exit(0)

    cur_model = cur_model.cuda()

    criterion = nn.MSELoss().cuda()
    if pretrain != "":
        cur_sample_column_name = "tcga_sample_id"
    else:
        cur_sample_column_name = "stripped_cell_line_name"

    # Use 1/5 of the data for validation, ONLY to determine the number of epochs to train before over-fitting
    cv_folds = create_cv_folds(train_data=training_set, train_attribute_name="data_info",
                               sample_column_name=cur_sample_column_name, n_folds=5,
                               class_data_index=None, subset_type="cell_line",
                               class_column_name="primary_disease", seed=42, verbose=False)
    train_valid_fold = [cv_folds[0]]
    # Determine the number of epochs required for training before validation loss doesn't improve
    print("Using validation set to determine performance plateau\n" +
          "Batch Size:", config['batch_size'])
    train_loss, valid_loss, final_epoch = cross_validate(train_data=training_set, train_function=omic_train,
                                                         cv_folds=train_valid_fold, batch_size=config['batch_size'],
                                                         cur_model=cur_model, criterion=criterion,
                                                         patience=5,
                                                         delta=0.001,
                                                         max_epochs=num_epochs,
                                                         learning_rate=config['lr'],
                                                         verbose=True)

    print("Starting training based on number of epochs before validation loss worsens")
    cur_model = DNNAutoEncoder(input_dim=training_set.width(), first_layer_size=config["first_layer_size"],
                               code_layer_size=config["code_layer_size"], num_layers=config["num_layers"],
                               batchnorm_list=all_batch_norm, act_fun_list=all_act_funcs,
                               dropout_list=all_dropout, name=args.data_type)
    cur_model = cur_model.float()
    cur_model = cur_model.cuda()
    optimizer = optim.Adam(cur_model.parameters(), lr=config["lr"])

    train_loader = data.DataLoader(training_set, batch_size=config["batch_size"], num_workers=0, shuffle=True,
                                   pin_memory=True)

    avg_train_losses = []
    start = time.time()
    for epoch in range(final_epoch):
        train_losses = omic_train(cur_model, criterion, optimizer, epoch=epoch, train_loader=train_loader, verbose=True)

        avg_train_losses.append(train_losses.avg)

        duration = time.time() - start
        print_msg = (f'[{epoch:>{num_epochs}}/{final_epoch:>{num_epochs}}] ' +
                     f'train_loss: {train_losses.avg:.5f} ' +
                     f'epoch_time: {duration:.4f}')

        print(print_msg)
        start = time.time()

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

    if args.data_type in ["cnv", "exp"]:
        print("Data type is:", args.data_type)
        config = {
            "first_layer_size": tune.randint(2 ** 9, 8192),
            "code_layer_size": tune.randint(2 ** 8, 4096),
            "num_layers": tune.randint(2, 5),
            'n_folds': 10,
            'max_epochs': int(args.max_num_epochs),
            "activation_1": tune.choice(['none', 'relu', 'lrelu', 'prelu']),
            "activation_2": tune.choice(['none', 'relu', 'lrelu', 'prelu']),
            "activation_3": tune.choice(['none', 'relu', 'lrelu', 'prelu']),
            "activation_4": tune.choice(['none', 'relu', 'lrelu', 'prelu']),
            "activation_5": tune.choice(['none', 'relu', 'lrelu', 'prelu']),
            "batch_norm_1": tune.choice([True, False]),
            "batch_norm_2": tune.choice([True, False]),
            "batch_norm_3": tune.choice([True, False]),
            "batch_norm_4": tune.choice([True, False]),
            "batch_norm_5": tune.choice([True, False]),
            "dropout_1": tune.uniform(0, 0.25),
            "dropout_2": tune.uniform(0, 0.25),
            "dropout_3": tune.uniform(0, 0.25),
            "dropout_4": tune.uniform(0, 0.25),
            "dropout_5": tune.uniform(0, 0.25),
            "lr": tune.loguniform(1e-4, 1e-3),
            "batch_size": tune.randint(4, 32)
        }
        if args.data_type == "cnv":
            current_best_params = [{
                'first_layer_size': 529,
                'code_layer_size': 831,
                'num_layers': 2,
                'n_folds': 10,
                'max_epochs': 20,
                "activation_1": 'none',
                "activation_2": 'relu',
                "activation_3": 'none',
                "activation_4": 'none',
                "activation_5": 'lrelu',
                "batch_norm_1": True,
                "batch_norm_2": False,
                "batch_norm_3": False,
                "batch_norm_4": False,
                "batch_norm_5": False,
                "dropout_1": 0.08025254867287786,
                "dropout_2": 0.1325349864894265,
                "dropout_3": 0.08693645180241394,
                "dropout_4": 0.15921556875763504,
                "dropout_5": 0.05414028134057667,
                "lr": 0.0008629716445108056,
                "batch_size": 24
            }]
        else:
            # Exp data type
            current_best_params = [{
                'first_layer_size': 4096,
                'code_layer_size': 1024,
                'num_layers': 2,
                'n_folds': 10,
                'max_epochs': 20,
                "activation_1": 'relu',
                "activation_2": 'relu',
                "activation_3": 'relu',
                "activation_4": 'relu',
                "activation_5": 'relu',
                "batch_norm_1": True,
                "batch_norm_2": True,
                "batch_norm_3": True,
                "batch_norm_4": True,
                "batch_norm_5": True,
                "dropout_1": 0.01,
                "dropout_2": 0.01,
                "dropout_3": 0.01,
                "dropout_4": 0.01,
                "dropout_5": 0.01,
                "lr": 0.0001,
                "batch_size": 10
            }]

    elif args.data_type == "mut":
        print("Data type is:", args.data_type)
        config = {
            "first_layer_size": tune.randint(2 ** 7, 2 ** 10),
            "code_layer_size": tune.randint(2 ** 6, 2 ** 9),
            "num_layers": tune.randint(2, 5),
            "activation_1": tune.choice(['none', 'relu', 'lrelu', 'prelu', 'sigmoid', 'tanh']),
            "activation_2": tune.choice(['none', 'relu', 'lrelu', 'prelu', 'sigmoid', 'tanh']),
            "activation_3": tune.choice(['none', 'relu', 'lrelu', 'prelu', 'sigmoid', 'tanh']),
            "activation_4": tune.choice(['none', 'relu', 'lrelu', 'prelu', 'sigmoid', 'tanh']),
            "activation_5": tune.choice(['none', 'relu', 'lrelu', 'prelu', 'sigmoid', 'tanh']),
            "batch_norm_1": tune.choice([True, False]),
            "batch_norm_2": tune.choice([True, False]),
            "batch_norm_3": tune.choice([True, False]),
            "batch_norm_4": tune.choice([True, False]),
            "batch_norm_5": tune.choice([True, False]),
            "dropout_1": tune.uniform(0, 0.25),
            "dropout_2": tune.uniform(0, 0.25),
            "dropout_3": tune.uniform(0, 0.25),
            "dropout_4": tune.uniform(0, 0.25),
            "dropout_5": tune.uniform(0, 0.25),
            "lr": tune.loguniform(1e-4, 1e-3),
            "batch_size": tune.randint(4, 32)
        }
        current_best_params = [{
            'first_layer_size': 763,
            'code_layer_size': 500,
            'num_layers': 2,
            "activation_1": 'prelu',
            "activation_2": 'prelu',
            "activation_3": 'prelu',
            "activation_4": 'prelu',
            "activation_5": 'prelu',
            "batch_norm_1": False,
            "batch_norm_2": False,
            "batch_norm_3": False,
            "batch_norm_4": False,
            "batch_norm_5": False,
            "dropout_1": 0.0,
            "dropout_2": 0.0,
            "dropout_3": 0.0,
            "dropout_4": 0.0,
            "dropout_5": 0.0,
            "lr": 0.0002672,
            "batch_size": 32
        }]

    else:
        # prot
        print("Data type is:", args.data_type)
        config = {
            "first_layer_size": tune.randint(2 ** 9, 6000),
            "code_layer_size": tune.randint(2 ** 8, 5000),
            "num_layers": tune.randint(2, 5),
            "activation_1": tune.choice(['none', 'lrelu', 'prelu', 'sigmoid', 'tanh']),
            "activation_2": tune.choice(['none', 'lrelu', 'prelu', 'sigmoid', 'tanh']),
            "activation_3": tune.choice(['none', 'lrelu', 'prelu', 'sigmoid', 'tanh']),
            "activation_4": tune.choice(['none', 'lrelu', 'prelu', 'sigmoid', 'tanh']),
            "activation_5": tune.choice(['none', 'lrelu', 'prelu', 'sigmoid', 'tanh']),
            "batch_norm_1": tune.choice([True, False]),
            "batch_norm_2": tune.choice([True, False]),
            "batch_norm_3": tune.choice([True, False]),
            "batch_norm_4": tune.choice([True, False]),
            "batch_norm_5": tune.choice([True, False]),
            "dropout_1": tune.uniform(0, 0.25),
            "dropout_2": tune.uniform(0, 0.25),
            "dropout_3": tune.uniform(0, 0.25),
            "dropout_4": tune.uniform(0, 0.25),
            "dropout_5": tune.uniform(0, 0.25),
            "lr": tune.loguniform(1e-4, 1e-3),
            "batch_size": tune.randint(4, 32)
        }
        current_best_params = [{
            'first_layer_size': 2530,
            'code_layer_size': 1875,
            'num_layers': 2,
            "activation_1": 'prelu',
            "activation_2": 'prelu',
            "activation_3": 'prelu',
            "activation_4": 'prelu',
            "activation_5": 'prelu',
            "batch_norm_1": False,
            "batch_norm_2": False,
            "batch_norm_3": False,
            "batch_norm_4": False,
            "batch_norm_5": False,
            "dropout_1": 0.0,
            "dropout_2": 0.0,
            "dropout_3": 0.0,
            "dropout_4": 0.0,
            "dropout_5": 0.0,
            "lr": 0.00010036,
            "batch_size": 31
        }]

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
                                         n_folds=int(args.n_folds))

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
    hyperopt_search = HyperOptSearch(points_to_evaluate=current_best_params,
                                     random_state_seed=42)
    # hyperopt_search = HyperOptSearch()

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
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
        name="HyperOpt_CV_" + pretrain + args.data_type, verbose=1, resume=resume,
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

    # Training/model parameters if not running optimization
    parser.add_argument('--train', default='0')
    parser.add_argument('--num_epochs', default='100')
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
        analysis = Analysis(experiment_dir=PATH + "HyperOpt_CV_" + pretrain + args.data_type + "/",
                            default_metric="avg_cv_valid_loss", default_mode="min")
        best_config = analysis.get_best_config()

        train_auto_encoder(args, best_config, local_dir="/scratch/l/lstein/ftaj/",
                           data_dir='~/.conda/envs/drp1/Data/DRP_Training_Data/')
        exit()
    # You can change the number of GPUs per trial here:
    main(num_samples=int(args.num_samples),
         gpus_per_trial=float(args.gpus_per_trial),
         cpus_per_trial=int(int(args.init_cpus) / int(args.init_gpus)) * float(args.gpus_per_trial))
