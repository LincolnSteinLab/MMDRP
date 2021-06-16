import argparse
import os

import ray
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from torch.utils import data

from DataImportModules import MorganData
from Models import DNNAutoEncoder
from TrainFunctions import morgan_train
from TuneTrainables import MorganTrainable

print("Current working directory is:", os.getcwd())

use_cuda = torch.cuda.is_available()
torch.cuda.device_count()
device = torch.device("cuda:0" if use_cuda else "cpu")
# Turning off benchmarking makes highly dynamic models faster
cudnn.benchmark = False
cudnn.deterministic = True


def train_auto_encoder(config, local_dir, data_dir):
    # TODO Implement CNN AutoEncoder Training
    # batch_size = int(args.batch_size)
    # resume = bool(int(args.resume))
    num_epochs = int(args.num_epochs)
    checkpoint_dir = local_dir + '/' + args.name_tag
    morgan_file_name = "ChEMBL_Morgan_" + config["width"] + ".pkl"

    # Create a checkpoint directory based on given name_tag if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        print("Creating checkpoint directory:", args.name_tag)
        os.makedirs(checkpoint_dir)

    training_set = MorganData(path=data_dir, morgan_file_name=morgan_file_name)
    train_loader = data.DataLoader(training_set, batch_size=config["batch_size"],
                                   num_workers=0, shuffle=True)

    assert training_set.width() == int(
        config["width"]), "The width of the given training set doesn't match the expected Morgan width."

    # TODO add other hyperparameters
    cur_model = DNNAutoEncoder(input_dim=int(config["width"]),
                               first_layer_size=config["first_layer_size"],
                               code_layer_size=config["code_layer_size"],
                               num_layers=config["num_layers"],
                               name="morgan")
    cur_model = cur_model.float()
    cur_model = cur_model.cuda()
    last_epoch = 0

    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(cur_model.parameters(), lr=config["lr"])

    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    for epoch in range(last_epoch, num_epochs):
        train_losses = morgan_train(cur_model, criterion, optimizer, epoch=epoch, train_loader=train_loader)

        avg_train_losses.append(train_losses.avg)

        print_msg = ('[{epoch:>{epoch_len}}/{200:>{epoch_len}}] ' +
                     'train_loss: {train_losses.avg:.5f} ' +
                     'epoch_time: {duration:.4f}')

        print(print_msg)
        # print("Finished epoch in", str(epoch + 1), str(duration), "seconds")
        print("Final file name:", checkpoint_dir + '/Morgan_' + args.width + "_AutoEncoder_Checkpoint.pt")
        # Save the entire model, not just the state dict
        torch.save(cur_model, checkpoint_dir + '/Morgan_' + args.width + "_AutoEncoder_Checkpoint.pt")
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': cur_model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': avg_train_losses
        # }, checkpoint_dir + '/Morgan_' + args.width + "_AutoEncoder_Checkpoint")
    print("Finished training!")


def main(num_samples=10, max_num_epochs=100, gpus_per_trial=1.0, cpus_per_trial=6.0):
    if args.machine == "cluster":
        local_dir = "/.mounts/labs/steinlab/scratch/ftaj/"
        # ray.init(_lru_evict=True, num_cpus=int(args.init_cpus), num_gpus=int(args.init_gpus),
        #          object_store_memory=1.5 * 10 ** 10)
    elif args.machine == "mist":
        local_dir = "/scratch/l/lstein/ftaj/"
        # ray.init(_lru_evict=True, num_cpus=int(args.init_cpus), num_gpus=int(args.init_gpus))
    else:
        local_dir = "//"

    # Ray address for the multi-node case
    if args.address == "":
        ray.init(_lru_evict=True, num_cpus=int(args.init_cpus), num_gpus=int(args.init_gpus))
    else:
        # When connecting to an existing cluster, _lru_evict, num_cpus and num_gpus must not be provided.
        ray.init(address=args.address)

    if args.model_type == "dnn":
        config = {
            "width": tune.choice(["512", "1024", "2048", "4096"]),
            "first_layer_size": tune.randint(2 ** 7, 2 ** 12),
            "code_layer_size": tune.randint(2 ** 6, 2 ** 10),
            "num_layers": tune.randint(2, 5),
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
            "batch_size": tune.randint(32, 256)
        }
        current_best_params = [{
            "width": "4096",
            "first_layer_size": 2061,
            "code_layer_size": 445,
            "num_layers": 2,
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
            "dropout_1": 0.0,
            "dropout_2": 0.0,
            "dropout_3": 0.0,
            "dropout_4": 0.0,
            "dropout_5": 0.0,
            "lr": 0.000121820,
            "batch_size": 256
        }]
    else:
        # model type CNN
        # Cannot use even-length strides!
        config = {
            "model_type": "cnn",
            "width": tune.choice(["512", "1024", "2048", "4096"]),
            "num_branch": tune.randint(2, 7),
            "stride": tune.choice([1, 3, 5]),
            "activation_1": tune.choice(['none', 'relu', 'lrelu', 'prelu']),
            "activation_2": tune.choice(['none', 'relu', 'lrelu', 'prelu']),
            "activation_3": tune.choice(['none', 'relu', 'lrelu', 'prelu']),
            "activation_4": tune.choice(['none', 'relu', 'lrelu', 'prelu']),
            "activation_5": tune.choice(['none', 'relu', 'lrelu', 'prelu']),
            "activation_6": tune.choice(['none', 'relu', 'lrelu', 'prelu']),
            "activation_7": tune.choice(['none', 'relu', 'lrelu', 'prelu']),
            "batch_norm_1": tune.choice([True, False]),
            "batch_norm_2": tune.choice([True, False]),
            "batch_norm_3": tune.choice([True, False]),
            "batch_norm_4": tune.choice([True, False]),
            "batch_norm_5": tune.choice([True, False]),
            "batch_norm_6": tune.choice([True, False]),
            "batch_norm_7": tune.choice([True, False]),
            "dropout_1": tune.uniform(0, 0.25),
            "dropout_2": tune.uniform(0, 0.25),
            "dropout_3": tune.uniform(0, 0.25),
            "dropout_4": tune.uniform(0, 0.25),
            "dropout_5": tune.uniform(0, 0.25),
            "dropout_6": tune.uniform(0, 0.25),
            "dropout_7": tune.uniform(0, 0.25),
            "lr": tune.loguniform(1e-4, 1e-3),
            # Note the smaller batch size for CNNs compared to DNNs
            "batch_size": tune.randint(32, 256)
        }

    # config_space = CS.ConfigurationSpace()
    # config_space.add_hyperparameter(
    #     CS.CategoricalHyperparameter("width", choices=["512", "1024", "2048", "4096"]))
    # config_space.add_hyperparameter(
    #     CS.UniformIntegerHyperparameter("first_layer_size", lower=128, upper=4096))
    # config_space.add_hyperparameter(
    #     CS.UniformIntegerHyperparameter("last_layer_size", lower=64, upper=1024))
    # config_space.add_hyperparameter(
    #     CS.UniformIntegerHyperparameter("num_layers", lower=2, upper=5))
    # config_space.add_hyperparameter(
    #     CS.UniformFloatHyperparameter("lr", lower=1e-4, upper=1e-3))
    # config_space.add_hyperparameter(
    #     CS.UniformIntegerHyperparameter("batch_size", lower=32, upper=256))

    # scheduler = HyperBandForBOHB(
    #     time_attr="training_iteration", max_t=max_num_epochs, reduction_factor=2)
    # bohb_search = TuneBOHB(
    #     space=config_space,  # If you want to set the space manually
    #     mode="min",
    #     max_concurrent=4)
    scheduler = ASHAScheduler(
        grace_period=3,
        reduction_factor=4,
        brackets=3)
    # hyperopt_search = HyperOptSearch(points_to_evaluate=[{'width': '4096', 'first_layer_size': 3850,
    #                                                       'code_layer_size': 318, 'num_layers': 2,
    #                                                       'lr': 0.00011123898831190815, 'batch_size': 778}])
    if args.model_type == "dnn":
        hyperopt_search = HyperOptSearch(points_to_evaluate=current_best_params)
    else:
        hyperopt_search = HyperOptSearch()

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["sum_valid_loss", "sum_train_loss", "training_iteration", "time_this_iter_s"])

    if args.resume in ['0', '1']:
        resume = bool(int(args.resume))
    elif args.resume == "ERRORED_ONLY":
        resume = "ERRORED_ONLY"
    else:
        resume = False
        Warning("Invalid --resume arg, defaulting to False")

    result = tune.run(
        # partial(train_auto_encoder, checkpoint_dir="/.mounts/labs/steinlab/scratch/ftaj/", data_dir=data_dir),
        MorganTrainable,
        name="HyperOpt_Test_Morgan_" + args.model_type, verbose=1, resume=resume,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},  # "memory": 8 * 10 ** 9},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        # max_failures=-1,
        search_alg=hyperopt_search,
        progress_reporter=reporter,
        stop={"training_iteration": max_num_epochs},
        # time_budget_s=int(args.max_seconds),
        metric="sum_valid_loss",
        mode="min",
        keep_checkpoints_num=1,
        checkpoint_at_end=True,
        # checkpoint_score_attr="min-loss",
        reuse_actors=True, local_dir=local_dir)

    best_trial = result.get_best_trial(metric="sum_valid_loss", mode="min", scope="all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final valid loss: {}".format(
        best_trial.last_result["sum_valid_loss"]))
    print("Best trial number of epochs:", best_trial.last_result["training_iteration"])


if __name__ == '__main__':
    # path = "/content/gdrive/My Drive/Python/RNAi/Train_Data/"
    # PATH = "/u/ftaj/anaconda3/envs/Python/Data/DRP_Training_Data/"
    # PATH = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
    parser = argparse.ArgumentParser(
        description="This programs trains/optimizes an auto-encoder module on drug (fingerprint) data")

    parser.add_argument('--machine', help='Whether code is run on cluster or else', default="cluster")

    # Tune parameters ====
    parser.add_argument("--address", help="ray head node address", default="")
    parser.add_argument('--max_num_epochs', help='Number of epochs to run', required=False)
    # parser.add_argument('--max_seconds', help='Number of seconds to run', default="3600000")
    parser.add_argument('--init_cpus', help='Number of Total CPUs to use for ray.init', required=False)
    parser.add_argument('--init_gpus', help='Number of Total GPUs to use for ray.init', required=False)
    parser.add_argument('--num_samples',
                        help="Number of samples drawn from parameter combinations, i.e. PBT's population",
                        required=False)
    parser.add_argument('--resume',
                        help='Whether to continue training from saved model at epoch, or start training from scratch. Required for model interpretation.',
                        default="0")
    parser.add_argument('--model_type', help='Model type (cnn or dnn) to be used in training this auto-encoder',
                        required=False, default="dnn")
    parser.add_argument('--gpus_per_trial', help="Can be a float between 0 and 1, or higher", default="1.0")

    # Training parameters ====
    parser.add_argument('--train', default='0')
    parser.add_argument('--num_epochs', default='100')
    parser.add_argument('--first_layer_size', required=False)
    parser.add_argument('--code_layer_size', required=False)
    parser.add_argument('--num_layers', required=False)
    parser.add_argument('--batchnorm_list', required=False)
    parser.add_argument('--act_fun_list', required=False)
    parser.add_argument('--dropout_list', required=False)
    parser.add_argument('--batch_size', required=False)
    parser.add_argument('--lr', required=False)
    parser.add_argument('--name_tag', help='A string that will be added to the CSV file name generated by this program',
                        required=False)
    parser.add_argument('--width', help='Fingerprint width used for training', required=False)

    args = parser.parse_args()
    # machine = args.machine
    # num_epochs = int(args.num_epochs)
    # batch_size = int(args.batch_size)
    # resume = bool(int(args.resume))
    # width = args.width
    # model_type = args.model_type

    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    if bool(int(args.train)) is True:
        best_config = {'width': args.width,
                       'first_layer_size': int(args.first_layer_size),
                       'code_layer_size': int(args.code_layer_size),
                       'num_layers': int(args.num_layers),
                       'batch_size': int(args.batch_size),
                       'lr': float(args.lr)
                       }

        train_auto_encoder(best_config, local_dir="/scratch/l/lstein/ftaj/",
                           data_dir='~/.conda/envs/drp1/Data/DRP_Training_Data/')
        exit()
    # You can change the number of GPUs per trial here: (using int rounds floats down)
    main(num_samples=int(args.num_samples), max_num_epochs=int(args.max_num_epochs),
         gpus_per_trial=float(args.gpus_per_trial),
         cpus_per_trial=int(int(args.init_cpus) / int(args.init_gpus)) * float(args.gpus_per_trial))
