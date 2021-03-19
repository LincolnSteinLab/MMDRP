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

from DRP.src.DataImportModules import OmicData
from Models import DNNAutoEncoder
from DRP.src.TrainFunctions import omic_train
from TuneTrainables import OmicTrainable

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

print("Current working directory is:", os.getcwd())

use_cuda = torch.cuda.is_available()
torch.cuda.device_count()
device = torch.device("cuda:0" if use_cuda else "cpu")
# Turning off benchmarking makes highly dynamic models faster
cudnn.benchmark = False
cudnn.deterministic = True

file_name_dict = {"drug_file_name": "CTRP_AAC_MORGAN.hdf",
                  "mut_file_name": "DepMap_20Q2_CGC_Mutations_by_Cell.hdf",
                  "cnv_file_name": "DepMap_20Q2_CopyNumber.hdf",
                  "exp_file_name": "DepMap_20Q2_Expression.hdf",
                  "prot_file_name": "DepMap_20Q2_No_NA_ProteinQuant.hdf",
                  "tum_file_name": "DepMap_20Q2_Line_Info.csv",
                  "gdsc1_file_name": "GDSC1_AAC_MORGAN.hdf",
                  "gdsc2_file_name": "GDSC2_AAC_MORGAN.hdf"}


def train_auto_encoder(args, config, local_dir, data_dir):
    # batch_size = int(args.batch_size)
    # resume = bool(int(args.resume))
    num_epochs = int(args.num_epochs)
    checkpoint_dir = local_dir + '/' + args.name_tag

    # Create a checkpoint directory based on given name_tag if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        print("Creating checkpoint directory:", args.name_tag)
        os.makedirs(checkpoint_dir)

    training_set = OmicData(path=data_dir, omic_file_name=file_name_dict[args.data_type + "_file_name"])
    train_loader = data.DataLoader(training_set, batch_size=config["batch_size"], num_workers=0, shuffle=True,
                                   pin_memory=False)
    # TODO Add other hyperparameters
    cur_model = DNNAutoEncoder(input_dim=training_set.width(), first_layer_size=config["first_layer_size"],
                               code_layer_size=config["code_layer_size"], num_layers=config["num_layers"],
                               name=args.data_type)
    cur_model = cur_model.float()
    cur_model = cur_model.cuda()

    last_epoch = 0

    criterion = nn.MSELoss().cuda()
    # optimizer = optim.Adam(cur_model.parameters(), lr=0.001)
    optimizer = optim.Adam(cur_model.parameters(), lr=config["lr"])

    # Initialize the early_stopping object, don't save for the first N epochs
    # early_stopping = AutoEncoderEarlyStopping(patience=10, delta=0.0001,
    #                                           verbose=True, ignore=20,
    #                                           path=args.name_tag + '/' + args.data_type + '_embed_pytorch.pth.tar')

    # NOTE: Don't use scheduler for population based training
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, threshold_mode="rel", patience=2,
    #                                                  cooldown=0, verbose=True)

    # if resume is True:
    #     # Load model checkpoint, optimizer state, last epoch and loss
    #     model_path = 'exp_embed_pytorch.pth.tar'
    #     checkpoint = torch.load(model_path, map_location=device)
    #     cur_model.load_state_dict(checkpoint['state_dict'], strict=True)
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     last_epoch = checkpoint['epoch']
    #     # loss = checkpoint['loss']

    # if checkpoint_dir:
    #     model_state, optimizer_state = torch.load(
    #         os.path.join(checkpoint_dir, args.data_type.upper() + "_Omic_AutoEncoder_Checkpoint"))
    #     cur_model.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)

    # cur_model, optimizer = amp.initialize(cur_model, optimizer, opt_level=opt_level)

    # Loop over epochs
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    start = time.time()
    for epoch in range(last_epoch, num_epochs):
        train_losses = omic_train(train_loader, cur_model, criterion, optimizer, epoch=epoch)

        avg_train_losses.append(train_losses.avg)

        duration = time.time() - start
        print_msg = (f'[{epoch:>{num_epochs}}/{200:>{num_epochs}}] ' +
                     f'train_loss: {train_losses.avg:.5f} ' +
                     f'epoch_time: {duration:.4f}')

        print(print_msg)
        start = time.time()

        # print("Finished epoch in", str(epoch + 1), str(duration), "seconds")
    print("Final file name:", checkpoint_dir + '/' + args.data_type.upper() + "_Omic_AutoEncoder_Checkpoint")
    # Save entire model, not just the state dict
    torch.save(cur_model, checkpoint_dir + '/' + args.data_type.upper() + "_Omic_AutoEncoder_Checkpoint.pt")
    # torch.save({
    #     'epoch': num_epochs,
    #     'model_state_dict': cur_model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': avg_train_losses
    # }, checkpoint_dir + '/' + args.data_type.upper() + "_Omic_AutoEncoder_Checkpoint")
        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((cur_model.state_dict(), optimizer.state_dict()), path)
        # tune.report(loss=(train_losses.sum / (len(training_set) // config["batch_size"])))
        # early_stopping(train_losses.avg, cur_model, optimizer, epoch)
        #
        # if early_stopping.early_stop:
        #     print("Early Stopping")
        #     break

        # Update learning rate scheduler
        # scheduler.step(train_losses.avg)
    print("Finished training!")


def main(num_samples=10, max_num_epochs=100, gpus_per_trial=1.0, cpus_per_trial=6.0):
    if args.machine == "cluster":
        local_dir = "/.mounts/labs/steinlab/scratch/ftaj/"
        ray.init(_lru_evict=True, num_cpus=int(args.init_cpus), num_gpus=int(args.init_gpus),
                 object_store_memory=1.5 * 10 ** 10)
    elif args.machine == "mist":
        local_dir = "/scratch/l/lstein/ftaj/"
        ray.init(_lru_evict=True, num_cpus=int(args.init_cpus), num_gpus=int(args.init_gpus))
    else:
        local_dir = "//"

    if args.data_type in ["cnv", "exp"]:
        print("Data type is:", args.data_type)
        config = {
            "data_type": args.data_type,
            "first_layer_size": tune.randint(2 ** 9, 10005),
            "code_layer_size": tune.randint(2 ** 8, 6005),
            "num_layers": tune.randint(2, 5),
            "batchnorm_list": tune.choice([True, False]),
            "act_fun_list": tune.choice(['none', 'relu', 'prelu', 'lrelu']),
            "lr": tune.loguniform(1e-4, 1e-3),
            "batch_size": tune.randint(4, 32)
        }
        if args.data_type == "cnv":
            current_best_params = [{
                "data_type": "cnv",
                'first_layer_size': 9479,
                'code_layer_size': 797,
                'num_layers': 2,
                "batchnorm_list": False,
                "act_fun_list": 'relu',
                "lr": 0.0001124,
                "batch_size": 5
            }]
        else:
            current_best_params = [{
                "data_type": "exp",
                'first_layer_size': 2356,
                'code_layer_size': 532,
                'num_layers': 2,
                "batchnorm_list": False,
                "act_fun_list": 'relu',
                "lr": 0.0001076,
                "batch_size": 5
            }]

    elif args.data_type == "mut":
        print("Data type is:", args.data_type)
        config = {
            "data_type": args.data_type,
            "first_layer_size": tune.randint(2 ** 7, 2 ** 10),
            "code_layer_size": tune.randint(2 ** 6, 2 ** 9),
            "num_layers": tune.randint(2, 5),
            "batchnorm_list": tune.choice([True, False]),
            "act_fun_list": tune.choice(['none', 'relu', 'prelu', 'lrelu']),
            "lr": tune.loguniform(1e-4, 1e-3),
            "batch_size": tune.randint(4, 32)
        }
        current_best_params = [{
            "data_type": "mut",
            'first_layer_size': 763,
            'code_layer_size': 500,
            'num_layers': 2,
            "batchnorm_list": False,
            "act_fun_list": 'relu',
            "lr": 0.0002672,
            "batch_size": 32
        }]

    else:
        # prot
        print("Data type is:", args.data_type)
        config = {
            "data_type": args.data_type,
            "first_layer_size": tune.randint(2 ** 9, 6000),
            "code_layer_size": tune.randint(2 ** 8, 5000),
            "num_layers": tune.randint(2, 5),
            "batchnorm_list": tune.choice([True, False]),
            "act_fun_list": tune.choice(['none', 'relu', 'prelu', 'lrelu']),
            "lr": tune.loguniform(1e-4, 1e-3),
            "batch_size": tune.randint(4, 32)
        }
        current_best_params = [{
            "data_type": "prot",
            'first_layer_size': 2530,
            'code_layer_size': 1875,
            'num_layers': 2,
            "batchnorm_list": False,
            "act_fun_list": 'relu',
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
    scheduler = ASHAScheduler(
        grace_period=5,
        reduction_factor=3,
        brackets=3)
    hyperopt_search = HyperOptSearch(points_to_evaluate=current_best_params)

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
        OmicTrainable,
        name="HyperOpt_Test_Full_" + args.data_type, verbose=1, resume=resume,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},  # "memory": 8 * 10 ** 9},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
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
    print("Best trial final sum validation loss: {}".format(
        best_trial.last_result["sum_valid_loss"]))
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
    parser.add_argument('--name_tag', help='A string that will be added to the CSV file name generated by this program',
                        required=False)

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

    # Create a checkpoint directory based on given name_tag if it doesn't exist
    # if not os.path.exists(args.name_tag):
    #     print("Creating checkpoint directory:", args.name_tag)
    #     os.makedirs(args.name_tag)

    # Turning off benchmarking makes highly dynamic models faster
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    if bool(int(args.train)) is True:
        best_config = {'first_layer_size': int(args.first_layer_size),
                       'code_layer_size': int(args.code_layer_size),
                       'num_layers': int(args.num_layers),
                       'batchnorm_list': bool(int(args.batchnorm)),
                       'act_fun_list': args.act_fun,
                       'batch_size': int(args.batch_size),
                       'lr': float(args.lr)
                       }

        train_auto_encoder(args, best_config, local_dir="/scratch/l/lstein/ftaj/",
                           data_dir='~/.conda/envs/drp1/Data/DRP_Training_Data/')
        exit()
    # You can change the number of GPUs per trial here:
    main(num_samples=int(args.num_samples), max_num_epochs=int(args.max_num_epochs),
         gpus_per_trial=float(args.gpus_per_trial),
         cpus_per_trial=int(int(args.init_cpus) / int(args.init_gpus)) * float(args.gpus_per_trial))
