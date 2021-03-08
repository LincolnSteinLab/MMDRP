# This script contains the training pipeline for the DRP module of the FULL DRP MODEL. It can take any of the omics data
# plus the drug auto-encoders, extract the trained encoders and use associated data types to train the DRP module.

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import ray
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import Repeater
from ray.tune.suggest.hyperopt import HyperOptSearch
from torch.utils import data

from CustomFunctions import AverageMeter, ProgressMeter, EarlyStopping
from DRPPreparation import drp_load_datatypes, drp_create_datasets, drp_main_prep
from DataImportModules import DRCurveData
from Models import DrugResponsePredictor
from ModuleLoader import ExtractEncoder
from TrainFunctions import drp_train
from TuneTrainables import FullModelTrainable, DRPTrainable

# NUM_CPU = multiprocessing.cpu_count()
# print("# of CPUs:", NUM_CPU)
NUM_CPU = 0
use_cuda = torch.cuda.is_available()

if use_cuda:
    print('CUDA is available. # of GPUs:', torch.cuda.device_count())

file_name_dict = {"drug_file_name": "CTRP_AAC_MORGAN.hdf",
                  "mut_file_name": "DepMap_20Q2_CGC_Mutations_by_Cell.hdf",
                  "cnv_file_name": "DepMap_20Q2_CopyNumber.hdf",
                  "exp_file_name": "DepMap_20Q2_Expression.hdf",
                  "prot_file_name": "DepMap_20Q2_No_NA_ProteinQuant.hdf",
                  "tum_file_name": "DepMap_20Q2_Line_Info.csv",
                  "gdsc1_file_name": "GDSC1_AAC_MORGAN.hdf",
                  "gdsc2_file_name": "GDSC2_AAC_MORGAN.hdf",
                  "mut_embed_file_name": "optimal_autoencoders/MUT_Omic_AutoEncoder_Checkpoint",
                  "cnv_embed_file_name": "optimal_autoencoders/CNV_Omic_AutoEncoder_Checkpoint",
                  "exp_embed_file_name": "optimal_autoencoders/EXP_Omic_AutoEncoder_Checkpoint",
                  "prot_embed_file_name": "optimal_autoencoders/PROT_Omic_AutoEncoder_Checkpoint",
                  "4096_drug_embed_file_name": "optimal_autoencoders/Morgan_4096_AutoEncoder_Checkpoint",
                  "2048_drug_embed_file_name": "optimal_autoencoders/Morgan_2048_AutoEncoder_Checkpoint",
                  "1024_drug_embed_file_name": "optimal_autoencoders/Morgan_1024_AutoEncoder_Checkpoint",
                  "512_drug_embed_file_name": "optimal_autoencoders/Morgan_512_AutoEncoder_Checkpoint"}

activation_dict = nn.ModuleDict({
    'lrelu': nn.LeakyReLU(),
    'prelu': nn.PReLU(),
    'relu': nn.ReLU(),
    'none': nn.Identity()
})


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

    criterion = nn.MSELoss().cuda()
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
                                          encoder_requires_grad=bool(int(args.encoder_train)))

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
            print("Total Average MSE Loss on", len(test_loader), "batches:", np.asarray(test_losses).mean())

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


def train_full_model(args):
    """
    Takes the data combo and associated model from the main_worker function and trains the model for specified epochs.
    :param args:
    :return:
    """
    main_start = time.time()
    num_epochs = int(args.num_epochs)
    # opt_level = args.opt_level
    batch_size = int(args.batch_size)
    resume = bool(int(args.resume))
    module_list = args.data_types

    local_dir = "/scratch/l/lstein/ftaj/"
    checkpoint_dir = local_dir + '/' + args.dir_tag
    print("Final file directory:", checkpoint_dir)

    prep_gen = drp_main_prep(module_list, args.train_file)
    for prep_list in prep_gen:
        multi_gpu, final_address, subset_data, subset_keys, subset_encoders, data_list, key_columns, required_data_indices = prep_list
        # print("required_data_indices are:", required_data_indices)
        # Create model, convert to float, and upload to the device (GPU)
        cur_layer_sizes = list(np.linspace(int(args.drp_first_layer_size),
                                           int(args.drp_last_layer_size),
                                           int(args.drp_num_layers)).astype(int))
        cur_layer_sizes.append(1)
        model_args = [cur_layer_sizes, [0] * len(subset_encoders),
                      False, activation_dict[args.act_fun],
                      bool(int(args.batchnorm)),
                      float(args.dropout)]
        for encoder in subset_encoders:
            model_args.append(encoder)
        cur_model = DrugResponsePredictor(*model_args)

        # NOTE: Must move model to GPU before initializing optimizers!!!
        cur_model = cur_model.float()
        cur_model = cur_model.cuda()
        optimizer = optim.Adam(cur_model.parameters(), lr=float(args.lr))
        criterion = nn.MSELoss().cuda()

        start_epoch = 0
        if resume is True:
            # Load model checkpoint, optimizer state, last epoch and loss
            model_path = checkpoint_dir + '/' + final_address + '_' + args.name_tag + "_DRP_Checkpoint.pt"
            checkpoint = torch.load(model_path)
            # Strict=True ensures that the models are identical
            cur_model.load_state_dict(checkpoint['model_stat_dict'], strict=True)

            # Load training and validation indices
            try:
                check_train_idx = checkpoint['train_idx']
                check_valid_idx = checkpoint['valid_idx']
            except:
                print(
                    "RESUME: Previous training and validation indices weren't found!\nThis makes valid_loss calculation after resume meaningless as validation set is different and incomparable.\nQuitting...")
                exit()
            # Load previous loss information
            try:
                best_loss = checkpoint['loss']
                print("RESUME: Best previous loss is:", best_loss)
                early_stopping = EarlyStopping(train_idx=check_train_idx, valid_idx=check_valid_idx, patience=10,
                                               delta=0.0001,
                                               verbose=True, best_score=best_loss,
                                               path=checkpoint_dir + '/' + final_address + '_' + args.name_tag + "_DRP_Checkpoint.pt")
            except:
                print("RESUME: No previous loss information (lowest loss score) found, assuming inf...")
            # Load previous optimizer state
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print("RESUME: Optimizer wasn't loaded, re-initializing...")
            # Load the last epoch the model was trained on
            try:
                start_epoch = checkpoint['epoch']
                print("Last checkpoint is at epoch", start_epoch)
            except:
                print("RESUME: No previous epoch information found, starting from 0...")

            # Remove the loaded checkpoint (large file) as it's no longer needed
            del checkpoint
            # amp.load_state_dict(checkpoint['amp'])

        else:
            # The following line sets up mixed-precision training: O0 for FP32, O3 for FP16, and O1/O2 for in between
            # cur_model, optimizer = amp.initialize(cur_model, optimizer, opt_level=opt_level)
            print("Training model from scratch!")
            print("Final file name:",
                  checkpoint_dir + '/' + final_address + '_' + args.name_tag + "_DRP_Checkpoint.pt")
        # Must indicate the required_data_indices for the current combination if we are restricting the data to the
        # subset (bottleneck) that is available to all modules (to make training and combo comparison fair)

        if bool(int(args.bottleneck)) is True:
            print("Training on bottleneck data (subset of complete samples available to all modules)")

        if resume is True:
            # Use train and validation indices from checkpoint
            train_data, train_sampler, valid_sampler, train_idx, valid_idx = drp_create_datasets(data_list, key_columns,
                                                                                                 train_idx=check_train_idx,
                                                                                                 valid_idx=check_valid_idx,
                                                                                                 drug_index=0,
                                                                                                 drug_dr_column="area_above_curve",
                                                                                                 test_drug_data=None,
                                                                                                 bottleneck=bool(int(
                                                                                                     args.bottleneck)),
                                                                                                 required_data_indices=required_data_indices)
        else:
            # Then the function will automatically make new indices and return them for saving in the checkpoint file
            train_data, train_sampler, valid_sampler, train_idx, valid_idx = drp_create_datasets(data_list, key_columns,
                                                                                                 drug_index=0,
                                                                                                 drug_dr_column="area_above_curve",
                                                                                                 test_drug_data=None,
                                                                                                 bottleneck=bool(int(
                                                                                                     args.bottleneck)),
                                                                                                 required_data_indices=required_data_indices)

        train_loader = data.DataLoader(train_data, batch_size=batch_size,
                                       sampler=train_sampler,
                                       num_workers=0, pin_memory=True, drop_last=True)
        # load validation data in batches 4 times the size
        valid_loader = data.DataLoader(train_data, batch_size=batch_size * 4,
                                       sampler=valid_sampler,
                                       num_workers=0, pin_memory=True, drop_last=True)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.0001, threshold_mode="abs", patience=2,
                                                         cooldown=0, verbose=True)
        # Initialize the early_stopping object
        early_stopping = EarlyStopping(train_idx=train_idx, valid_idx=valid_idx, patience=10, delta=0.0001,
                                       verbose=True,
                                       path=checkpoint_dir + '/' + final_address + '_' + args.name_tag + "_DRP_Checkpoint.pt")

        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        print("Finished setup in", time.time() - main_start, "seconds")
        # The following for-loop is the standard way of training in Pytorch
        for epoch in range(start_epoch, num_epochs):
            start_time = time.time()

            # === Train and Validate The Model ===
            train_losses, valid_losses = drp_train(train_loader=train_loader, valid_loader=valid_loader,
                                                   cur_model=cur_model,
                                                   criterion=criterion, optimizer=optimizer, epoch=epoch,
                                                   train_len=len(train_idx), valid_len=len(valid_idx),
                                                   batch_size=batch_size)
            if args.benchmark is not None:
                break
            duration = time.time() - start_time
            print("Finished epoch in", str(epoch + 1), str(duration), "seconds")

            avg_train_losses.append(train_losses.avg)
            avg_valid_losses.append(valid_losses.avg)

            epoch_len = len(str(epoch))

            print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_losses.avg:.5f} ' +
                         f'valid_loss: {valid_losses.avg:.5f}')

            print(print_msg)
            print("Final file name:",
                  checkpoint_dir + '/' + final_address + '_' + args.name_tag + "_DRP_Checkpoint.pt")

            # early_stopping needs the validation loss to check if it has decreased,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_losses.avg, cur_model, optimizer, epoch)

            if early_stopping.early_stop:
                print("Early Stopping")
                break

            # Update learning rate scheduler
            scheduler.step(valid_losses.avg)
        print("Final file name:", checkpoint_dir + '/' + final_address + '_' + args.name_tag + "_DRP_Checkpoint.pt")
        print("Re-saving the whole model in place of checkpoint...")
        # Reload checkpointed model
        model_path = checkpoint_dir + '/' + final_address + '_' + args.name_tag + "_DRP_Checkpoint.pt"
        checkpoint = torch.load(model_path)
        cur_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        # Save the full model
        torch.save(cur_model, checkpoint_dir + '/' + final_address + '_' + args.name_tag + "_DRP_Checkpoint.pt")

        # Save entire model, not just the state dict
        # torch.save(cur_model, checkpoint_dir + '/' + final_address + '_' + args.name_tag + "_DRP_Checkpoint.pt")


def hypopt(num_samples: int = 10, max_num_epochs: int = 100, gpus_per_trial: float = 1.0, cpus_per_trial: float = 6.0):
    if args.machine == "cluster":
        local_dir = "/.mounts/labs/steinlab/scratch/ftaj/"
    elif args.machine == "mist":
        local_dir = "/scratch/l/lstein/ftaj/"
    else:
        local_dir = "//"

    if args.address == "":
        ray.init(_lru_evict=True, num_cpus=int(args.init_cpus), num_gpus=int(args.init_gpus))
    else:
        # When connecting to an existing cluster, _lru_evict, num_cpus and num_gpus must not be provided.
        ray.init(address=args.address)

    if bool(int(args.full)) is True:
        # Full DRP
        config = {
            "train_file": args.train_file,
            "data_types": '_'.join(args.data_types),
            "bottleneck": tune.choice([True, False]),
            "mut_first_layer_size": tune.randint(2 ** 7, 2 ** 10),
            "mut_code_layer_size": tune.randint(2 ** 6, 2 ** 9),
            "mut_num_layers": tune.randint(2, 4),
            "cnv_first_layer_size": tune.randint(2 ** 9, 10000),
            "cnv_code_layer_size": tune.randint(2 ** 8, 6000),
            "cnv_num_layers": tune.randint(2, 4),
            "exp_first_layer_size": tune.randint(2 ** 9, 10000),
            "exp_code_layer_size": tune.randint(2 ** 8, 6000),
            "exp_num_layers": tune.randint(2, 4),
            "prot_first_layer_size": tune.randint(2 ** 9, 6000),
            "prot_code_layer_size": tune.randint(2 ** 8, 5000),
            "prot_num_layers": tune.randint(2, 4),
            "drp_first_layer_size": tune.randint(2 ** 9, 6000),
            "drp_last_layer_size": tune.randint(2 ** 8, 1000),
            "drp_num_layers": tune.randint(2, 5),
            "lr": tune.loguniform(1e-4, 1e-3),
            "batch_size": tune.choice([4, 8, 16, 32])
        }
        current_best_parameters = [{
            "train_file": args.train_file,
            "mut_first_layer_size": 512,
            "mut_code_layer_size": 128,
            "mut_num_layers": 2,
            "cnv_first_layer_size": 8192,
            "cnv_code_layer_size": 1024,
            "cnv_num_layers": 2,
            "exp_first_layer_size": 8192,
            "exp_code_layer_size": 2048,
            "exp_num_layers": 2,
            "prot_first_layer_size": 2048,
            "prot_code_layer_size": 1024,
            "prot_num_layers": 3,
            "drp_first_layer_size": 2048,
            "drp_last_layer_size": 256,
            "drp_num_layers": 3,
            "lr": 0.001,
            "batch_size": 4
        }]

    else:
        config = {
            "train_file": args.train_file,
            "data_types": '_'.join(args.data_types),
            "bottleneck": tune.choice([True, False]),
            # NOTE: n_folds should only have one possibility
            "n_folds": int(args.n_folds),
            "drp_first_layer_size": tune.randint(2 ** 9, 6000),
            "drp_last_layer_size": tune.randint(2 ** 8, 1000),
            "drp_num_layers": tune.randint(2, 5),
            "lr": tune.loguniform(1e-4, 1e-3),
            "batchnorm": tune.choice([True, False]),
            "act_fun": tune.choice(['none', 'relu', 'prelu', 'lrelu']),
            "dropout": tune.uniform(0, 0.25),
            "batch_size": tune.choice([4, 8, 16, 32])
        }
        current_best_parameters = [{
            "train_file": args.train_file,
            "data_types": '_'.join(args.data_types),
            "bottleneck": True,
            # NOTE: n_folds should only have one possibility
            "n_folds": int(args.n_folds),
            "drp_first_layer_size": 5980,
            "drp_last_layer_size": 314,
            "drp_num_layers": 4,
            "lr": 0.000116483,
            "batchnorm": True,
            "act_fun": 'lrelu',
            "dropout": 0.0016931,
            "batch_size": 32
        }]

    # Must NOT use an early-stopping TrialScheduler when performing cross-validation
    if int(args.n_folds) == 1:
        scheduler = ASHAScheduler(
            grace_period=3,
            reduction_factor=3,
            brackets=3)
        search_algo = HyperOptSearch(points_to_evaluate=current_best_parameters)
    else:
        scheduler = None
        search_algo = Repeater(HyperOptSearch(points_to_evaluate=current_best_parameters),
                               repeat=int(args.n_folds), set_index=True)

    reporter = CLIReporter(
        metric_columns=["sum_valid_loss", "avg_valid_loss", "sum_train_loss", "training_iteration", "num_samples", "time_this_iter_s"])

    if bool(int(args.full)) is True:
        tag = "FullModel"
        cur_trainable = FullModelTrainable
    else:
        tag = "ResponseOnly"
        cur_trainable = DRPTrainable
    print("Checkpoint directory:", "HyperOpt_Test_DRP_" + tag + '_' + "_".join(args.data_types) + '_' + args.name_tag)

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
        name="HyperOpt_Test_DRP_" + tag + '_' + "_".join(args.data_types) + '_' + args.name_tag,
        verbose=1,
        resume=resume,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},  # "memory": 8 * 10 ** 9},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=search_algo,
        progress_reporter=reporter,
        metric='sum_valid_loss',
        mode='min',
        keep_checkpoints_num=1,
        stop={"training_iteration": max_num_epochs},
        checkpoint_at_end=True,
        # checkpoint_score_attr="min-loss",
        reuse_actors=True, local_dir=local_dir)

    best_trial = result.get_best_trial(metric="sum_valid_loss", mode="min", scope="all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["sum_valid_loss"]))
    print("Best trial number of epochs:", best_trial.last_result["training_iteration"])


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
    parser.add_argument('--num_samples',
                        help="Number of samples drawn from parameter combinations", required=False)
    # parser.add_argument('--fp_width', help="Width of the drug fingerprints used")
    parser.add_argument('--full', help="whether to optimize all modules (full) or just the DRP module itself",
                        default='0')


    parser.add_argument('--data_types', nargs="+", help='List of data types to be used in training this DRP model')
    parser.add_argument('--bottleneck',
                        help='Whether to restrict the training data to complete samples that have all 4 omic data types',
                        default="0")

    # Training parameters ###
    parser.add_argument('--train_file', help='Name of file used for training model, e.g. CTRP AAC data.')
    parser.add_argument('--num_epochs', help='Number of epochs to run', required=False)
    parser.add_argument('--batch_size', help='Size of each training batch', required=False)
    parser.add_argument('--resume',
                        help='Whether to continue training from saved model at epoch, or start training from scratch. Required for model interpretation.',
                        default="0")
    parser.add_argument('--test', help='Name of the testing data file if testing is requested')
    parser.add_argument('--interpret', help='Whether a fully trained model should be interpreted using SHAP',
                        default="0")
    parser.add_argument('--all_combos', help="Train a separate model on all possible combinations of omic data types",
                        default="0")
    parser.add_argument('--encoder_train', help="Whether to freeze (0) or unfreeze (1) encoder modules", default="0")
    parser.add_argument('--name_tag',
                        help='A string that will be added to the model checkpoint file name generated by this program',
                        required=True)
    parser.add_argument('--dir_tag', help='The name of the subdirectory where checkpoints will be saved',
                        required=False)

    parser.add_argument('--benchmark',
                        help="Whether the code should be run in benchmark mode for torch.utils.bottleneck")
    parser.add_argument('--force', help="Whether to force a re-run of training if a checkpoint already exists")

    parser.add_argument('--drp_first_layer_size')
    parser.add_argument('--drp_last_layer_size')
    parser.add_argument('--drp_num_layers')
    parser.add_argument('--lr')
    parser.add_argument('--batchnorm')
    parser.add_argument('--act_fun')
    parser.add_argument('--dropout')

    # parser.add_argument("--drug_file_name", help="Address of the drug SMILES + DR targets used for model evaluation")

    args = parser.parse_args()

    if args.train_file is not None:
        assert os.path.isfile(
            "Data/DRP_Training_Data/" + args.train_file), "Given train file could not be found in main directory"

    # Create a checkpoint directory based on given name_tag if it doesn't exist
    if not os.path.exists(args.name_tag):
        print("Creating checkpoint directory:", args.name_tag)
        os.makedirs(args.name_tag)

    # Path("/" + args.name_tag).mkdir(exist_ok=True)

    if bool(int(args.interpret)) is True:
        assert bool(int(args.resume)) is True, "Must have a fully trained model for interpretation, with resume mode on"

    # Turning off benchmarking makes highly dynamic models faster
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    if bool(int(args.optimize)) is True:
        hypopt(num_samples=int(args.num_samples), max_num_epochs=int(args.max_num_epochs),
               gpus_per_trial=float(args.gpus_per_trial),
               cpus_per_trial=int(int(args.init_cpus) / int(args.init_gpus)) * float(args.gpus_per_trial))

        exit()
    if args.test is None:
        # TODO
        assert len(args.data_types) > 1, "Multiple valid data types should be provided"
        # Perform training and validation
        train_full_model(args)
    else:
        # Test the model
        test(args=args)
