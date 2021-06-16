import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ray.tune import Trainable
from torch.utils import data

from DRPPreparation import drp_create_datasets, autoencoder_create_datasets, drp_load_datatypes, \
    create_cv_folds
from DataImportModules import OmicData, MorganData
from Models import DNNAutoEncoder, MultiHeadCNNAutoEncoder, DrugResponsePredictor
from ModuleLoader import ExtractEncoder
from TrainFunctions import morgan_train, omic_train, drp_train, cross_validate

file_name_dict = {"drug_file_name": "CTRP_AAC_MORGAN.hdf",
                  "pretrain_exp_file_name": "TCGA_PreTraining_Expression.hdf",
                  "pretrain_cnv_file_name": "TCGA_PreTraining_CopyNumber.hdf",
                  "mut_file_name": "DepMap_21Q2_Mutations_by_Cell.hdf",
                  "cnv_file_name": "DepMap_21Q2_Training_CopyNumber.hdf",
                  "exp_file_name": "DepMap_21Q2_Training_Expression.hdf",
                  "prot_file_name": "DepMap_20Q2_No_NA_ProteinQuant.hdf",
                  "tum_file_name": "DepMap_21Q2_Line_Info.csv",
                  "gdsc1_file_name": "GDSC1_AAC_MORGAN.hdf",
                  "gdsc2_file_name": "GDSC2_AAC_MORGAN.hdf",
                  "mut_embed_file_name": "optimal_autoencoders/MUT_Omic_AutoEncoder_Checkpoint.pt",
                  "cnv_embed_file_name": "optimal_autoencoders/CNV_Omic_AutoEncoder_Checkpoint.pt",
                  "pretrain_cnv_embed_file_name": "optimal_autoencoders/CNV_pretrain_Omic_AutoEncoder_Checkpoint.pt",
                  "exp_embed_file_name": "optimal_autoencoders/EXP_Omic_AutoEncoder_Checkpoint.pt",
                  "pretrain_exp_embed_file_name": "optimal_autoencoders/EXP_pretrain_Omic_AutoEncoder_Checkpoint.pt",
                  "prot_embed_file_name": "optimal_autoencoders/PROT_Omic_AutoEncoder_Checkpoint.pt",
                  "4096_drug_embed_file_name": "optimal_autoencoders/Morgan_4096_AutoEncoder_Checkpoint.pt",
                  "2048_drug_embed_file_name": "optimal_autoencoders/Morgan_2048_AutoEncoder_Checkpoint.pt",
                  "1024_drug_embed_file_name": "optimal_autoencoders/Morgan_1024_AutoEncoder_Checkpoint.pt",
                  "512_drug_embed_file_name": "optimal_autoencoders/Morgan_512_AutoEncoder_Checkpoint.pt"}

NUM_WORKERS = 0


def get_layer_configs(config, model_type="dnn"):
    all_act_funcs = []
    all_batch_norm = []
    all_dropout = []
    num_layers_or_branch = ""
    if model_type == "dnn":
        num_layers_or_branch = "num_layers"
    else:
        num_layers_or_branch = "num_branch"

    for i in range(config[num_layers_or_branch]):
        all_act_funcs.append(config['activation_' + str(i + 1)])
        all_batch_norm.append(config['batch_norm_' + str(i + 1)])
        all_dropout.append(config['dropout_' + str(i + 1)])

    return all_act_funcs, all_batch_norm, all_dropout


def initialize_autoencoders(cur_modules: [str], cur_config: {}, drug_width: int, mut_width: int,
                            cnv_width: int, exp_width: int, prot_width: int):
    cur_autoencoders = []
    drug_model = DNNAutoEncoder(input_dim=drug_width,
                                first_layer_size=cur_config['morgan_first_layer_size'],
                                code_layer_size=cur_config['morgan_code_layer_size'],
                                num_layers=cur_config['morgan_num_layers'],
                                batchnorm_list=[True] * (cur_config['morgan_num_layers'] - 1) + [False],
                                dropout_list=[0.0] + [0.05] * (cur_config['morgan_num_layers'] - 2) + [0.0],
                                name="drug")
    cur_autoencoders.append(drug_model)

    if 'mut' in cur_modules:
        mut_model = DNNAutoEncoder(input_dim=mut_width,
                                   first_layer_size=cur_config['mut_first_layer_size'],
                                   code_layer_size=cur_config['mut_code_layer_size'],
                                   batchnorm_list=[True] * (cur_config['mut_num_layers'] - 1) + [False],
                                   dropout_list=[0.0] + [0.05] * (cur_config['mut_num_layers'] - 2) + [0.0],
                                   num_layers=cur_config['mut_num_layers'], name="mut")
        cur_autoencoders.append(mut_model)
    if 'cnv' in cur_modules:
        cnv_model = DNNAutoEncoder(input_dim=cnv_width,
                                   first_layer_size=cur_config['cnv_first_layer_size'],
                                   code_layer_size=cur_config['cnv_code_layer_size'],
                                   batchnorm_list=[True] * (cur_config['cnv_num_layers'] - 1) + [False],
                                   dropout_list=[0.0] + [0.05] * (cur_config['cnv_num_layers'] - 2) + [0.0],
                                   num_layers=cur_config['cnv_num_layers'], name="cnv")
        cur_autoencoders.append(cnv_model)
    if 'exp' in cur_modules:
        exp_model = DNNAutoEncoder(input_dim=exp_width,
                                   first_layer_size=cur_config['exp_first_layer_size'],
                                   code_layer_size=cur_config['exp_code_layer_size'],
                                   batchnorm_list=[True] * (cur_config['exp_num_layers'] - 1) + [False],
                                   dropout_list=[0.0] + [0.05] * (cur_config['exp_num_layers'] - 2) + [0.0],
                                   num_layers=cur_config['exp_num_layers'], name="exp")
        cur_autoencoders.append(exp_model)
    if 'prot' in cur_modules:
        prot_model = DNNAutoEncoder(input_dim=prot_width,
                                    first_layer_size=cur_config['prot_first_layer_size'],
                                    code_layer_size=cur_config['prot_code_layer_size'],
                                    batchnorm_list=[True] * (cur_config['prot_num_layers'] - 1) + [False],
                                    dropout_list=[0.0] + [0.05] * (cur_config['prot_num_layers'] - 2) + [0.0],
                                    num_layers=cur_config['prot_num_layers'], name="prot")
        cur_autoencoders.append(prot_model)

    return cur_autoencoders


class MorganTrainable(Trainable):
    def setup(self, config, checkpoint_dir="/.mounts/labs/steinlab/scratch/ftaj/"):
        self.data_dir = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/"
        # self.data_dir = "/u/ftaj/anaconda3/envs/Python/Data/DRP_Training_Data/"
        # self.data_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
        self.timestep = 0

        # Prepare data loaders =========
        morgan_file_name = "ChEMBL_Morgan_" + config["width"] + ".pkl"
        # For reverse compatibility
        try:
            self.model_type = config["model_type"]
        except:
            self.model_type = "dnn"
        self.train_data = MorganData(path=self.data_dir, morgan_file_name=morgan_file_name,
                                     model_type=self.model_type)
        assert self.train_data.width() == int(
            config["width"]), "The width of the given training set doesn't match the expected Morgan width."

        print("Train data length:", len(self.train_data))

        self.train_data, self.train_sampler, self.valid_sampler, \
        self.train_idx, self.valid_idx = autoencoder_create_datasets(train_data=self.train_data)
        # Create data_loaders
        self.train_loader = data.DataLoader(self.train_data, batch_size=config["batch_size"],
                                            sampler=self.train_sampler, num_workers=NUM_WORKERS,
                                            pin_memory=True, drop_last=True)
        self.valid_loader = data.DataLoader(self.train_data, batch_size=config["batch_size"] * 4,
                                            sampler=self.valid_sampler, num_workers=NUM_WORKERS,
                                            pin_memory=True, drop_last=True)

        # Prepare model =========
        if self.model_type == "cnn":
            all_act_funcs, all_batch_norm, all_dropout = get_layer_configs(self.config, model_type="cnn")
            self.cur_model = MultiHeadCNNAutoEncoder(
                input_width=int(self.config["width"]),
                num_branch=self.config["num_branch"],
                stride=self.config["stride"],
                batchnorm_list=all_batch_norm,
                act_fun_list=all_act_funcs,
                dropout_list=all_dropout
            )
        else:
            all_act_funcs, all_batch_norm, all_dropout = get_layer_configs(self.config, model_type="dnn")
            self.cur_model = DNNAutoEncoder(input_dim=int(self.config["width"]),
                                            first_layer_size=self.config["first_layer_size"],
                                            code_layer_size=self.config["code_layer_size"],
                                            num_layers=self.config["num_layers"],
                                            batchnorm_list=all_batch_norm,
                                            act_fun_list=all_act_funcs,
                                            dropout_list=all_dropout)

        self.cur_model = self.cur_model.float()
        self.cur_model = self.cur_model.cuda()
        self.criterion = nn.L1Loss().cuda()
        self.optimizer = optim.Adam(self.cur_model.parameters(), lr=self.config["lr"])

    def reset_config(self, new_config):
        try:
            # Reset model with new_config ==============
            if self.model_type == "cnn":
                all_act_funcs, all_batch_norm, all_dropout = get_layer_configs(self.config, model_type="cnn")
                self.cur_model = MultiHeadCNNAutoEncoder(
                    input_width=int(new_config["width"]),
                    num_branch=new_config["num_branch"],
                    stride=new_config["stride"],
                    batchnorm_list=all_batch_norm,
                    act_fun_list=all_act_funcs,
                    dropout_list=all_dropout
                )
            else:
                all_act_funcs, all_batch_norm, all_dropout = get_layer_configs(self.config, model_type="dnn")
                self.cur_model = DNNAutoEncoder(
                    input_dim=int(new_config["width"]),
                    first_layer_size=new_config["first_layer_size"],
                    code_layer_size=new_config["code_layer_size"],
                    num_layers=new_config["num_layers"],
                    batchnorm_list=all_batch_norm,
                    act_fun_list=all_act_funcs,
                    dropout_list=all_dropout
                )
            self.cur_model = self.cur_model.float()
            self.cur_model = self.cur_model.cuda()
            self.criterion = nn.L1Loss().cuda()
            self.optimizer = optim.Adam(self.cur_model.parameters(), lr=new_config["lr"])

            # Reset data loaders =================
            self.train_loader = data.DataLoader(self.train_data, batch_size=new_config["batch_size"],
                                                sampler=self.train_sampler, num_workers=NUM_WORKERS,
                                                pin_memory=True, drop_last=True)
            self.valid_loader = data.DataLoader(self.train_data, batch_size=new_config["batch_size"] * 4,
                                                sampler=self.valid_sampler, num_workers=NUM_WORKERS,
                                                pin_memory=True, drop_last=True)
            return True
        except:
            return False

    def step(self):
        self.timestep += 1
        # to track the average training loss per epoch as the model trains
        # avg_train_losses = []
        # for epoch in range(last_epoch, num_epochs):
        start_time = time.time()
        train_losses, valid_losses = morgan_train(train_loader=self.train_loader, valid_loader=self.valid_loader,
                                                  cur_model=self.cur_model, criterion=self.criterion,
                                                  optimizer=self.optimizer, epoch=self.timestep)

        duration = time.time() - start_time
        # avg_train_losses.append(train_losses.avg)
        self.sum_train_loss = train_losses.sum
        self.sum_valid_loss = valid_losses.sum
        self.avg_valid_loss = valid_losses.avg

        return {"sum_valid_loss": self.sum_valid_loss,
                "avg_valid_loss": self.avg_valid_loss,
                "sum_train_loss": self.sum_train_loss,
                "time_this_iter_s": duration}

    def save_checkpoint(self, tmp_checkpoint_dir):
        path = os.path.join(tmp_checkpoint_dir, "checkpoint")
        torch.save({"model_state_dict": self.cur_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "timestep": self.timestep}, path)

        return path

    def load_checkpoint(self, checkpoint):
        cur_checkpoint = torch.load(checkpoint)
        self.cur_model.load_state_dict(cur_checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(cur_checkpoint["optimizer_state_dict"])
        self.timestep = cur_checkpoint["timestep"]


class OmicTrainable(Trainable):
    def setup(self, config, max_epochs: int = None, pretrain: bool = None, data_type: str = None, n_folds: int = None,
              checkpoint_dir="/.mounts/labs/steinlab/scratch/ftaj/"):
        self.data_dir = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/"
        self.cv_fold_idx = -1
        self.timestep = 0
        self.config = config
        self.max_epochs = max_epochs
        self.data_type = data_type
        self.bottleneck_path = "bottleneck_keys.csv"
        self.bottleneck_keys = None
        self.pretrain = pretrain
        self.n_folds = n_folds
        self.max_epochs = max_epochs
        self.timestep = 0
        self.config = config
        self.batch_size = config['batch_size']

        # Prepare data loaders ==========
        file_name = file_name_dict[self.pretrain + self.data_type + "_file_name"]
        self.train_data = OmicData(path=self.data_dir,
                                   omic_file_name=file_name,
                                   to_gpu=True)

        print("Train data length:", len(self.train_data))

        if self.pretrain != "":
            cur_sample_column_name = "tcga_sample_id"
        else:
            cur_sample_column_name = "stripped_cell_line_name"

        self.cv_folds = create_cv_folds(train_data=self.train_data, train_attribute_name="data_info",
                                        sample_column_name=cur_sample_column_name, n_folds=self.n_folds,
                                        class_data_index=None, subset_type="cell_line",
                                        class_column_name="primary_disease", seed=42, verbose=False)

        # Prepare model =============
        all_act_funcs, all_batch_norm, all_dropout = get_layer_configs(self.config, model_type="dnn")
        self.cur_model = DNNAutoEncoder(
            input_dim=self.train_data.width(),
            first_layer_size=self.config["first_layer_size"],
            code_layer_size=self.config["code_layer_size"],
            num_layers=self.config["num_layers"],
            batchnorm_list=all_batch_norm,
            act_fun_list=all_act_funcs,
            dropout_list=all_dropout
        )

        self.cur_model = self.cur_model.float()
        self.cur_model = self.cur_model.cuda()
        self.criterion = nn.L1Loss().cuda()
        self.learning_rate = config['lr']
        # self.optimizer = optim.Adam(self.cur_model.parameters(), lr=self.config["lr"])

    def reset_config(self, new_config):
        try:
            self.batch_size = new_config['batch_size']
            # Reset model with new_config ===========
            all_act_funcs, all_batch_norm, all_dropout = get_layer_configs(new_config, model_type="dnn")

            self.cur_model = DNNAutoEncoder(
                input_dim=self.train_data.width(),
                first_layer_size=new_config["first_layer_size"],
                code_layer_size=new_config["code_layer_size"],
                num_layers=new_config["num_layers"],
                batchnorm_list=all_batch_norm,
                act_fun_list=all_act_funcs,
                dropout_list=all_dropout
            )
            self.cur_model = self.cur_model.float()
            self.cur_model = self.cur_model.cuda()
            self.criterion = nn.L1Loss().cuda()
            self.learning_rate = new_config['lr']
            # self.optimizer = optim.Adam(self.cur_model.parameters(), lr=new_config["lr"])

            return True
        except:
            return False

    def step(self):
        # TODO: Current CV implementation does 20 epochs in 1 step, without checkpointing in between
        # is this the best way?
        # to track the average training loss per epoch as the model trains
        # avg_train_losses = []
        # for epoch in range(last_epoch, num_epochs):
        self.timestep += 1
        start_time = time.time()
        # train_losses, valid_losses = omic_train(self.train_loader, self.cur_model, self.criterion, self.optimizer,
        #                                         epoch=self.timestep, valid_loader=self.valid_loader)

        self.avg_train_losses, \
        self.avg_valid_losses, \
        self.max_final_epoch = cross_validate(train_data=self.train_data,
                                              train_function=omic_train,
                                              cv_folds=self.cv_folds,
                                              batch_size=self.batch_size,
                                              cur_model=self.cur_model,
                                              criterion=self.criterion,
                                              # optimizer=self.optimizer,
                                              max_epochs=self.max_epochs,
                                              learning_rate=self.learning_rate,
                                              NUM_WORKERS=NUM_WORKERS,
                                              theoretical_loss=True,
                                              verbose=True)

        duration = time.time() - start_time

        return {"avg_cv_train_loss": self.avg_train_losses,
                "avg_cv_valid_loss": self.avg_valid_losses,
                "max_final_epoch": self.max_final_epoch,
                "time_this_iter_s": duration}

    def save_checkpoint(self, tmp_checkpoint_dir):
        path = os.path.join(tmp_checkpoint_dir, "checkpoint")
        torch.save({"model_state_dict": self.cur_model.state_dict(),
                    # "optimizer_state_dict": self.optimizer.state_dict(),
                    "timestep": self.timestep}, path)

        return path

    def load_checkpoint(self, checkpoint):
        cur_checkpoint = torch.load(checkpoint)
        self.cur_model.load_state_dict(cur_checkpoint["model_state_dict"])
        # self.optimizer.load_state_dict(cur_checkpoint["optimizer_state_dict"])
        self.timestep = cur_checkpoint["timestep"]


class DRPTrainable(Trainable):
    """
    This class implements the scenario where only the DRP module architecture + configuration is
    being optimized, and the encoders modules already exist. However, it does allow the choice to
    freeze or unfreeze the enoder modules during the optimization of the DRP module configuration.
    """
    def setup(self, config, train_file: str = None, data_types: str = None, bottleneck: bool = None,
              pretrain: bool = None, n_folds: int = None, max_epochs: int = None, encoder_train: bool = None,
              cv_subset_type: str = None, stratify: bool = None, checkpoint_dir: str = "/.mounts/labs/steinlab/scratch/ftaj/"):
        self.data_dir = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/"
        # self.data_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data"
        self.train_file = train_file
        self.data_types = data_types
        self.bottleneck = bottleneck
        self.bottleneck_path = "bottleneck_keys.csv"
        self.bottleneck_keys = None
        self.pretrain = pretrain
        self.n_folds = n_folds
        self.max_epochs = max_epochs
        self.encoder_train = encoder_train
        self.cv_subset_type = cv_subset_type
        self.stratify = stratify
        self.timestep = 0
        self.config = config
        self.batch_size = self.config["batch_size"]

        if self.bottleneck is True:
            try:
                self.bottleneck_keys = pd.read_csv(self.data_dir + self.bottleneck_path)['keys'].to_list()
            except:
                exit("Could not read bottleneck file")

        cur_modules = self.data_types.split('_')
        assert "drug" in cur_modules, "Drug data must be provided for drug-response prediction (training or testing)"
        assert len(cur_modules) > 1, "Data types to be used must be indicated by: mut, cnv, exp, prot and drug"

        # Load data and auto-encoders
        self.data_list, autoencoder_list, self.key_columns, \
        self.gpu_locs = drp_load_datatypes(train_file=self.train_file,
                                           module_list=cur_modules,
                                           PATH=self.data_dir,
                                           file_name_dict=file_name_dict,
                                           load_autoencoders=True,
                                           _pretrain=self.pretrain,
                                           device='cpu',
                                           verbose=False)
        # Extract encoders from loaded auto-encoders
        self.encoders = [ExtractEncoder(autoencoder) for autoencoder in autoencoder_list]

        # Determine layer sizes, add final target layer
        cur_layer_sizes = list(np.linspace(self.config['drp_first_layer_size'],
                                           self.config['drp_last_layer_size'],
                                           self.config['drp_num_layers']).astype(int))
        cur_layer_sizes.append(1)
        # Note that the order of arguments must be respected here!!!
        # print("len(self.subset_encoders):", len(self.subset_encoders))

        # Create variable length argument set for DRP model creation, depending on number of given encoders
        model_args = [cur_layer_sizes, [0] * len(self.encoders),
                      self.encoder_train,
                      # TODO: last layer must use a sigmoid to ensure values are between 0 & 1
                      ['prelu'] * (config['drp_num_layers'] + 1) + ['relu'],
                      [True] * (config['drp_num_layers'] + 1) + [False],
                      [0.0] + [0.05] * config['drp_num_layers'] + [0.0]]
        for encoder in self.encoders:
            model_args.append(encoder)
        self.cur_model = DrugResponsePredictor(*model_args)

        # NOTE: Must move model to GPU before initializing optimizers!!!
        self.cur_model = self.cur_model.float()
        self.cur_model = self.cur_model.cuda()
        self.criterion = nn.L1Loss().cuda()
        # NOTE: optimizer should be defined immediately after a model is copied/moved/altered etc.
        # Not doing so will point the optimizer to a zombie model!
        # self.optimizer = optim.Adam(self.cur_model.parameters(), lr=self.config["lr"])
        self.learning_rate = config['lr']

        # Then the function will automatically make "new" indices and return them for saving in the checkpoint file
        # NOTE: The seed is the same, so the same subset is selected every time (!)
        # Load bottleneck and full data once, choose later on
        if self.bottleneck is True:
            self.cur_train_data, \
            self.cur_cv_folds = drp_create_datasets(self.data_list,
                                                    self.key_columns,
                                                    n_folds=self.n_folds,
                                                    drug_index=0,
                                                    drug_dr_column="area_above_curve",
                                                    class_column_name="primary_disease",
                                                    subset_type=self.cv_subset_type,
                                                    stratify=self.stratify,
                                                    bottleneck_keys=self.bottleneck_keys,
                                                    to_gpu=True,
                                                    verbose=False)
        else:
            self.cur_train_data, \
            self.cur_cv_folds = drp_create_datasets(self.data_list,
                                                    self.key_columns,
                                                    n_folds=self.n_folds,
                                                    drug_index=0,
                                                    drug_dr_column="area_above_curve",
                                                    class_column_name="primary_disease",
                                                    subset_type=self.cv_subset_type,
                                                    stratify=self.stratify,
                                                    test_drug_data=None,
                                                    to_gpu=True,
                                                    verbose=False)

        # if config['bottleneck'] is True:
        #     # Get train and validation indices for the current fold
        #     self.cur_cv_folds = self.bottleneck_cv_folds
        #     self.cur_train_data = self.bottleneck_train_data
        #
        # else:
        #     self.cur_cv_folds = self.cv_folds
        #     self.cur_train_data = self.train_data

    def reset_config(self, new_config):
        try:
            self.batch_size = self.config["batch_size"]
            # Determine layer sizes, add final target layer
            cur_layer_sizes = list(np.linspace(new_config['drp_first_layer_size'],
                                               new_config['drp_last_layer_size'],
                                               new_config['drp_num_layers']).astype(int))

            # Recreate model with new configuration =========
            cur_layer_sizes.append(1)
            model_args = [cur_layer_sizes, [0] * len(self.encoders),
                          self.encoder_train,
                          ['prelu'] * (new_config['drp_num_layers'] + 1) + ['relu'],
                          [True] * (new_config['drp_num_layers'] + 1) + [False],
                          [0.0] + [0.05] * new_config['drp_num_layers'] + [0.0]]
            for encoder in self.encoders:
                model_args.append(encoder)
            self.cur_model = DrugResponsePredictor(*model_args)

            self.cur_model = self.cur_model.float()
            self.cur_model = self.cur_model.cuda()
            self.criterion = nn.L1Loss().cuda()
            self.learning_rate = new_config['lr']
            # self.optimizer = optim.Adam(self.cur_model.parameters(), lr=new_config["lr"])

            # Reset data loaders ==============
            # if new_config['bottleneck'] is True:
            #     self.cur_cv_folds = self.bottleneck_cv_folds
            #     self.cur_train_data = self.bottleneck_train_data
            #
            # else:
            #     self.cur_cv_folds = self.cv_folds
            #     self.cur_train_data = self.train_data

            return True
        except:
            # If reset_config fails, catch exception and force Ray to make a new actor
            return False

    def step(self):
        self.timestep += 1
        start_time = time.time()

        self.avg_train_losses, \
        self.avg_valid_losses, \
        self.max_final_epoch = cross_validate(train_data=self.cur_train_data,
                                              train_function=drp_train,
                                              cv_folds=self.cur_cv_folds,
                                              batch_size=self.batch_size,
                                              cur_model=self.cur_model,
                                              criterion=self.criterion,
                                              max_epochs=self.max_epochs,
                                              learning_rate=self.learning_rate,
                                              NUM_WORKERS=NUM_WORKERS,
                                              verbose=True)

        duration = time.time() - start_time

        return {"avg_cv_train_loss": self.avg_train_losses,
                "avg_cv_valid_loss": self.avg_valid_losses,
                "max_final_epoch": self.max_final_epoch,
                "num_samples": len(self.cur_cv_folds[0][0]),
                "time_this_iter_s": duration}

    def save_checkpoint(self, tmp_checkpoint_dir):
        path = os.path.join(tmp_checkpoint_dir, "checkpoint")
        torch.save({"model_state_dict": self.cur_model.state_dict(),
                    # "optimizer_state_dict": self.optimizer.state_dict(),
                    "timestep": self.timestep}, path)

        return path

    def load_checkpoint(self, checkpoint):
        cur_checkpoint = torch.load(checkpoint)
        self.cur_model.load_state_dict(cur_checkpoint["model_state_dict"])
        # self.optimizer.load_state_dict(cur_checkpoint["optimizer_state_dict"])
        self.timestep = cur_checkpoint["timestep"]


class FullModelTrainable(Trainable):
    """
    This class implements the scenario where the entire model architecture is optimized, including
    both the selected Omic encoder modules as well as the DRP module. (excluding the drug module)
    """

    def setup(self, config, train_file: str = None, data_types: str = None, bottleneck: bool = None,
              n_folds: int = None, max_epochs: int = None, encoder_train: bool = None, cv_subset_type: str = None,
              stratify: bool = None, checkpoint_dir: str = "/.mounts/labs/steinlab/scratch/ftaj/"):
        self.data_dir = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/"
        # self.data_dir = "/u/ftaj/anaconda3/envs/Python/Data/DRP_Training_Data/"
        # self.data_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
        self.train_file = train_file
        self.data_types = data_types
        self.bottleneck = bottleneck
        self.bottleneck_path = "bottleneck_keys.csv"
        self.bottleneck_keys = None
        self.timestep = 0
        self.config = config
        self.n_folds = n_folds
        self.max_epochs = max_epochs
        self.encoder_train = encoder_train
        self.cv_subset_type = cv_subset_type
        self.stratify = stratify

        if self.bottleneck is True:
            try:
                self.bottleneck_keys = pd.read_csv(self.data_dir + self.bottleneck_path)['keys'].to_list()
            except:
                exit("Could not read bottleneck file")

        self.batch_size = self.config["batch_size"]
        # Since args.all_combos will be false, the main_prep function will only yield the desired combo
        self.cur_modules = self.data_types.split('_')
        assert "drug" in self.cur_modules, "Drug data must be provided for drug-response prediction (training or testing)"
        assert len(self.cur_modules) > 1, "Data types to be used must be indicated by: mut, cnv, exp, prot and drug"
        # Load all data types; this helps with instantiating auto-encoders based on correct input size
        all_data_list, _, self.key_columns, \
        self.gpu_locs = drp_load_datatypes(train_file=self.train_file,
                                           module_list=['drug', 'mut', 'cnv', 'exp', 'prot'],
                                           PATH=self.data_dir,
                                           file_name_dict=file_name_dict,
                                           load_autoencoders=False,
                                           device='gpu')
        self.cur_data_list = [all_data_list[0]]
        if 'mut' in self.cur_modules:
            self.cur_data_list.append(all_data_list[1])
        if 'cnv' in self.cur_modules:
            self.cur_data_list.append(all_data_list[2])
        if 'exp' in self.cur_modules:
            self.cur_data_list.append(all_data_list[3])
        if 'prot' in self.cur_modules:
            self.cur_data_list.append(all_data_list[4])

        self.drug_width = all_data_list[0].width()
        self.mut_width = all_data_list[1].width()
        self.cnv_width = all_data_list[2].width()
        self.exp_width = all_data_list[3].width()
        self.prot_width = all_data_list[4].width()
        del all_data_list

        # Instantiate auto-encoders based on the given cur_modules and data input sizes
        cur_autoencoders = initialize_autoencoders(cur_modules=self.cur_modules,
                                                   cur_config=config,
                                                   drug_width=self.drug_width,
                                                   mut_width=self.mut_width,
                                                   cnv_width=self.cnv_width,
                                                   exp_width=self.exp_width,
                                                   prot_width=self.prot_width)
        # Extract encoders from auto-encoders
        subset_encoders = [ExtractEncoder(autoencoder) for autoencoder in cur_autoencoders]
        del cur_autoencoders

        # Determine layer sizes, add final target layer
        cur_layer_sizes = list(np.linspace(config['drp_first_layer_size'],
                                           config['drp_last_layer_size'],
                                           config['drp_num_layers']).astype(int))
        cur_layer_sizes.append(1)
        # Create variable length argument set for DRP model creation, depending on number of given encoders
        model_args = [cur_layer_sizes, [0] * len(subset_encoders),
                      self.encoder_train,
                      ['prelu'] * (config['drp_num_layers'] + 1) + ['relu'],
                      [True] * (config['drp_num_layers'] + 1) + [False],
                      [0.0] + [0.05] * config['drp_num_layers'] + [0.0]]
        for encoder in subset_encoders:
            model_args.append(encoder)
        self.cur_model = DrugResponsePredictor(*model_args)

        # NOTE: Must move model to GPU before initializing optimizers!!!
        self.cur_model = self.cur_model.float()
        self.cur_model = self.cur_model.cuda()
        self.criterion = nn.L1Loss().cuda()
        # NOTE: optimizer should be defined immediately after a model is copied/moved/altered etc.
        # Not doing so will point the optimizer to a zombie model!
        # self.optimizer = optim.Adam(self.cur_model.parameters(), lr=config["lr"])
        self.learning_rate = config['lr']

        # Then the function will automatically make "new" indices and return them for saving in the checkpoint file
        # NOTE: The seed is the same, so the same subset is selected every time
        if self.bottleneck is True:
            self.cur_train_data, \
            self.cur_cv_folds = drp_create_datasets(self.cur_data_list,
                                                    self.key_columns,
                                                    n_folds=self.n_folds,
                                                    drug_index=0,
                                                    drug_dr_column="area_above_curve",
                                                    class_column_name="primary_disease",
                                                    subset_type=self.cv_subset_type,
                                                    stratify=self.stratify,
                                                    bottleneck_keys=self.bottleneck_keys,
                                                    to_gpu=True)
        else:
            self.cur_train_data, \
            self.cur_cv_folds = drp_create_datasets(self.cur_data_list,
                                                    self.key_columns,
                                                    n_folds=self.n_folds,
                                                    drug_index=0,
                                                    drug_dr_column="area_above_curve",
                                                    class_column_name="primary_disease",
                                                    subset_type=self.cv_subset_type,
                                                    stratify=self.stratify,
                                                    test_drug_data=None, to_gpu=True)

        # if config['bottleneck'] is True:
        #     # Get train and validation indices for the current fold
        #     self.cur_cv_folds = self.bottleneck_cv_folds
        #     self.cur_train_data = self.bottleneck_train_data
        #
        # else:
        #     self.cur_cv_folds = self.cv_folds
        #     self.cur_train_data = self.train_data

    def reset_config(self, new_config):
        try:
            # Reset model with new_config ===================
            cur_autoencoders = initialize_autoencoders(cur_modules=self.cur_modules,
                                                       cur_config=new_config,
                                                       drug_width=self.drug_width,
                                                       mut_width=self.mut_width,
                                                       cnv_width=self.cnv_width,
                                                       exp_width=self.exp_width,
                                                       prot_width=self.prot_width)

            subset_encoders = [ExtractEncoder(autoencoder) for autoencoder in cur_autoencoders]
            del cur_autoencoders
            # Determine layer sizes, add final target layer
            cur_layer_sizes = list(np.linspace(new_config['drp_first_layer_size'],
                                               new_config['drp_last_layer_size'],
                                               new_config['drp_num_layers']).astype(int))
            cur_layer_sizes.append(1)
            # Create variable length argument set for DRP model creation, depending on number of given encoders
            model_args = [cur_layer_sizes, [0] * len(subset_encoders),
                          self.encoder_train,
                          ['prelu'] * (new_config['drp_num_layers'] + 1) + ['relu'],
                          [True] * (new_config['drp_num_layers'] + 1) + [False],
                          [0.0] + [0.05] * new_config['drp_num_layers'] + [0.0]]
            for encoder in subset_encoders:
                model_args.append(encoder)
            self.cur_model = DrugResponsePredictor(*model_args)
            # NOTE: Must move model to GPU before initializing optimizers!!!
            self.cur_model = self.cur_model.float()
            self.cur_model = self.cur_model.cuda()
            self.criterion = nn.L1Loss().cuda()
            # self.optimizer = optim.Adam(self.cur_model.parameters(), lr=new_config["lr"])
            self.learning_rate = new_config['lr']

            # Reset data loaders =========================
            # if new_config['bottleneck'] is True:
            #     self.cur_cv_folds = self.bottleneck_cv_folds
            #     self.cur_train_data = self.bottleneck_train_data
            #
            # else:
            #     self.cur_cv_folds = self.cv_folds
            #     self.cur_train_data = self.train_data

            return True
        except:
            return False

    def step(self):
        self.timestep += 1
        start_time = time.time()

        self.avg_train_losses, \
        self.avg_valid_losses, \
        self.max_final_epoch = cross_validate(train_data=self.cur_train_data,
                                              train_function=drp_train,
                                              cv_folds=self.cur_cv_folds,
                                              batch_size=self.batch_size,
                                              cur_model=self.cur_model,
                                              criterion=self.criterion,
                                              max_epochs=self.max_epochs,
                                              learning_rate=self.learning_rate,
                                              NUM_WORKERS=NUM_WORKERS,
                                              theoretical_loss=True,
                                              verbose=True)

        duration = time.time() - start_time

        return {"avg_cv_train_loss": self.avg_train_losses,
                "avg_cv_valid_loss": self.avg_valid_losses,
                "max_final_epoch": self.max_final_epoch,
                "num_samples": len(self.cur_cv_folds[0][0]),
                "time_this_iter_s": duration}

    def save_checkpoint(self, tmp_checkpoint_dir):
        # print("save_checkpoint dir:", tmp_checkpoint_dir)
        path = os.path.join(tmp_checkpoint_dir, "checkpoint")
        torch.save({"model_state_dict": self.cur_model.state_dict(),
                    # "optimizer_state_dict": self.optimizer.state_dict(),
                    "timestep": self.timestep}, path)

        return path

    def load_checkpoint(self, checkpoint):
        # print("load_checkpoint dir:", checkpoint)
        cur_checkpoint = torch.load(checkpoint)
        self.cur_model.load_state_dict(cur_checkpoint["model_state_dict"])
        # self.optimizer.load_state_dict(cur_checkpoint["optimizer_state_dict"])
        self.timestep = cur_checkpoint["timestep"]
