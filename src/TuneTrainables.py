import os
import time
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ray.tune import Trainable
from torch.utils import data
from torch_geometric.nn import AttentiveFP

from DRPPreparation import drp_create_datasets, autoencoder_create_datasets, drp_load_datatypes, \
    create_cv_folds, change_ae_input_size
from DataImportModules import OmicData, MorganData
from Models import DNNAutoEncoder, MultiHeadCNNAutoEncoder, DrugResponsePredictor, LMF, LMFTest
from ModuleLoader import ExtractEncoder
from TrainFunctions import morgan_train, omic_train, drp_train, cross_validate, gnn_drp_train
from file_names import file_name_dict
from loss_functions import WeightedRMSELoss, RMSLELoss

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


def initialize_autoencoders(cur_modules: [str], cur_config: Dict, merge_method: str,
                            drug_width: int, mut_width: int, cnv_width: int, exp_width: int, prot_width: int,
                            mirna_width: int, hist_width: int, rppa_width: int, metab_width: int):
    cur_autoencoders = []
    if 'drug' in cur_modules:
        drug_model = DNNAutoEncoder(input_dim=drug_width,
                                    first_layer_size=cur_config['morgan_first_layer_size'],
                                    code_layer_size=cur_config[
                                        'morgan_code_layer_size'] if merge_method == 'concat' else
                                    cur_config['global_code_size'],
                                    num_layers=cur_config['morgan_num_layers'],
                                    act_fun_list='prelu',
                                    batchnorm_list=[True] * (cur_config['morgan_num_layers'] - 1) + [False],
                                    dropout_list=[0.1] + [0.2] * (cur_config['morgan_num_layers'] - 2) + [0.0],
                                    name="drug")
        cur_autoencoders.append(drug_model)

    if 'mut' in cur_modules:
        mut_model = DNNAutoEncoder(input_dim=mut_width,
                                   first_layer_size=cur_config['mut_first_layer_size'],
                                   code_layer_size=cur_config['mut_code_layer_size'] if merge_method == 'concat' else
                                   cur_config['global_code_size'],
                                   num_layers=cur_config['mut_num_layers'],
                                   act_fun_list='prelu',
                                   batchnorm_list=[True] * (cur_config['mut_num_layers'] - 1) + [False],
                                   dropout_list=[0.1] + [0.2] * (cur_config['mut_num_layers'] - 2) + [0.0],
                                   name="mut")
        cur_autoencoders.append(mut_model)
    if 'cnv' in cur_modules:
        cnv_model = DNNAutoEncoder(input_dim=cnv_width,
                                   first_layer_size=cur_config['cnv_first_layer_size'],
                                   code_layer_size=cur_config['cnv_code_layer_size'] if merge_method == 'concat' else
                                   cur_config['global_code_size'],
                                   num_layers=cur_config['cnv_num_layers'],
                                   act_fun_list='prelu',
                                   batchnorm_list=[True] * (cur_config['cnv_num_layers'] - 1) + [False],
                                   dropout_list=[0.1] + [0.2] * (cur_config['cnv_num_layers'] - 2) + [0.0],
                                   name="cnv")
        cur_autoencoders.append(cnv_model)
    if 'exp' in cur_modules:
        exp_model = DNNAutoEncoder(input_dim=exp_width,
                                   first_layer_size=cur_config['exp_first_layer_size'],
                                   code_layer_size=cur_config['exp_code_layer_size'] if merge_method == 'concat' else
                                   cur_config['global_code_size'],
                                   num_layers=cur_config['exp_num_layers'],
                                   act_fun_list='prelu',
                                   batchnorm_list=[True] * (cur_config['exp_num_layers'] - 1) + [False],
                                   dropout_list=[0.1] + [0.2] * (cur_config['exp_num_layers'] - 2) + [0.0],
                                   name="exp")
        cur_autoencoders.append(exp_model)
    if 'prot' in cur_modules:
        prot_model = DNNAutoEncoder(input_dim=prot_width,
                                    first_layer_size=cur_config['prot_first_layer_size'],
                                    code_layer_size=cur_config['prot_code_layer_size'] if merge_method == 'concat' else
                                    cur_config['global_code_size'],
                                    num_layers=cur_config['prot_num_layers'],
                                    # Linear activation ensures that negative numbers can be predicted
                                    act_fun_list=['none'] + ['prelu'] * (cur_config['prot_num_layers'] - 1),
                                    batchnorm_list=[True] * (cur_config['prot_num_layers'] - 1) + [False],
                                    dropout_list=[0.1] + [0.2] * (cur_config['prot_num_layers'] - 2) + [0.0],
                                    name="prot")
        cur_autoencoders.append(prot_model)

    if 'mirna' in cur_modules:
        mirna_model = DNNAutoEncoder(input_dim=mirna_width,
                                     first_layer_size=cur_config['mirna_first_layer_size'],
                                     code_layer_size=cur_config[
                                         'mirna_code_layer_size'] if merge_method == 'concat' else
                                     cur_config['global_code_size'],
                                     num_layers=cur_config['mirna_num_layers'],
                                     act_fun_list=['prelu'] * (cur_config['mirna_num_layers']),
                                     batchnorm_list=[True] * (cur_config['mirna_num_layers'] - 1) + [False],
                                     dropout_list=[0.1] + [0.2] * (cur_config['mirna_num_layers'] - 2) + [0.0],
                                     name="mirna")
        cur_autoencoders.append(mirna_model)

    if 'hist' in cur_modules:
        hist_model = DNNAutoEncoder(input_dim=hist_width,
                                    first_layer_size=cur_config['hist_first_layer_size'],
                                    code_layer_size=cur_config['hist_code_layer_size'] if merge_method == 'concat' else
                                    cur_config['global_code_size'],
                                    num_layers=cur_config['hist_num_layers'],
                                    # Linear activation ensures that negative numbers can be predicted
                                    act_fun_list=['none'] + ['prelu'] * (cur_config['hist_num_layers'] - 1),
                                    batchnorm_list=[True] * (cur_config['hist_num_layers'] - 1) + [False],
                                    dropout_list=[0.1] + [0.2] * (cur_config['hist_num_layers'] - 2) + [0.0],
                                    name="hist")
        cur_autoencoders.append(hist_model)

    if 'rppa' in cur_modules:
        rppa_model = DNNAutoEncoder(input_dim=rppa_width,
                                    first_layer_size=cur_config['rppa_first_layer_size'],
                                    code_layer_size=cur_config['rppa_code_layer_size'] if merge_method == 'concat' else
                                    cur_config['global_code_size'],
                                    num_layers=cur_config['rppa_num_layers'],
                                    # Linear activation ensures that negative numbers can be predicted
                                    act_fun_list=['none'] + ['prelu'] * (cur_config['rppa_num_layers'] - 1),
                                    batchnorm_list=[True] * (cur_config['rppa_num_layers'] - 1) + [False],
                                    dropout_list=[0.1] + [0.2] * (cur_config['rppa_num_layers'] - 2) + [0.0],
                                    name="rppa")
        cur_autoencoders.append(rppa_model)

    if 'metab' in cur_modules:
        metab_model = DNNAutoEncoder(input_dim=metab_width,
                                     first_layer_size=cur_config['metab_first_layer_size'],
                                     code_layer_size=cur_config[
                                         'metab_code_layer_size'] if merge_method == 'concat' else
                                     cur_config['global_code_size'],
                                     num_layers=cur_config['metab_num_layers'],
                                     # Linear activation ensures that negative numbers can be predicted
                                     act_fun_list=['prelu'] * (cur_config['metab_num_layers']),
                                     batchnorm_list=[True] * (cur_config['metab_num_layers'] - 1) + [False],
                                     dropout_list=[0.1] + [0.2] * (cur_config['metab_num_layers'] - 2) + [0.0],
                                     name="metab")
        cur_autoencoders.append(metab_model)

    return cur_autoencoders


class MorganTrainable(Trainable):
    def setup(self, config, checkpoint_dir="/.mounts/labs/steinlab/scratch/ftaj/",
              loss_type: str = "mae", morgan_width: int = 512, model_type: str = "dnn"):
        self.data_dir = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/"
        # self.data_dir = "/u/ftaj/anaconda3/envs/Python/Data/DRP_Training_Data/"
        # self.data_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
        self.timestep = 0
        self.model_type = model_type
        self.morgan_width = morgan_width
        self.loss_type = loss_type
        # Prepare data loaders =========
        morgan_file_name = "ChEMBL_Morgan_" + str(self.morgan_width) + ".pkl"

        self.train_data = MorganData(path=self.data_dir, morgan_file_name=morgan_file_name,
                                     model_type=self.model_type)
        assert self.train_data.width() == morgan_width, "The width of the given training set doesn't match the expected Morgan width."

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
            self.cur_model = DNNAutoEncoder(input_dim=self.morgan_width,
                                            first_layer_size=self.config["first_layer_size"],
                                            code_layer_size=self.config["code_layer_size"],
                                            num_layers=self.config["num_layers"],
                                            batchnorm_list=all_batch_norm,
                                            act_fun_list=all_act_funcs,
                                            dropout_list=all_dropout)

        self.cur_model = self.cur_model.float()
        self.cur_model = self.cur_model.cuda()

        if self.loss_type == "mae":
            self.criterion = nn.L1Loss().cuda()
        elif self.loss_type == "mse":
            self.criterion = nn.MSELoss().cuda()
        elif self.loss_type == "rmsle":
            self.criterion = RMSLELoss().cuda()
        elif self.loss_type == "weighted_rmse" or self.loss_type == "rmse":
            self.criterion = WeightedRMSELoss().cuda()
        else:
            exit("Unknown loss function requested")

        self.learning_rate = config['lr']
        self.optimizer = optim.Adam(self.cur_model.parameters(), lr=self.config["lr"])

    def reset_config(self, new_config):
        try:
            # Reset model with new_config ==============
            if self.model_type == "cnn":
                all_act_funcs, all_batch_norm, all_dropout = get_layer_configs(self.config, model_type="cnn")
                self.cur_model = MultiHeadCNNAutoEncoder(
                    input_width=self.morgan_width,
                    num_branch=new_config["num_branch"],
                    stride=new_config["stride"],
                    batchnorm_list=all_batch_norm,
                    act_fun_list=all_act_funcs,
                    dropout_list=all_dropout
                )
            else:
                all_act_funcs, all_batch_norm, all_dropout = get_layer_configs(self.config, model_type="dnn")
                self.cur_model = DNNAutoEncoder(
                    input_dim=self.morgan_width,
                    first_layer_size=new_config["first_layer_size"],
                    code_layer_size=new_config["code_layer_size"],
                    num_layers=new_config["num_layers"],
                    batchnorm_list=all_batch_norm,
                    act_fun_list=all_act_funcs,
                    dropout_list=all_dropout
                )
            self.cur_model = self.cur_model.float()
            self.cur_model = self.cur_model.cuda()
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
              loss_type: str = "mae", omic_standardize: bool = True,
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
        self.loss_type = loss_type
        self.omic_standardize = omic_standardize

        # Prepare data loaders ==========
        file_name = file_name_dict[self.pretrain + "depmap_" + self.data_type + "_file_name"]
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
        # all_act_funcs, all_batch_norm, all_dropout = get_layer_configs(self.config, model_type="dnn")

        self.cur_model = DNNAutoEncoder(
            input_dim=self.train_data.width(),
            first_layer_size=self.config["first_layer_size"],
            code_layer_size=self.config["code_layer_size"],
            num_layers=self.config["num_layers"],
            batchnorm_list=[False] + [False] * (self.config["num_layers"] - 1),
            act_fun_list=['silu'] + ['silu'] * (self.config["num_layers"] - 1),
            dropout_list=[0.0] + [0.0] * (self.config["num_layers"] - 1)
        )

        self.cur_model = self.cur_model.float()
        self.cur_model = self.cur_model.cuda()
        if self.loss_type == "mae":
            self.criterion = nn.L1Loss().cuda()
        elif self.loss_type == "mse":
            self.criterion = nn.MSELoss().cuda()
        elif self.loss_type == "rmsle":
            self.criterion = RMSLELoss().cuda()
        elif self.loss_type == "weighted_rmse" or self.loss_type == "rmse":
            self.criterion = WeightedRMSELoss().cuda()
        else:
            exit("Unknown loss function requested")
        self.learning_rate = config['lr']
        # self.optimizer = optim.Adam(self.cur_model.parameters(), lr=self.config["lr"])

    def reset_config(self, new_config):
        try:
            self.batch_size = new_config['batch_size']
            # Reset model with new_config ===========
            # all_act_funcs, all_batch_norm, all_dropout = get_layer_configs(new_config, model_type="dnn")

            self.cur_model = DNNAutoEncoder(
                input_dim=self.train_data.width(),
                first_layer_size=new_config["first_layer_size"],
                code_layer_size=new_config["code_layer_size"],
                num_layers=new_config["num_layers"],
                batchnorm_list=[False] + [False] * (self.config["num_layers"] - 1),
                act_fun_list=['silu'] + ['silu'] * (self.config["num_layers"] - 1),
                dropout_list=[0.0] + [0.0] * (self.config["num_layers"] - 1)
            )
            self.cur_model = self.cur_model.float()
            self.cur_model = self.cur_model.cuda()
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
        self.avg_untrained_losses, \
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
                                              patience=10,
                                              delta=0.01,
                                              omic_standardize=self.omic_standardize,
                                              verbose=True)

        duration = time.time() - start_time

        return {"avg_cv_train_loss": self.avg_train_losses,
                "avg_cv_valid_loss": self.avg_valid_losses,
                "avg_cv_untrained_loss": self.avg_untrained_losses,
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
              cv_subset_type: str = None, stratify: bool = None, random_morgan: bool = False,
              merge_method: str = "concat",
              loss_type: str = 'mae', one_hot_drugs: bool = False, transform: str = None,
              min_dr_target: float = None, max_dr_target: float = None, gnn_drug: bool = False,
              omic_standardize: bool = False, name_tag: str = None,
              to_gpu: bool = True, test_mode: bool = False,
              data_dir: str = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/",
              checkpoint_dir: str = "/.mounts/labs/steinlab/scratch/ftaj/"):
        self.data_dir = data_dir
        self.to_gpu = to_gpu
        self.cur_config = config
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
        self.random_morgan = random_morgan
        self.loss_type = loss_type
        self.merge_method = merge_method
        self.batch_size = config["batch_size"]
        self.one_hot_drugs = one_hot_drugs
        self.transform = transform
        self.min_dr_target = min_dr_target
        self.max_dr_target = max_dr_target
        self.gnn_drug = gnn_drug
        self.omic_standardize = omic_standardize
        self.test_mode = test_mode
        self.name_tag = name_tag

        if self.bottleneck is True:
            try:
                self.bottleneck_keys = pd.read_csv(self.data_dir + self.bottleneck_path)['keys'].to_list()
            except:
                exit("Could not read bottleneck file")

        self.cur_modules = self.data_types.split('_')
        # assert "drug" in self.cur_modules, "Drug data must be provided for drug-response prediction (training or testing)"
        # assert len(cur_modules) > 1, "Omic Data types to be used must be indicated by: mut, cnv, exp, prot and drug"

        # Must maintain a constant encoder size for summing
        if self.merge_method == "sum":
            global_code_size = config['global_code_size']
        else:
            global_code_size = None

        # Load data and auto-encoders
        all_data_dict, autoencoder_list, self.key_columns, \
        self.gpu_locs = drp_load_datatypes(train_file=self.train_file,
                                           module_list=self.cur_modules,
                                           PATH=self.data_dir,
                                           file_name_dict=file_name_dict,
                                           load_autoencoders=True,
                                           _pretrain=self.pretrain,
                                           global_code_size=global_code_size,
                                           random_morgan=self.random_morgan,
                                           transform=transform,
                                           # one-hot drugs cannot be pretrained/embedded,
                                           # so a new auto-encoder will be instantiated
                                           device='cpu',
                                           verbose=True)

        # Add drug data TODO What if someone doesn't wanna use morgan data? some other drug data? or nothing...
        if self.gnn_drug is False:
            self.cur_data_list: List = [all_data_dict['morgan']]
        else:
            self.cur_data_list: List = [all_data_dict['gnndrug']]

        self.mut_width = None
        self.cnv_width = None
        self.exp_width = None
        self.prot_width = None
        self.mirna_width = None
        self.hist_width = None
        self.rppa_width = None
        self.metab_width = None

        if 'mut' in self.cur_modules:
            self.cur_data_list.append(all_data_dict['mut'])
            self.mut_width = all_data_dict['mut'].width()
        if 'cnv' in self.cur_modules:
            self.cur_data_list.append(all_data_dict['cnv'])
            self.cnv_width = all_data_dict['cnv'].width()
        if 'exp' in self.cur_modules:
            self.cur_data_list.append(all_data_dict['exp'])
            self.exp_width = all_data_dict['exp'].width()
        if 'prot' in self.cur_modules:
            self.cur_data_list.append(all_data_dict['prot'])
            self.prot_width = all_data_dict['prot'].width()

        if 'mirna' in self.cur_modules:
            self.cur_data_list.append(all_data_dict['mirna'])
            self.mirna_width = all_data_dict['mirna'].width()
        if 'hist' in self.cur_modules:
            self.cur_data_list.append(all_data_dict['hist'])
            self.hist_width = all_data_dict['hist'].width()
        if 'rppa' in self.cur_modules:
            self.cur_data_list.append(all_data_dict['rppa'])
            self.rppa_width = all_data_dict['rppa'].width()
        if 'metab' in self.cur_modules:
            self.cur_data_list.append(all_data_dict['metab'])
            self.metab_width = all_data_dict['metab'].width()

        del all_data_dict

        if 'weighted' in self.loss_type:
            lds = True
        else:
            lds = False
            # loss must be weighted, PairData class must return smoothed label weights
        # Then the function will automatically make "new" indices and return them for saving in the checkpoint file
        # NOTE: The seed is the same, so the same subset is selected every time (!)
        # Load bottleneck and full data once, choose later on
        if self.bottleneck is True:
            self.cur_train_data, \
            self.cur_cv_folds = drp_create_datasets(data_list=self.cur_data_list,
                                                    key_columns=self.key_columns,
                                                    n_folds=self.n_folds,
                                                    drug_index=0,
                                                    drug_dr_column="area_above_curve",
                                                    class_column_name="primary_disease",
                                                    subset_type=self.cv_subset_type,
                                                    stratify=self.stratify,
                                                    bottleneck_keys=self.bottleneck_keys,
                                                    to_gpu=self.to_gpu,
                                                    one_hot_drugs=self.one_hot_drugs,
                                                    dr_sub_min_target=self.min_dr_target,
                                                    dr_sub_max_target=self.max_dr_target,
                                                    lds=lds,
                                                    gnn_mode=self.gnn_drug,
                                                    verbose=False)
        else:
            self.cur_train_data, \
            self.cur_cv_folds = drp_create_datasets(data_list=self.cur_data_list,
                                                    key_columns=self.key_columns,
                                                    n_folds=self.n_folds,
                                                    drug_index=0,
                                                    drug_dr_column="area_above_curve",
                                                    class_column_name="primary_disease",
                                                    subset_type=self.cv_subset_type,
                                                    stratify=self.stratify,
                                                    test_drug_data=None,
                                                    to_gpu=self.to_gpu,
                                                    one_hot_drugs=self.one_hot_drugs,
                                                    dr_sub_min_target=self.min_dr_target,
                                                    dr_sub_max_target=self.max_dr_target,
                                                    lds=lds,
                                                    gnn_mode=self.gnn_drug,
                                                    verbose=False)

        if self.gnn_drug is False:

            # Drug input with can depend on whether one-hot vectors or fingerprints are used! Must update here
            # TODO drug_input_width attr doesn't exists anymore, must update
            self.drug_width = self.cur_train_data.drug_input_width

            if self.one_hot_drugs is True:
                print("Modifying Drug AE input size to", self.drug_width)
                # one_hot_drugs cannot be pre-trained on e.g. ChEMBL so must instantiate a new auto-encoder
                autoencoder_list[0] = change_ae_input_size(ae=autoencoder_list[0], input_size=self.drug_width,
                                                           name="drug")

            # Extract encoders from loaded auto-encoders
            self.encoders = [ExtractEncoder(autoencoder) for autoencoder in autoencoder_list]

        else:
            self.gnn_out_channels = config['gnn_out_channels'] if global_code_size is None else global_code_size
            # gnndrug does not have an autoencoder yet, so ignore first element
            self.encoders = [ExtractEncoder(autoencoder) for autoencoder in autoencoder_list[1:]]
            # Add AttentiveFP model for drug processing
            self.attentive_fp = AttentiveFP(in_channels=37,
                                            hidden_channels=config['gnn_hidden_channels'],
                                            out_channels=self.gnn_out_channels,
                                            edge_dim=10,
                                            num_layers=config['gnn_num_layers'],
                                            num_timesteps=config['gnn_num_timesteps'],
                                            dropout=config['gnn_dropout'])
            self.encoders = [self.attentive_fp] + self.encoders

        # Determine layer sizes, add final target layer
        cur_layer_sizes = list(np.linspace(config['drp_first_layer_size'],
                                           config['drp_last_layer_size'],
                                           config['drp_num_layers']).astype(int))
        cur_layer_sizes.append(1)
        # Note that the order of arguments must be respected here!!!
        # print("len(self.subset_encoders):", len(self.subset_encoders))

        if self.merge_method != "lmf":
            # Create variable length argument set for DRP model creation, depending on number of given encoders
            model_args = [
                cur_layer_sizes,
                self.encoder_train,  # encoder_requirees_grad
                [self.gnn_drug, self.gnn_out_channels] if self.gnn_drug is True else
                [self.gnn_drug, None],  # gnn_info
                # TODO: last layer must use a sigmoid to ensure values are between 0 & 1
                ['silu'] * (config['drp_num_layers']) + ['sigmoid'],  # act_fun_list
                [False] * (config['drp_num_layers']) + [False],  # batchnorm_list
                [0.0] * config['drp_num_layers'] + [0.0],  # dropout_list
                self.merge_method
            ]  # merge_method
            for encoder in self.encoders:
                model_args.append(encoder)
            self.cur_model = DrugResponsePredictor(*model_args)

        else:
            # Low-rank Multi-modal fusion
            model_args = [
                cur_layer_sizes,
                self.encoder_train,
                [self.gnn_drug, self.gnn_out_channels] if self.gnn_drug is True else
                [self.gnn_drug, None],  # gnn_info
                config['lmf_output_dim'],  # output dim
                config['lmf_rank'],  # rank
                ['silu'] * (config['drp_num_layers']) + ['sigmoid'],  # act_fun_list
                [False] * (config['drp_num_layers']) + [False],  # batchnorm_list
                [0.0] * config['drp_num_layers'] + [0.0],  # dropout_list
            ]
            for encoder in self.encoders:
                model_args.append(encoder)
            if self.test_mode:
                self.cur_model = LMFTest(*model_args)
            else:
                self.cur_model = LMF(*model_args)

        # NOTE: Must move model to GPU before initializing optimizers!!!
        self.cur_model = self.cur_model.float()

        if self.loss_type == "mae":
            self.criterion = nn.L1Loss()
        elif self.loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss_type == "rmsle":
            self.criterion = RMSLELoss()
        elif self.loss_type == "weighted_rmse" or self.loss_type == "rmse":
            # WeightedRMSELoss can also handle equally weighted losses
            self.criterion = WeightedRMSELoss()
        else:
            exit("Unknown loss function requested")

        if self.to_gpu is True:
            # self.cur_model = self.cur_model.cuda()
            self.criterion = self.criterion.cuda()

        # NOTE: optimizer should be defined immediately after a model is copied/moved/altered etc.
        # Not doing so will point the optimizer to a zombie model!
        # self.optimizer = optim.Adam(self.cur_model.parameters(), lr=self.config["lr"])
        self.learning_rate = config['lr']

        # if config['bottleneck'] is True:
        #     # Get train and validation indices for the current fold
        #     self.cur_cv_folds = self.bottleneck_cv_folds
        #     self.cur_train_data = self.bottleneck_train_data
        #
        # else:
        #     self.cur_cv_folds = self.cv_folds
        #     self.cur_train_data = self.train_data

    def reset_config(self, new_config):
        self.cur_config = new_config
        if self.merge_method == "sum":
            global_code_size = new_config['global_code_size']
        else:
            global_code_size = None

        self.gnn_out_channels = new_config['gnn_out_channels'] if global_code_size is None else global_code_size

        try:
            if self.gnn_drug is True:
                self.gnn_drug = AttentiveFP(in_channels=37,
                                            hidden_channels=new_config['gnn_hidden_channels'],
                                            out_channels=self.gnn_out_channels,
                                            edge_dim=10,
                                            num_layers=new_config['gnn_num_layers'],
                                            num_timesteps=new_config['gnn_num_timesteps'],
                                            dropout=new_config['gnn_dropout'])
                # Update/replace gnn drug encoder with the new configuration
                self.encoders = [self.gnn_drug] + self.encoders[1:]

            self.batch_size = new_config["batch_size"]
            # Determine layer sizes, add final target layer
            cur_layer_sizes = list(np.linspace(new_config['drp_first_layer_size'],
                                               new_config['drp_last_layer_size'],
                                               new_config['drp_num_layers']).astype(int))

            # Recreate model with new configuration =========
            cur_layer_sizes.append(1)
            if self.merge_method != "lmf":
                model_args = [cur_layer_sizes, [0] * len(self.encoders),
                              self.encoder_train,
                              [self.gnn_drug, self.gnn_out_channels] if self.gnn_drug is True else
                              [self.gnn_drug, None],  # gnn_info
                              ['silu'] * (new_config['drp_num_layers']) + ['sigmoid'],
                              [False] * (new_config['drp_num_layers']) + [False],
                              [0.0] * new_config['drp_num_layers'] + [0.0],
                              self.merge_method]
                for encoder in self.encoders:
                    model_args.append(encoder)
                self.cur_model = DrugResponsePredictor(*model_args)

            else:
                # Low-rank Multi-modal fusion
                model_args = [
                    cur_layer_sizes,
                    self.encoder_train,
                    [self.gnn_drug, self.gnn_out_channels] if self.gnn_drug is True else
                    [self.gnn_drug, None],  # gnn_info
                    new_config['lmf_output_dim'],  # output dim
                    new_config['lmf_rank'],  # rank
                    ['silu'] * (new_config['drp_num_layers']) + ['sigmoid'],  # act_fun_list
                    [False] * (new_config['drp_num_layers']) + [False],  # batchnorm_list
                    [0.0] * new_config['drp_num_layers'] + [0.0],  # dropout_list
                ]
                for encoder in self.encoders:
                    model_args.append(encoder)
                self.cur_model = LMF(*model_args)

            self.cur_model = self.cur_model.float()
            if self.loss_type == "mae":
                self.criterion = nn.L1Loss()
            elif self.loss_type == "mse":
                self.criterion = nn.MSELoss()
            elif self.loss_type == "rmsle":
                self.criterion = RMSLELoss()
            # elif self.loss_type == "rmse":
            #     self.criterion = RMSELoss().cuda()
            elif self.loss_type == "weighted_rmse" or self.loss_type == "rmse":
                self.criterion = WeightedRMSELoss()

            else:
                exit("Unknown loss function requested")

            if self.to_gpu is True:
                # self.cur_model = self.cur_model.cuda()
                self.criterion = self.criterion.cuda()

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

        # cur_dict = {k: self.cur_config[k] for k in ('drp_first_layer_size', 'drp_last_layer_size', 'batch_size')}
        # epoch_save_folder = "/scratch/l/lstein/ftaj/EpochResults/" + str(cur_dict)
        epoch_save_folder = "/scratch/l/lstein/ftaj/EpochResults/CrossValidation/" + self.name_tag

        self.avg_train_losses, \
        self.avg_valid_losses, \
        self.avg_untrained_losses, \
        self.max_final_epoch = cross_validate(train_data=self.cur_train_data,
                                              train_function=drp_train if self.gnn_drug is False
                                              else gnn_drp_train,
                                              cv_folds=self.cur_cv_folds,
                                              batch_size=self.batch_size,
                                              cur_model=self.cur_model,
                                              criterion=self.criterion,
                                              max_epochs=self.max_epochs,
                                              learning_rate=self.learning_rate,
                                              theoretical_loss=False,
                                              NUM_WORKERS=NUM_WORKERS,
                                              verbose=True,
                                              patience=5,
                                              delta=0.01,
                                              omic_standardize=self.omic_standardize,
                                              save_epoch_results=False,
                                              save_cv_preds=False,
                                              epoch_save_folder=epoch_save_folder,
                                              to_gpu=self.to_gpu
                                              )

        duration = time.time() - start_time

        return {"avg_cv_train_loss": self.avg_train_losses,
                "avg_cv_valid_loss": self.avg_valid_losses,
                "avg_cv_untrained_loss": self.avg_untrained_losses,
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
              stratify: bool = None, random_morgan: bool = False, merge_method: str = "concat", loss_type: str = 'mae',
              one_hot_drugs: bool = False, transform: str = None, min_dr_target: float = None,
              max_dr_target: float = None, gnn_drug: bool = False,
              checkpoint_dir: str = "/.mounts/labs/steinlab/scratch/ftaj/"):
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
        self.merge_method = merge_method
        self.loss_type = loss_type
        self.one_hot_drugs = one_hot_drugs
        self.transform = transform
        self.min_dr_target = min_dr_target
        self.max_dr_target = max_dr_target
        self.gnn_drug = gnn_drug

        if self.bottleneck is True:
            try:
                self.bottleneck_keys = pd.read_csv(self.data_dir + self.bottleneck_path)['keys'].to_list()
            except:
                exit("Could not read bottleneck file")

        self.batch_size = self.config["batch_size"]
        # Since args.all_combos will be false, the main_prep function will only yield the desired combo
        self.cur_modules = self.data_types.split('_')
        assert "drug" in self.cur_modules, "Drug data must be provided for drug-response prediction (training or testing)"
        # assert len(self.cur_modules) > 1, "Data types to be used must be indicated by: mut, cnv, exp, prot and drug"
        # Load all data types; this helps with instantiating auto-encoders based on correct input size
        all_data_dict, _, self.key_columns, \
        self.gpu_locs = drp_load_datatypes(train_file=self.train_file,
                                           module_list=self.cur_modules,
                                           PATH=self.data_dir,
                                           file_name_dict=file_name_dict,
                                           load_autoencoders=False,
                                           random_morgan=random_morgan,
                                           transform=transform,
                                           device='gpu')

        # Add drug data TODO What if someone doesn't wanna use morgan data? some other drug data? or nothin...
        if self.gnn_drug is False:
            self.cur_data_list: List = [all_data_dict['morgan']]
        else:
            self.cur_data_list: List = [all_data_dict['gnndrug']]

        self.mut_width = None
        self.cnv_width = None
        self.exp_width = None
        self.prot_width = None
        self.mirna_width = None
        self.hist_width = None
        self.rppa_width = None
        self.metab_width = None

        if 'mut' in self.cur_modules:
            self.cur_data_list.append(all_data_dict['mut'])
            self.mut_width = all_data_dict['mut'].width()
        if 'cnv' in self.cur_modules:
            self.cur_data_list.append(all_data_dict['cnv'])
            self.cnv_width = all_data_dict['cnv'].width()
        if 'exp' in self.cur_modules:
            self.cur_data_list.append(all_data_dict['exp'])
            self.exp_width = all_data_dict['exp'].width()
        if 'prot' in self.cur_modules:
            self.cur_data_list.append(all_data_dict['prot'])
            self.prot_width = all_data_dict['prot'].width()

        if 'mirna' in self.cur_modules:
            self.cur_data_list.append(all_data_dict['mirna'])
            self.mirna_width = all_data_dict['mirna'].width()
        if 'hist' in self.cur_modules:
            self.cur_data_list.append(all_data_dict['hist'])
            self.hist_width = all_data_dict['hist'].width()
        if 'rppa' in self.cur_modules:
            self.cur_data_list.append(all_data_dict['rppa'])
            self.rppa_width = all_data_dict['rppa'].width()
        if 'metab' in self.cur_modules:
            self.cur_data_list.append(all_data_dict['metab'])
            self.metab_width = all_data_dict['metab'].width()

        del all_data_dict

        if 'weighted' in self.loss_type:
            lds = True
        else:
            lds = False

        # Then the function will automatically make "new" indices and return them for saving in the checkpoint file
        # NOTE: The seed is the same, so the same subset is selected every time
        if self.bottleneck is True:
            self.cur_train_data, \
            self.cur_cv_folds = drp_create_datasets(data_list=self.cur_data_list,
                                                    key_columns=self.key_columns,
                                                    n_folds=self.n_folds,
                                                    drug_index=0,
                                                    drug_dr_column="area_above_curve",
                                                    class_column_name="primary_disease",
                                                    subset_type=self.cv_subset_type,
                                                    stratify=self.stratify,
                                                    bottleneck_keys=self.bottleneck_keys,
                                                    one_hot_drugs=self.one_hot_drugs,
                                                    dr_sub_min_target=self.min_dr_target,
                                                    dr_sub_max_target=self.max_dr_target,
                                                    lds=lds,
                                                    gnn_mode=self.gnn_drug,
                                                    to_gpu=True)
        else:
            self.cur_train_data, \
            self.cur_cv_folds = drp_create_datasets(data_list=self.cur_data_list,
                                                    key_columns=self.key_columns,
                                                    n_folds=self.n_folds,
                                                    drug_index=0,
                                                    drug_dr_column="area_above_curve",
                                                    class_column_name="primary_disease",
                                                    subset_type=self.cv_subset_type,
                                                    stratify=self.stratify,
                                                    test_drug_data=None,
                                                    one_hot_drugs=self.one_hot_drugs,
                                                    dr_sub_min_target=self.min_dr_target,
                                                    dr_sub_max_target=self.max_dr_target,
                                                    lds=lds,
                                                    gnn_mode=self.gnn_drug,
                                                    to_gpu=True)

        if self.gnn_drug is False:
            # Drug input with can depend on whether one-hot vectors or fingerprints are used
            # TODO drug_input_width attr doesn't exists anymore, must update
            self.drug_width = self.cur_train_data.drug_input_width

        # Instantiate auto-encoders based on the given cur_modules and data input sizes
        print("Current auto-encoder modules are:", self.cur_modules)
        cur_autoencoders = initialize_autoencoders(cur_modules=self.cur_modules,
                                                   cur_config=config,
                                                   merge_method=self.merge_method,
                                                   drug_width=self.drug_width,
                                                   mut_width=self.mut_width,
                                                   cnv_width=self.cnv_width,
                                                   exp_width=self.exp_width,
                                                   prot_width=self.prot_width,
                                                   mirna_width=self.mirna_width,
                                                   hist_width=self.hist_width,
                                                   rppa_width=self.rppa_width,
                                                   metab_width=self.metab_width)
        # Extract encoders from auto-encoders
        self.encoders = [ExtractEncoder(autoencoder) for autoencoder in cur_autoencoders]

        if self.gnn_drug is True:
            # gnndrug does not have an autoencoder yet, must create Attentive FP model for drug processing
            self.gnn_drug = AttentiveFP(in_channels=37,
                                        hidden_channels=config['gnn_hidden_channels'],
                                        out_channels=config['gnn_out_channels'],
                                        edge_dim=10,
                                        num_layers=config['gnn_num_layers'],
                                        num_timesteps=config['gnn_num_timesteps'],
                                        dropout=config['gnn_dropout'])
            self.encoders = [self.gnn_drug] + self.encoders

        print("Current number of encoders are:", len(self.encoders))
        del cur_autoencoders

        # Determine layer sizes, add final target layer
        cur_layer_sizes = list(np.linspace(config['drp_first_layer_size'],
                                           config['drp_last_layer_size'],
                                           config['drp_num_layers']).astype(int))
        cur_layer_sizes.append(1)

        if self.merge_method != "lmf":
            # Create variable length argument set for DRP model creation, depending on number of given encoders
            model_args = [
                cur_layer_sizes,
                [0] * len(self.encoders),  # gpu_locs
                self.encoder_train,  # encoder_requirees_grad
                [self.gnn_drug, config['gnn_out_channels']],  # gnn_info
                # TODO: last layer must use a sigmoid to ensure values are between 0 & 1
                ['silu'] * (config['drp_num_layers'] + 1) + ['silu'],  # act_fun_list
                [True] * (config['drp_num_layers'] + 1) + [False],  # batchnorm_list
                [0.0] + [0.05] * config['drp_num_layers'] + [0.0],  # dropout_list
                self.merge_method
            ]  # merge_method
            for encoder in self.encoders:
                model_args.append(encoder)
            self.cur_model = DrugResponsePredictor(*model_args)

        else:
            # Low-rank Multi-modal fusion
            model_args = [
                cur_layer_sizes,
                self.encoder_train,
                [self.gnn_drug, config['gnn_out_channels']],  # gnn_info
                config['lmf_output_dim'],  # output dim
                config['lmf_rank'],  # rank
                ['silu'] * (config['drp_num_layers'] + 1) + ['silu'],  # act_fun_list
                [True] * (config['drp_num_layers'] + 1) + [False],  # batchnorm_list
                [0.0] + [0.05] * config['drp_num_layers'] + [0.0],  # dropout_list
            ]
            for encoder in self.encoders:
                model_args.append(encoder)
            self.cur_model = LMF(*model_args)

        # NOTE: Must move model to GPU before initializing optimizers!!!
        self.cur_model = self.cur_model.float()
        self.cur_model = self.cur_model.cuda()
        if self.loss_type == "mae":
            self.criterion = nn.L1Loss().cuda()
        elif self.loss_type == "mse":
            self.criterion = nn.MSELoss().cuda()
        elif self.loss_type == "rmsle":
            self.criterion = RMSLELoss().cuda()
        elif self.loss_type == "weighted_rmse" or self.loss_type == "rmse":
            self.criterion = WeightedRMSELoss().cuda()
        else:
            exit("Unknown loss function requested")
        # NOTE: optimizer should be defined immediately after a model is copied/moved/altered etc.
        # Not doing so will point the optimizer to a zombie model!
        # self.optimizer = optim.Adam(self.cur_model.parameters(), lr=config["lr"])
        self.learning_rate = config['lr']

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
                                                       merge_method=self.merge_method,
                                                       drug_width=self.drug_width,
                                                       mut_width=self.mut_width,
                                                       cnv_width=self.cnv_width,
                                                       exp_width=self.exp_width,
                                                       prot_width=self.prot_width,
                                                       mirna_width=self.mirna_width,
                                                       hist_width=self.hist_width,
                                                       rppa_width=self.rppa_width,
                                                       metab_width=self.metab_width)

            self.encoders = [ExtractEncoder(autoencoder) for autoencoder in cur_autoencoders]
            del cur_autoencoders

            if self.gnn_drug is True:
                self.gnn_drug = AttentiveFP(in_channels=37,
                                            hidden_channels=new_config['gnn_hidden_channels'],
                                            out_channels=new_config['gnn_out_channels'],
                                            edge_dim=10,
                                            num_layers=new_config['gnn_num_layers'],
                                            num_timesteps=new_config['gnn_num_timesteps'],
                                            dropout=new_config['gnn_dropout'])
                # Update/replace gnn drug encoder with the new configuration
                self.encoders = [self.gnn_drug] + self.encoders[1:]

            # Determine layer sizes, add final target layer
            cur_layer_sizes = list(np.linspace(new_config['drp_first_layer_size'],
                                               new_config['drp_last_layer_size'],
                                               new_config['drp_num_layers']).astype(int))
            cur_layer_sizes.append(1)
            # Create variable length argument set for DRP model creation, depending on number of given encoders
            if self.merge_method != "lmf":
                model_args = [cur_layer_sizes, [0] * len(self.encoders),
                              self.encoder_train,
                              [self.gnn_drug, new_config['gnn_out_channels']],  # gnn_info
                              ['silu'] * (new_config['drp_num_layers'] + 1) + ['silu'],
                              [True] * (new_config['drp_num_layers'] + 1) + [False],
                              [0.0] + [0.05] * new_config['drp_num_layers'] + [0.0],
                              self.merge_method]
                for encoder in self.encoders:
                    model_args.append(encoder)
                self.cur_model = DrugResponsePredictor(*model_args)

            else:
                # Low-rank Multi-modal fusion
                model_args = [
                    cur_layer_sizes,
                    self.encoder_train,
                    [self.gnn_drug, new_config['gnn_out_channels']],  # gnn_info
                    new_config['lmf_output_dim'],  # output dim
                    new_config['lmf_rank'],  # rank
                    ['silu'] * (new_config['drp_num_layers'] + 1) + ['silu'],  # act_fun_list
                    [True] * (new_config['drp_num_layers'] + 1) + [False],  # batchnorm_list
                    [0.0] + [0.05] * new_config['drp_num_layers'] + [0.0],  # dropout_list
                ]
                for encoder in self.encoders:
                    model_args.append(encoder)
                self.cur_model = LMF(*model_args)

            # NOTE: Must move model to GPU before initializing optimizers!!!
            self.cur_model = self.cur_model.float()
            self.cur_model = self.cur_model.cuda()
            if self.loss_type == "mae":
                self.criterion = nn.L1Loss().cuda()
            elif self.loss_type == "mse":
                self.criterion = nn.MSELoss().cuda()
            elif self.loss_type == "rmsle":
                self.criterion = RMSLELoss().cuda()
            elif self.loss_type == "weighted_rmse" or self.loss_type == "rmse":
                self.criterion = WeightedRMSELoss().cuda()
            else:
                exit("Unknown loss function requested")
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
        self.avg_untrained_losses, \
        self.max_final_epoch = cross_validate(train_data=self.cur_train_data,
                                              train_function=drp_train if self.gnn_drug is False
                                              else gnn_drp_train,
                                              cv_folds=self.cur_cv_folds,
                                              batch_size=self.batch_size,
                                              cur_model=self.cur_model,
                                              criterion=self.criterion,
                                              max_epochs=self.max_epochs,
                                              learning_rate=self.learning_rate,
                                              NUM_WORKERS=NUM_WORKERS,
                                              theoretical_loss=True,
                                              patience=15,
                                              delta=0.01,
                                              verbose=True)

        duration = time.time() - start_time

        return {"avg_cv_train_loss": self.avg_train_losses,
                "avg_cv_valid_loss": self.avg_valid_losses,
                "avg_cv_untrain_loss": self.avg_untrained_losses,
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
