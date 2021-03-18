import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ray.tune import Trainable
from ray.tune.utils import validate_save_restore
from torch.utils import data

from DRPPreparation import drp_create_datasets, drp_main_prep, autoencoder_create_datasets, drp_load_datatypes
from DataImportModules import OmicData, MorganData
from Models import DNNAutoEncoder, MultiHeadCNNAutoEncoder, DrugResponsePredictor
from ModuleLoader import ExtractEncoder
from TrainFunctions import morgan_train, omic_train, drp_train, drp_cross_validate

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

NUM_WORKERS = 0


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
            self.cur_model = MultiHeadCNNAutoEncoder(
                input_width=int(self.config["width"]),
                num_branch=self.config["num_branch"],
                stride=self.config["stride"],
                batchnorm=self.config["batchnorm"],
                act_fun=self.config["act_fun"],
                dropout=self.config["dropout"]
            )
        else:
            self.cur_model = DNNAutoEncoder(input_dim=int(self.config["width"]),
                                            first_layer_size=self.config["first_layer_size"],
                                            code_layer_size=self.config["code_layer_size"],
                                            num_layers=self.config["num_layers"],
                                            batchnorm=self.config["batchnorm"],
                                            act_fun=self.config["act_fun"],
                                            dropout=self.config['dropout'])

        self.cur_model = self.cur_model.float()
        self.cur_model = self.cur_model.cuda()
        self.criterion = nn.MSELoss().cuda()
        self.optimizer = optim.Adam(self.cur_model.parameters(), lr=self.config["lr"])

    def reset_config(self, new_config):
        try:
            # Reset model with new_config ==============
            if self.model_type == "cnn":
                self.cur_model = MultiHeadCNNAutoEncoder(
                    input_width=int(new_config["width"]),
                    num_branch=new_config["num_branch"],
                    stride=new_config["stride"],
                    batchnorm=new_config["batchnorm"],
                    act_fun=new_config["act_fun"],
                    dropout=new_config["dropout"]
                )
            else:
                self.cur_model = DNNAutoEncoder(input_dim=int(new_config["width"]),
                                                first_layer_size=new_config["first_layer_size"],
                                                code_layer_size=new_config["code_layer_size"],
                                                num_layers=new_config["num_layers"],
                                                batchnorm=new_config["batchnorm"],
                                                act_fun=new_config["act_fun"],
                                                dropout=new_config['dropout'])

            self.cur_model = self.cur_model.float()
            self.cur_model = self.cur_model.cuda()
            self.criterion = nn.MSELoss().cuda()
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
        train_losses, valid_losses = morgan_train(self.train_loader, self.cur_model, self.criterion,
                                                  self.optimizer, epoch=self.timestep,
                                                  valid_loader=self.valid_loader)

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
    def setup(self, config, checkpoint_dir="/.mounts/labs/steinlab/scratch/ftaj/"):
        self.data_dir = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/"
        self.timestep = 0
        self.config = config

        # Prepare data loaders ==========
        self.train_data = OmicData(path=self.data_dir,
                                   omic_file_name=file_name_dict[config["data_type"] + "_file_name"])
        self.train_data, self.train_sampler, self.valid_sampler, \
        self.train_idx, self.valid_idx = autoencoder_create_datasets(train_data=self.train_data)

        print("Train data length:", len(self.train_data))

        # Create data_loaders
        self.train_loader = data.DataLoader(self.train_data, batch_size=config["batch_size"],
                                            sampler=self.train_sampler, num_workers=NUM_WORKERS, pin_memory=True,
                                            drop_last=True)
        self.valid_loader = data.DataLoader(self.train_data, batch_size=config["batch_size"] * 4,
                                            sampler=self.valid_sampler, num_workers=NUM_WORKERS, pin_memory=True,
                                            drop_last=True)

        # Prepare model =============
        self.cur_model = DNNAutoEncoder(input_dim=self.train_data.width(),
                                        first_layer_size=self.config["first_layer_size"],
                                        code_layer_size=self.config["code_layer_size"],
                                        num_layers=self.config["num_layers"],
                                        batchnorm=self.config["batchnorm"],
                                        act_fun=self.config["act_fun"])

        self.cur_model = self.cur_model.float()
        self.cur_model = self.cur_model.cuda()
        self.criterion = nn.MSELoss().cuda()
        self.optimizer = optim.Adam(self.cur_model.parameters(), lr=self.config["lr"])

    def reset_config(self, new_config):
        try:
            # Reset model with new_config ===========
            self.cur_model = DNNAutoEncoder(input_dim=self.train_data.width(),
                                            first_layer_size=new_config["first_layer_size"],
                                            code_layer_size=new_config["code_layer_size"],
                                            num_layers=new_config["num_layers"],
                                            batchnorm=new_config["batchnorm"],
                                            act_fun=new_config["act_fun"])

            self.cur_model = self.cur_model.float()
            self.cur_model = self.cur_model.cuda()
            self.criterion = nn.MSELoss().cuda()
            self.optimizer = optim.Adam(self.cur_model.parameters(), lr=new_config["lr"])

            # Reset data loaders ===============
            self.train_loader = data.DataLoader(self.train_data, batch_size=new_config["batch_size"],
                                                sampler=self.train_sampler, num_workers=NUM_WORKERS, pin_memory=True,
                                                drop_last=True)
            self.valid_loader = data.DataLoader(self.train_data, batch_size=new_config["batch_size"] * 4,
                                                sampler=self.valid_sampler, num_workers=NUM_WORKERS, pin_memory=True,
                                                drop_last=True)
            return True
        except:
            return False

    def step(self):
        self.timestep += 1
        # to track the average training loss per epoch as the model trains
        # avg_train_losses = []
        # for epoch in range(last_epoch, num_epochs):
        start_time = time.time()
        train_losses, valid_losses = omic_train(self.train_loader, self.cur_model, self.criterion, self.optimizer,
                                                epoch=self.timestep, valid_loader=self.valid_loader)

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


class DRPTrainable(Trainable):
    def setup(self, config, checkpoint_dir="/.mounts/labs/steinlab/scratch/ftaj/"):
        self.data_dir = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/"
        self.bottleneck_path = "bottleneck_keys.csv"
        self.bottleneck_keys = None
        self.bottleneck = config['bottleneck']
        if config['bottleneck'] is True:
            try:
                self.bottleneck_keys = pd.read_csv(self.data_dir + self.bottleneck_path)['keys'].to_list()
            except:
                exit("Could not read bottleneck file")

        cur_modules = config["data_types"].split('_')
        assert "drug" in cur_modules, "Drug data must be provided for drug-response prediction (training or testing)"
        assert len(cur_modules) > 1, "Data types to be used must be indicated by: mut, cnv, exp, prot and drug"

        # Load data and auto-encoders
        self.data_list, autoencoder_list, self.key_columns, \
        self.gpu_locs = drp_load_datatypes(train_file=config['train_file'],
                                           # module_list=['drug', 'mut', 'cnv', 'exp', 'prot'],
                                           module_list=cur_modules,
                                           PATH=self.data_dir,
                                           file_name_dict=file_name_dict,
                                           device='cpu')
        # Extract encoders from loaded auto-encoders
        self.encoders = [ExtractEncoder(autoencoder) for autoencoder in autoencoder_list]

        # Subset data and auto-encoders based on Trial's data types

        self.timestep = 0
        self.config = config
        self.batch_size = self.config["batch_size"]

        # prep_list = drp_main_prep(module_list=cur_modules, train_file=config['train_file'])

        # prep but ignore the created model. required_data_indices stems from cur_modules
        # _, _, self.subset_data, self.subset_keys, self.subset_encoders, \
        # self.data_list, self.key_columns = prep_list

        # Determine layer sizes, add final target layer
        cur_layer_sizes = list(np.linspace(self.config['drp_first_layer_size'],
                                           self.config['drp_last_layer_size'],
                                           self.config['drp_num_layers']).astype(int))
        cur_layer_sizes.append(1)
        # Note that the order of arguments must be respected here!!!
        # print("len(self.subset_encoders):", len(self.subset_encoders))

        # Create variable length argument set for DRP model creation, depending on number of given encoders
        model_args = [cur_layer_sizes, [0] * len(self.encoders),
                      False, self.config['act_fun'],
                      self.config['batchnorm'],
                      self.config['dropout']]
        for encoder in self.encoders:
            model_args.append(encoder)
        self.cur_model = DrugResponsePredictor(*model_args)

        # NOTE: Must move model to GPU before initializing optimizers!!!
        self.cur_model = self.cur_model.float()
        self.cur_model = self.cur_model.cuda()
        self.criterion = nn.MSELoss().cuda()
        self.optimizer = optim.Adam(self.cur_model.parameters(), lr=self.config["lr"])

        # Then the function will automatically make "new" indices and return them for saving in the checkpoint file
        # NOTE: The seed is the same, so the same subset is selected every time (!)
        # TODO Cross validation
        # Load bottleneck and full data once, choose later on
        self.bottleneck_train_data, \
        self.bottleneck_cv_folds = drp_create_datasets(self.data_list,
                                                       self.key_columns,
                                                       n_folds=config['n_folds'],
                                                       drug_index=0,
                                                       drug_dr_column="area_above_curve",
                                                       class_column_name="primary_disease",
                                                       subset_type="cell_line",
                                                       bottleneck_keys=self.bottleneck_keys)
        self.train_data, self.cv_folds = drp_create_datasets(self.data_list,
                                                             self.key_columns,
                                                             n_folds=config['n_folds'],
                                                             drug_index=0,
                                                             drug_dr_column="area_above_curve",
                                                             class_column_name="primary_disease",
                                                             subset_type="cell_line",
                                                             test_drug_data=None)

        # Create K cross-validation folds which will be selected based on index
        # self.cv_index = config[ray.tune.suggest.repeater.TRIAL_INDEX]

        if config['bottleneck'] is True:
            # Get train and validation indices for the current fold
            self.cur_cv_folds = self.bottleneck_cv_folds
            self.cur_train_data = self.bottleneck_train_data

            # self.bottleneck_cur_fold = self.bottleneck_cv_folds[self.cv_index]
            # self.bottleneck_cur_train_sampler = SubsetRandomSampler(self.bottleneck_cur_fold[0])
            # self.bottleneck_cur_valid_sampler = SubsetRandomSampler(self.bottleneck_cur_fold[1])
            # # Create data loaders based on current fold's indices
            # self.train_loader = data.DataLoader(self.bottleneck_train_data, batch_size=self.batch_size,
            #                                     sampler=self.bottleneck_cur_train_sampler,
            #                                     num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
            # # load validation data in batches 4 times the size
            # self.valid_loader = data.DataLoader(self.bottleneck_train_data, batch_size=self.batch_size * 4,
            #                                     sampler=self.bottleneck_cur_valid_sampler,
            #                                     num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        else:
            self.cur_cv_folds = self.cv_folds
            self.cur_train_data = self.train_data

            # self.cur_fold = self.cv_folds[self.cv_index]
            # self.cur_train_sampler = SubsetRandomSampler(self.cur_fold[0])
            # self.cur_valid_sampler = SubsetRandomSampler(self.cur_fold[1])
            #
            # # Create data loaders based on current fold's indices
            # self.train_loader = data.DataLoader(self.train_data, batch_size=self.batch_size,
            #                                     sampler=self.cur_train_sampler,
            #                                     num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
            # # load validation data in batches 4 times the size
            # self.valid_loader = data.DataLoader(self.train_data, batch_size=self.batch_size * 4,
            #                                     sampler=self.cur_valid_sampler,
            #                                     num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

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
                          False, new_config['act_fun'],
                          new_config['batchnorm'],
                          new_config['dropout']]
            for encoder in self.encoders:
                model_args.append(encoder)
            self.cur_model = DrugResponsePredictor(*model_args)

            self.cur_model = self.cur_model.float()
            self.cur_model = self.cur_model.cuda()
            self.criterion = nn.MSELoss().cuda()
            self.optimizer = optim.Adam(self.cur_model.parameters(), lr=new_config["lr"])

            # Reset data loaders ==============
            # Create K cross-validation folds which will be selected based on index
            # self.cv_index = new_config[ray.tune.suggest.repeater.TRIAL_INDEX]

            if new_config['bottleneck'] is True:
                self.cur_cv_folds = self.bottleneck_cv_folds
                self.cur_train_data = self.bottleneck_train_data

                # # Get train and validation indices for the current fold
                # self.bottleneck_cur_fold = self.bottleneck_cv_folds[self.cv_index]
                # self.bottleneck_cur_train_sampler = SubsetRandomSampler(self.bottleneck_cur_fold[0])
                # self.bottleneck_cur_valid_sampler = SubsetRandomSampler(self.bottleneck_cur_fold[1])
                # # Create data loaders based on current fold's indices
                # self.train_loader = data.DataLoader(self.bottleneck_train_data, batch_size=self.batch_size,
                #                                     sampler=self.bottleneck_cur_train_sampler,
                #                                     num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
                # # load validation data in batches 4 times the size
                # self.valid_loader = data.DataLoader(self.bottleneck_train_data, batch_size=self.batch_size * 4,
                #                                     sampler=self.bottleneck_cur_valid_sampler,
                #                                     num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
            else:
                self.cur_cv_folds = self.cv_folds
                self.cur_train_data = self.train_data

                # self.cur_fold = self.cv_folds[self.cv_index]
                # self.cur_train_sampler = SubsetRandomSampler(self.cur_fold[0])
                # self.cur_valid_sampler = SubsetRandomSampler(self.cur_fold[1])
                #
                # # Create data loaders based on current fold's indices
                # self.train_loader = data.DataLoader(self.train_data, batch_size=self.batch_size,
                #                                     sampler=self.cur_train_sampler,
                #                                     num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
                # # load validation data in batches 4 times the size
                # self.valid_loader = data.DataLoader(self.train_data, batch_size=self.batch_size * 4,
                #                                     sampler=self.cur_valid_sampler,
                #                                     num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

            return True
        except:
            # If reset_config fails, catch exception and force Ray to make a new actor
            return False

    def step(self):
        self.timestep += 1
        start_time = time.time()

        self.all_sum_train_losses,\
        self.all_sum_valid_losses,\
        self.all_avg_valid_losses = drp_cross_validate(train_data=self.cur_train_data,
                                                       cv_folds=self.cur_cv_folds,
                                                       batch_size=self.batch_size,
                                                       cur_model=self.cur_model,
                                                       criterion=self.criterion,
                                                       optimizer=self.optimizer,
                                                       epoch=self.timestep,
                                                       NUM_WORKERS=NUM_WORKERS)
        # train_losses, valid_losses = drp_train(train_loader=self.train_loader, valid_loader=self.valid_loader,
        #                                        cur_model=self.cur_model,
        #                                        criterion=self.criterion, optimizer=self.optimizer, epoch=self.timestep)

        duration = time.time() - start_time
        self.sum_cv_train_loss = sum(self.all_sum_train_losses)
        self.sum_cv_valid_loss = sum(self.all_sum_valid_losses)
        self.avg_cv_valid_loss = sum(self.all_sum_valid_losses) / len(self.all_sum_valid_losses)

        return {"sum_cv_valid_loss": self.sum_cv_valid_loss,
                "sum_cv_train_loss": self.sum_cv_train_loss,
                "avg_cv_valid_loss": self.avg_cv_valid_loss,
                "num_samples": len(self.cur_cv_folds[0][0]),
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


class FullModelTrainable(Trainable):
    def setup(self, config, checkpoint_dir="/.mounts/labs/steinlab/scratch/ftaj/"):
        self.data_dir = "/u/ftaj/anaconda3/envs/Python/Data/DRP_Training_Data/"
        # self.data_dir = "/home/l/lstein/ftaj/.conda/envs/drp1/Data/DRP_Training_Data/"
        # self.data_dir = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
        self.timestep = 0
        self.config = config
        print(config)

        self.batch_size = self.config["batch_size"]
        # Since args.all_combos will be false, the main_prep function will only yield the desired combo
        prep_gen = drp_main_prep(["drug", "mut", "cnv", "exp", "prot"], train_file=config['train_file'])
        prep_list = next(prep_gen)
        # prep but ignore the created modules (except for drug data)
        _, self.final_address, self.subset_data, self.subset_keys, cur_encoders, \
        self.data_list, self.key_columns, self.required_data_indices = prep_list
        # del cur_encoders
        del _
        morgan_encoder = cur_encoders[0]
        # drug_model = AutoEncoder(input_dim=self.subset_data[0].width(),
        #                            first_layer_size=config['morgan_first_layer_size'],
        #                            code_layer_size=config['morgan_code_layer_size'],
        #                            num_layers=config['morgan_num_layers'], name="drug")

        mut_model = DNNAutoEncoder(input_dim=self.subset_data[1].width(),
                                   first_layer_size=config['mut_first_layer_size'],
                                   code_layer_size=config['mut_code_layer_size'],
                                   num_layers=config['mut_num_layers'], name="mut")
        cnv_model = DNNAutoEncoder(input_dim=self.subset_data[2].width(),
                                   first_layer_size=config['cnv_first_layer_size'],
                                   code_layer_size=config['cnv_code_layer_size'],
                                   num_layers=config['cnv_num_layers'], name="cnv")
        exp_model = DNNAutoEncoder(input_dim=self.subset_data[3].width(),
                                   first_layer_size=config['exp_first_layer_size'],
                                   code_layer_size=config['exp_code_layer_size'],
                                   num_layers=config['exp_num_layers'], name="exp")
        prot_model = DNNAutoEncoder(input_dim=self.subset_data[4].width(),
                                    first_layer_size=config['prot_first_layer_size'],
                                    code_layer_size=config['prot_code_layer_size'],
                                    num_layers=config['prot_num_layers'], name="prot")
        cur_autoencoders = [mut_model, cnv_model, exp_model, prot_model]
        self.subset_encoders = [ExtractEncoder(autoencoder) for autoencoder in cur_autoencoders]
        self.subset_encoders.insert(0, morgan_encoder)
        del cur_autoencoders
        # Determine layer sizes, add final target layer
        cur_layer_sizes = list(np.linspace(config['drp_first_layer_size'],
                                           config['drp_last_layer_size'],
                                           config['drp_num_layers']).astype(int))
        cur_layer_sizes.append(1)
        self.cur_model = DrugResponsePredictor(*[cur_layer_sizes, [0] * len(self.subset_encoders),
                                                 False,
                                                 self.subset_encoders[0],
                                                 self.subset_encoders[1],
                                                 self.subset_encoders[2],
                                                 self.subset_encoders[3],
                                                 self.subset_encoders[4]])
        # input_sizes = [(config['batch_size'], 1, self.data_list[0].width()),
        #                (config['batch_size'], 1, self.data_list[1].width()),
        #                (config['batch_size'], 1, self.data_list[2].width()),
        #                (config['batch_size'], 1, self.data_list[3].width()),
        #                (config['batch_size'], 1, self.data_list[4].width())]

        # NOTE: Must move model to GPU before initializing optimizers!!!
        self.cur_model = self.cur_model.float()
        self.cur_model = self.cur_model.cuda()

        self.criterion = nn.MSELoss().cuda()
        self.optimizer = optim.Adam(self.cur_model.parameters(), lr=config["lr"])

        # Then the function will automatically make "new" indices and return them for saving in the checkpoint file
        # NOTE: The seed is the same, so the same subset is selected every time
        self.train_data, self.train_sampler, self.valid_sampler, \
        self.train_idx, self.valid_idx = drp_create_datasets(self.data_list,
                                                             self.key_columns,
                                                             drug_index=0,
                                                             drug_dr_column="area_above_curve",
                                                             test_drug_data=None)

        self.train_loader = data.DataLoader(self.train_data, batch_size=self.batch_size,
                                            sampler=self.train_sampler,
                                            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        # load validation data in batches 4 times the size
        self.valid_loader = data.DataLoader(self.train_data, batch_size=self.batch_size * 4,
                                            sampler=self.valid_sampler,
                                            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    def reset_config(self, new_config):
        try:
            # Reset model with new_config ===================
            # Determine layer sizes, add final target layer
            cur_layer_sizes = list(np.linspace(new_config['first_layer_size'], new_config['last_layer_size'],
                                               new_config['num_layers']).astype(int))
            cur_layer_sizes.append(1)
            self.cur_model = DrugResponsePredictor(*[cur_layer_sizes, [0] * len(self.subset_encoders),
                                                     False, self.subset_encoders])
            # NOTE: Must move model to GPU before initializing optimizers!!!
            self.cur_model = self.cur_model.float()
            self.cur_model = self.cur_model.cuda()
            self.criterion = nn.MSELoss().cuda()
            self.optimizer = optim.Adam(self.cur_model.parameters(), lr=new_config["lr"])

            # Reset data loaders =========================
            self.train_loader = data.DataLoader(self.train_data, batch_size=self.batch_size,
                                                sampler=self.train_sampler,
                                                num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
            self.valid_loader = data.DataLoader(self.train_data, batch_size=self.batch_size * 4,
                                                sampler=self.valid_sampler,
                                                num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

            return True
        except:
            return False

    def step(self):
        self.timestep += 1
        start_time = time.time()
        train_losses, valid_losses = drp_train(train_loader=self.train_loader, valid_loader=self.valid_loader,
                                               cur_model=self.cur_model,
                                               criterion=self.criterion, optimizer=self.optimizer, epoch=self.timestep)

        duration = time.time() - start_time
        self.sum_valid_loss = valid_losses.sum
        self.sum_train_loss = train_losses.sum
        return {"sum_valid_loss": self.sum_valid_loss,
                "sum_train_loss": self.sum_train_loss,
                "time_this_iter_s": duration}

    def save_checkpoint(self, tmp_checkpoint_dir):
        validate_save_restore(FullModelTrainable)
        print("save_checkpoint dir:", tmp_checkpoint_dir)
        path = os.path.join(tmp_checkpoint_dir, "checkpoint")
        torch.save({"model_state_dict": self.cur_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "timestep": self.timestep}, path)

        return path

    def load_checkpoint(self, checkpoint):
        print("load_checkpoint dir:", checkpoint)
        cur_checkpoint = torch.load(checkpoint)
        self.cur_model.load_state_dict(cur_checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(cur_checkpoint["optimizer_state_dict"])
        self.timestep = cur_checkpoint["timestep"]
