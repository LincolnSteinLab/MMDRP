import argparse
import time
import torch
import torch.nn as nn
from torch import optim
from torch.backends import cudnn
from torch.utils import data
from typing import Union
from CustomPytorchLayers import CustomCoder, CustomDense, CustomCNN
from DRPPreparation import create_cv_folds
from Models import DNNAutoEncoder
from DataImportModules import OmicData, MorganData
from ModuleLoader import ExtractEncoder
from TrainFunctions import cross_validate, omic_train
from TuneTrainables import file_name_dict
import matplotlib as mpl
from matplotlib import pyplot as plt

import numpy as np
from sklearn import preprocessing
import pandas as pd

class ReduceViz:
    def __init__(self, path, data_filename: str, data_type: str, first_layer_size: int, code_layer_size: int,
                 num_layers: int, batchnorm_list=None, act_fun_list=None, dropout_list=0.0, to_gpu: bool = False,
                 verbose: bool = False):
        if code_layer_size > 3 or code_layer_size < 1:
            exit("Code layer size should be between 1 and 3")
        if data_type not in ["omic", "drug", "morgan"]:
            exit("Data type should be one of omic, drug or morgan")

        if data_type == "omic":
            self.cur_data = OmicData(path=path, omic_file_name=data_filename, to_gpu=to_gpu, verbose=verbose)
        else:
            # TODO to_gpu for MorganData?
            self.cur_data = MorganData(path=path, morgan_file_name=data_filename, model_type="dnn")

        self.first_layer_size = first_layer_size
        self.code_layer_size = code_layer_size
        self.num_layers = num_layers
        self.batchnorm_list = batchnorm_list
        self.act_fun_list = act_fun_list
        self.dropout_list = dropout_list
        self.to_gpu = to_gpu

        self.cur_model = DNNAutoEncoder(
            input_dim=self.cur_data.width(), first_layer_size=self.first_layer_size,
            code_layer_size=self.code_layer_size, num_layers=self.num_layers,
            batchnorm_list=self.batchnorm_list, act_fun_list=self.act_fun_list,
            dropout_list=self.dropout_list, name="DimReduce_AE"
        )
        self.cur_model.float()

    def fit_ae(self, max_num_epochs: int, early_stopping: bool, validation: bool, batch_size: int = 32,
               lr: float = 0.0001, save: bool = True, final_filename: str = None, verbose: bool = False):
        final_epoch = max_num_epochs
        if validation is True:
            # raise NotImplementedError
            cur_model = DNNAutoEncoder(
                input_dim=self.cur_data.width(), first_layer_size=self.first_layer_size,
                code_layer_size=self.code_layer_size, num_layers=self.num_layers,
                batchnorm_list=self.batchnorm_list, act_fun_list=self.act_fun_list,
                dropout_list=self.dropout_list, name="DimReduce_AE")
            cur_model = cur_model.float()

            cur_criterion = nn.L1Loss()
            if self.to_gpu is True:
                cur_model.cuda()
                cur_criterion.cuda()
                cur_criterion = nn.L1Loss()

            # Use 1/5 of the data for validation, ONLY to determine the number of epochs to train before over-fitting
            cv_folds = create_cv_folds(train_data=self.cur_data, train_attribute_name="data_info",
                                       sample_column_name="stripped_cell_line_name", n_folds=5,
                                       class_data_index=None, subset_type="cell_line",
                                       class_column_name="primary_disease", seed=42, verbose=verbose)
            train_valid_fold = [cv_folds[0]]
            # Determine the number of epochs required for training before validation loss doesn't improve
            print("Using validation set to determine performance plateau\n" +
                  "Batch Size:", batch_size)
            train_loss, valid_loss, final_epoch = cross_validate(train_data=self.cur_data, train_function=omic_train,
                                                                 cv_folds=train_valid_fold, batch_size=batch_size,
                                                                 cur_model=cur_model, criterion=cur_criterion,
                                                                 patience=5,
                                                                 delta=0.001,
                                                                 max_epochs=max_num_epochs,
                                                                 learning_rate=lr,
                                                                 to_gpu=self.to_gpu,
                                                                 verbose=verbose)

        print("Starting training based on number of epochs before validation loss worsens")
        cur_criterion = nn.L1Loss()
        if self.to_gpu is True:
            self.cur_model.cuda()
            cur_criterion.cuda()

        # Optimizer has to be defined after model, so that it points to the correct address
        optimizer = optim.Adam(self.cur_model.parameters(), lr=lr)
        train_loader = data.DataLoader(self.cur_data, batch_size=batch_size, num_workers=0, shuffle=True,
                                       pin_memory=False)

        avg_train_losses = []
        start = time.time()
        for epoch in range(final_epoch):
            train_losses = omic_train(self.cur_model, cur_criterion, optimizer, epoch=epoch, train_loader=train_loader,
                                      to_gpu=self.to_gpu, verbose=True)

            avg_train_losses.append(train_losses.avg)

            duration = time.time() - start
            print_msg = (f'[{epoch:>{max_num_epochs}}/{final_epoch:>{max_num_epochs}}] ' +
                         f'train_loss: {train_losses.avg:.5f} ' +
                         f'epoch_time: {duration:.4f}')

            print(print_msg)
            start = time.time()

            # print("Finished epoch in", str(epoch + 1), str(duration), "seconds")
        if save is True:
            print("Final file name:", final_filename)
            # Save entire model, not just the state dict
            torch.save(self.cur_model, final_filename)
            print("Finished training!")

    def viz(self, model_path, to_gpu: bool = True):
        if to_gpu is True:
            map_location = "cuda"
        else:
            map_location = "cpu"

        if model_path is not None:
            self.cur_model = torch.load(model_path, map_location=map_location)
            # cur_model = torch.load(model_path, map_location=map_location)

        # Extract encoder from auto-encoder
        cur_encoder = ExtractEncoder(self.cur_model)
        # cur_encoder = ExtractEncoder(cur_model)
        test_loader = data.DataLoader(self.cur_data, batch_size=32, num_workers=0, shuffle=False,
                                      pin_memory=False)
        cur_data = cur_viz.cur_data
        cur_data.data_info['primary_disease']
        test_loader = data.DataLoader(cur_data, batch_size=32, num_workers=0, shuffle=False,
                                      pin_memory=False)

        # Make predictions with the auto-encoder
        all_reduced_datapoints = []
        for i, data_point in enumerate(test_loader):
            all_reduced_datapoints.append(cur_encoder(data_point))

        all_reduced = torch.vstack(all_reduced_datapoints)
        all_reduced = all_reduced.detach().numpy()

        np.max(all_reduced[:, 0])
        np.min(all_reduced[:, 0])

        np.max(all_reduced[:, 1])
        np.min(all_reduced[:, 1])

        le = preprocessing.LabelEncoder()
        le.fit(cur_data.data_info['primary_disease'])
        tag = le.transform(cur_data.data_info['primary_disease'])
        N = len(pd.unique(cur_data.data_info['primary_disease'])) # Number of labels

        # setup the plot
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        # define the data
        x = all_reduced[:, 0]
        y = all_reduced[:, 1]

        # define the colormap
        cmap = plt.cm.jet
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

        # define the bins and normalize
        bounds = np.linspace(0,N,N+1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # make the scatter
        scat = ax.scatter(x,y,c=tag,cmap=cmap, norm=norm)
        # create the colorbar
        cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
        cb.set_label('Custom cbar')
        ax.set_title('Discrete color mappings')
        plt.show()

        plt.figure(figsize = (17,5))
        plt.subplot(131)
        plt.scatter(all_reduced[:,0], all_reduced[:,1], c=cur_data.data_info['primary_disease'])
        # plt.scatter(all_reduced[:,0],all_reduced[:,1],  c = y, cmap = "RdGy",
        #             edgecolor = "None", alpha=1, vmin = 75, vmax = 150)
        plt.colorbar()
        plt.title('AE Scatter Plot')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This programs trains an auto-encoder on omic data for visualization")

    parser.add_argument('--machine', help='Whether code is run on cluster or else', default="cluster")
    parser.add_argument('--max_num_epochs', help='Number of epochs to run', required=False)
    parser.add_argument('--data_type', help='Data type to be used in training this auto-encoder', required=True)
    parser.add_argument('--name_tag', help='A string that will be added to the file name generated by this program',
                        required=False)
    parser.add_argument('--first_layer_size', required=False)
    parser.add_argument('--code_layer_size', required=False)
    parser.add_argument('--num_layers', required=False)
    # parser.add_argument('--batchnorm_list', required=False)
    # parser.add_argument('--dropout_list', required=False)
    parser.add_argument('--act_fun', required=False)
    parser.add_argument('--batch_size', required=False)
    parser.add_argument('--lr', required=False)
    args = parser.parse_args()

    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    if args.machine == "mist":
        path = "~/.conda/envs/drp1/Data/DRP_Training_Data/"
        to_gpu = True
    else:
        path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
        to_gpu = False

    file_name = file_name_dict[args.data_type + "_file_name"]

    num_layers = int(args.num_layers)
    code_layer_size = int(args.code_layer_size)
    first_layer_size = int(args.first_layer_size)
    max_num_epochs = int(args.max_num_epochs)
    lr = float(args.lr)
    batch_size=int(args.batch_size)

    # num_layers = 2
    # code_layer_size = 3
    # first_layer_size = 135
    # max_num_epochs = 50
    # lr = 0.0001
    # batch_size = 32
    # to_gpu = False
    # file_name = "DepMap_21Q2_Training_Expression.hdf"
    cur_viz = ReduceViz(
        path=path, data_filename=file_name, data_type="omic", first_layer_size=first_layer_size,
        code_layer_size=code_layer_size, num_layers=num_layers,
        act_fun_list=[args.act_fun]*num_layers, batchnorm_list=[False]*num_layers,
        dropout_list=0.05,
        to_gpu=to_gpu, verbose=True
    )
    # cur_data = cur_viz.cur_data
    # train_loader = data.DataLoader(cur_viz.cur_data, batch_size=32,
    #                                    num_workers=0, pin_memory=False, drop_last=True)
    # train_iter = iter(train_loader)
    # next(train_iter)
    # next(train_iter).shape
    # TODO Allow omic_train to be run on CPU (must change auto-encoder prefetcher)
    cur_viz.fit_ae(
        max_num_epochs=max_num_epochs, batch_size=batch_size, validation=False, lr=lr,
        early_stopping=False, verbose=True,
        final_filename="/scratch/l/lstein/ftaj/DataViz/" + args.data_type.upper() + "_Embed_AE_Viz.pth")

    # cur_viz.viz(model_path=path+"/DataViz/EXP_Embed_AE_Viz.pth")
# python3 DRP/src/data_viz.py --machine mist --data_type exp --max_num_epochs 100 --first_layer_size 4096 --code_layer_size 2 --num_layers 4 --lr 0.0001
