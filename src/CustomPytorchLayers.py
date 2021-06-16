# This file contains custom Pytorch layers for use in other DRP modules
import sys

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

# activation_dict = NeuralNets.ModuleDict({
#     'lrelu': NeuralNets.LeakyReLU(),
#     'prelu': NeuralNets.PReLU(),
#     'relu': NeuralNets.ReLU()
# })


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).float()
        # self.noise = self.noise.to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class CustomDense(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, act_fun: str = None,
                 batch_norm: bool = False, dropout: float = 0., name: str = ""):
        super(CustomDense, self).__init__()
        if act_fun is None or act_fun == "none" or act_fun == "None" or act_fun == "NONE":
            cur_act_fun = nn.Identity()
        elif act_fun == 'relu':
            cur_act_fun = nn.ReLU()
        elif act_fun == 'lrelu':
            cur_act_fun = nn.LeakyReLU()
        elif act_fun == 'prelu':
            cur_act_fun = nn.PReLU()
        elif act_fun == 'sigmoid':
            cur_act_fun = nn.Sigmoid()
        elif act_fun == 'tanh':
            cur_act_fun = nn.Tanh()
        else:
            Warning("Uknown activation function given, defaulting to ReLU")
            cur_act_fun = nn.ReLU()

        self.custom_dense = OrderedDict([
            (name + "_linear", nn.Linear(in_features=input_size, out_features=hidden_size)),
            (name + "_batchnorm", nn.BatchNorm1d(num_features=hidden_size, affine=True, track_running_stats=False))
            if batch_norm is True else (name + "identity", nn.Identity()),
            (name + "_activation", cur_act_fun),
            (name + "_dropout", nn.Dropout(p=dropout))])
        self.custom_dense = nn.Sequential(self.custom_dense)

    def forward(self, x):
        return self.custom_dense(x)


class CustomCoder(nn.Module):
    """
    This class creates an encoder module that takes the first and last layer sizes as well as the desired
    number of layers in between, and returns an encoder neural network with the desired specifications.
    """

    def __init__(self, input_size, coder_layer_sizes: list = None, first_layer_size: int = None,
                 code_layer_size: int = None, code_layer: bool = False, num_layers: int = None,
                 batchnorm_list=None, act_fun_list: str = None, dropout_list=None,
                 encode: bool = True, name: str = ""):
        super(CustomCoder, self).__init__()

        # Check layer-wise parameters
        if num_layers is None:
            num_layers = len(coder_layer_sizes)
        if batchnorm_list is None:
            batchnorm_list = [False] * num_layers
        else:
            assert len(
                batchnorm_list) == num_layers, "Length of batchnorm_list should be the same as the number of layers"
        if dropout_list is None:
            dropout_list = [0.] * num_layers
        else:
            assert len(dropout_list) == num_layers, "Length of dropout_list should be the same as the number of layers"

        if act_fun_list is None:
            act_fun_list = [None] * num_layers

        self.batchnorm_list = batchnorm_list
        self.act_fun = act_fun_list
        self.dropout_list = dropout_list

        self.input_size = input_size
        self.layer_sizes = coder_layer_sizes
        self.first_layer_size = first_layer_size
        self.code_layer = code_layer
        self.code_layer_size = code_layer_size
        self.num_layers = num_layers
        self.encode = encode

        if not coder_layer_sizes and not first_layer_size and not code_layer_size and not num_layers:
            sys.exit("Must provide either the layer sizes or the 3 specifications to make coder automatically")

        # Differentiate encoder and decoder creation
        if encode is True:
            # np.linspace allows for the creation of an array with equidistant numbers from start to end
            if coder_layer_sizes is None:
                coder_layer_sizes = np.linspace(first_layer_size, code_layer_size, num_layers).astype(int)

            # The first layer should take the data input width and then continue until the code layer
            self.coder = [CustomDense(input_size=self.input_size, hidden_size=coder_layer_sizes[0],
                                      act_fun=act_fun_list[0], batch_norm=batchnorm_list[0],
                                      dropout=dropout_list[0], name="encoder_0_" + name)] + \
                         [CustomDense(input_size=coder_layer_sizes[i], hidden_size=coder_layer_sizes[i + 1],
                                      act_fun=act_fun_list[0], batch_norm=batchnorm_list[i + 1],
                                      dropout=dropout_list[i + 1], name="encoder_" + str(i) + '_' + name) for i in
                          range(len(coder_layer_sizes) - 2)]

            if code_layer is True:
                self.coder += [CustomDense(input_size=coder_layer_sizes[-2], hidden_size=coder_layer_sizes[-1],
                                           act_fun=act_fun_list[-1], batch_norm=batchnorm_list[-1],
                                           dropout=dropout_list[-1], name="code_layer_" + name)]

            self.coder = nn.Sequential(*self.coder)
        # Mirror the encoder setup for decoder creation
        else:
            batchnorm_list_rev = batchnorm_list[::-1]
            dropout_list_rev = dropout_list[::-1]
            if coder_layer_sizes is None:
                coder_layer_sizes = np.linspace(code_layer_size, first_layer_size, num_layers).astype(int)

            # The first layer is the code layer, and will continue until the layer that recreates the data input
            self.coder = []
            if code_layer is True:
                # Input to the code layer is simply from the layer after (mirror of the layer before)
                self.coder += [CustomDense(input_size=coder_layer_sizes[1], hidden_size=coder_layer_sizes[0],
                                           act_fun=act_fun_list[0], batch_norm=batchnorm_list_rev[0],
                                           dropout=dropout_list_rev[0], name="code_layer_" + name)]
            self.coder += [CustomDense(input_size=coder_layer_sizes[i], hidden_size=coder_layer_sizes[i + 1],
                                       act_fun=act_fun_list[i + 1], batch_norm=batchnorm_list_rev[i + 1],
                                       dropout=dropout_list_rev[i + 1], name="decoder_" + str(i) + '_' + name) for i in
                           range(len(coder_layer_sizes) - 1)] + \
                          [CustomDense(input_size=coder_layer_sizes[-1], hidden_size=self.input_size,
                                       act_fun=act_fun_list[-1], batch_norm=batchnorm_list_rev[-1],
                                       dropout=dropout_list_rev[-1], name="decoder_last_" + name)]
            self.coder = nn.Sequential(*self.coder)

    def forward(self, x):
        return self.coder(x)


class CustomCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 act_fun: str = None, transpose: bool = False,
                 batch_norm=False, dropout=0., name=""):
        super(CustomCNN, self).__init__()
        if act_fun is None:
            cur_act_fun = nn.Identity()
        elif act_fun == 'relu':
            cur_act_fun = nn.ReLU()
        elif act_fun == 'lrelu':
            cur_act_fun = nn.LeakyReLU()
        elif act_fun == 'prelu':
            cur_act_fun = nn.PReLU()
        else:
            Warning("Uknown activation function given, defaulting to ReLU")
            cur_act_fun = nn.ReLU()
        self.custom_conv = OrderedDict([
            (name + "_conv1d", nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                                         padding=padding)) if transpose is False else (
                name + "_convtrans1d", nn.ConvTranspose1d(
                    in_channels, out_channels, kernel_size, stride=stride, padding=padding)),
            (name + "_activation", cur_act_fun),
            (name + "_batchnorm1d", nn.BatchNorm1d(num_features=out_channels, affine=True, track_running_stats=False))
            if batch_norm is True else (name + "identity", nn.Identity()),
            (name + "_dropout", nn.Dropout(p=dropout))
        ])
        self.custom_conv = nn.Sequential(self.custom_conv)

    def forward(self, x):
        return self.custom_conv(x)

# class DeepCNNAutoEncoder(NeuralNets.Module):
#     def __init__(self, num_branch, code_layer_size=None, kernel_size_list=None, in_channels=1,
#                  out_channels_list=None, batchnorm_list=None, act_fun_list=None, dropout_list=None, encode=True):
#         super(DeepCNNAutoEncoder, self).__init__()
