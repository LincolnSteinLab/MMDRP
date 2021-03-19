# This script contains a convolutional auto-encoder based on the Inception an ResNet
# architectures to encode nucleotide sequences, such as shRNA

import pandas as pd
import numpy as np
import time
import sys
import gc
import logging
import threading
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchsummary import summary

# import pdb
from apex import amp
# torch.set_flush_denormal(True)

# from torch.utils.data import DataLoader

PATH = "/u/ftaj/anaconda3/envs/Drug_Response/Data/RNAi/Train_Data/"
# PATH = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/RNAi/Train_Data/"

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA is available!')
device = torch.device("cuda:0" if use_cuda else "cpu")
# Turning off benchmarking makes highly dynamic models faster
cudnn.benchmark = False
cudnn.deterministic = True
BOTTLENECK_CHANNELS = 4


class Conv1DBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv1DBN, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              bias=False, padding=padding, stride=stride)
        self.bn = nn.BatchNorm1d(num_features=out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.relu(x0)
        x2 = self.bn(x1)

        return x2


class Stem(nn.Module):
    def __init__(self, in_channels, x_filters, stem_oc):
        super(Stem, self).__init__()
        self.conv3 = Conv1DBN(in_channels, stem_oc*x_filters, kernel_size=3, padding=1)

    def forward(self, x):
        x0 = self.conv3(x)
        return x0


class InceptionA(nn.Module):
    def __init__(self, x_filters, scale, stem_oc):
        super(InceptionA, self).__init__()
        self.scale = scale
        self.branch_1 = nn.Sequential(
            Conv1DBN(in_channels=stem_oc*x_filters, out_channels=16*x_filters, kernel_size=1),
            Conv1DBN(in_channels=16*x_filters, out_channels=16*x_filters, kernel_size=3, padding=1)
        )
        self.branch_2 = nn.Sequential(
            Conv1DBN(in_channels=stem_oc*x_filters, out_channels=16*x_filters, kernel_size=1),
            Conv1DBN(in_channels=16*x_filters, out_channels=16*x_filters, kernel_size=5, padding=2),
            Conv1DBN(in_channels=16*x_filters, out_channels=16*x_filters, kernel_size=5, padding=2)
        )
        self.branch_3 = nn.Sequential(
            Conv1DBN(in_channels=stem_oc*x_filters, out_channels=16*x_filters, kernel_size=1, padding=0)
        )
        self.conv1 = Conv1DBN(in_channels=48*x_filters, out_channels=stem_oc*x_filters, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch_1(x)
        x1 = self.branch_2(x)
        x2 = self.branch_3(x)
        out = torch.cat((x0, x1, x2), dim=1)
        out = self.conv1(out)
        out = out * self.scale + x
        out = self.relu(out)

        return out


class ReductionA(nn.Module):
    def __init__(self, x_filters, stem_oc):
        super(ReductionA, self).__init__()
        self.branch_1 = nn.MaxPool1d(kernel_size=2)
        self.branch_2 = Conv1DBN(in_channels=stem_oc*x_filters, out_channels=16*x_filters, kernel_size=3, stride=2, padding=1)
        self.branch_3 = nn.Sequential(
            Conv1DBN(in_channels=stem_oc*x_filters, out_channels=8*x_filters, kernel_size=1, stride=1),
            Conv1DBN(in_channels=8*x_filters, out_channels=16*x_filters, kernel_size=3, stride=1, padding=1),
            Conv1DBN(in_channels=16*x_filters, out_channels=16*x_filters, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x0 = self.branch_1(x)
        x1 = self.branch_2(x)
        x2 = self.branch_3(x)
        out = torch.cat((x0, x1, x2), dim=1)

        return out


class InceptionB(nn.Module):
    def __init__(self, x_filters, scale, stem_oc):
        super(InceptionB, self).__init__()
        self.scale = scale

        self.branch_1 = nn.Sequential(
            Conv1DBN(in_channels=(16+16+stem_oc)*x_filters, out_channels=32*x_filters, kernel_size=1)
        )
        self.branch_2 = nn.Sequential(
            Conv1DBN(in_channels=((16+16+stem_oc))*x_filters, out_channels=32*x_filters, kernel_size=1),
            Conv1DBN(in_channels=32*x_filters, out_channels=32*x_filters, kernel_size=7, padding=3)
        )
        self.conv1 = Conv1DBN(in_channels=64*x_filters, out_channels=(16+16+stem_oc)*x_filters, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch_1(x)
        x1 = self.branch_2(x)
        out = torch.cat((x0, x1), dim=1)
        out = self.conv1(out)
        out = out * self.scale + x
        out = self.relu(out)

        return out


# (16+16+stem_oc)*x_filters
class ReductionB(nn.Module):
    def __init__(self, in_channels, x_filters, stem_oc):
        super(ReductionB, self).__init__()
        self.branch_1 = nn.MaxPool1d(kernel_size=2, padding=0)
        self.branch_2 = nn.Sequential(
            Conv1DBN(in_channels=in_channels, out_channels=32*x_filters, kernel_size=1, stride=1),
            Conv1DBN(in_channels=32*x_filters, out_channels=48*x_filters, kernel_size=3, stride=2, padding=1)
        )
        self.branch_3 = nn.Sequential(
            Conv1DBN(in_channels=in_channels, out_channels=32*x_filters, kernel_size=1, stride=1),
            Conv1DBN(in_channels=32*x_filters, out_channels=32*x_filters, kernel_size=3, stride=2, padding=1)
        )
        self.branch_4 = nn.Sequential(
            Conv1DBN(in_channels=in_channels, out_channels=32*x_filters, kernel_size=1, stride=1),
            Conv1DBN(in_channels=32*x_filters, out_channels=32*x_filters, kernel_size=3, stride=1, padding=1),
            Conv1DBN(in_channels=32*x_filters, out_channels=32*x_filters, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x0 = self.branch_1(x)
        x1 = self.branch_2(x)
        x2 = self.branch_3(x)
        x3 = self.branch_4(x)
        out = torch.cat((x0, x1, x2, x3), dim=1)

        return out


class InceptionC(nn.Module):
    def __init__(self, x_filters, scale, stem_oc):
        super(InceptionC, self).__init__()
        self.scale = scale

        self.branch_1 = nn.Sequential(
            Conv1DBN(in_channels=(16+16+stem_oc+32+32+48)*x_filters, out_channels=48*x_filters, kernel_size=1)
        )
        self.branch_2 = nn.Sequential(
            Conv1DBN(in_channels=(16+16+stem_oc+32+32+48)*x_filters, out_channels=48*x_filters, kernel_size=1),
            Conv1DBN(in_channels=48*x_filters, out_channels=48*x_filters, kernel_size=3, padding=1)
        )
        self.conv1 = Conv1DBN(in_channels=96*x_filters, out_channels=(16+16+stem_oc+32+32+48)*x_filters, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch_1(x)
        x1 = self.branch_2(x)
        out = torch.cat((x0, x1), dim=1)
        out = self.conv1(out)
        out = out * self.scale + x
        out = self.relu(out)

        return out


class InceptionBottleneck(nn.Module):
    def __init__(self, x_filters, stem_oc):
        super(InceptionBottleneck, self).__init__()
        # self.branch_1 = NeuralNets.MaxPool1d(kernel_size=2, padding=0)
        # self.branch_2 = NeuralNets.Sequential(
        #     Conv1DBN(in_channels=(16+16+stem_oc+32+32+48)*x_filters, out_channels=32*x_filters, kernel_size=1, stride=1),
        #     Conv1DBN(in_channels=32*x_filters, out_channels=32*x_filters, kernel_size=3, stride=2, padding=1)
        # )
        # self.branch_3 = NeuralNets.Sequential(
        #     Conv1DBN(in_channels=(16+16+stem_oc+32+32+48)*x_filters, out_channels=32*x_filters, kernel_size=1, stride=1),
        #     Conv1DBN(in_channels=32*x_filters, out_channels=32*x_filters, kernel_size=3, stride=2, padding=1)
        # )
        # self.branch_4 = NeuralNets.Sequential(
        #     Conv1DBN(in_channels=(16+16+stem_oc+32+32+48)*x_filters, out_channels=32*x_filters, kernel_size=1, stride=1),
        #     Conv1DBN(in_channels=32*x_filters, out_channels=32*x_filters, kernel_size=3, stride=1, padding=1),
        #     Conv1DBN(in_channels=32*x_filters, out_channels=32*x_filters, kernel_size=3, stride=2, padding=1)
        # )
        self.redB_1 = ReductionB(in_channels=(16+16+stem_oc+32+32+48)*x_filters, x_filters=x_filters, stem_oc=stem_oc)
        self.redB_2 = ReductionB(in_channels=(16+16+stem_oc+32+32+48+32+32+48)*x_filters, x_filters=x_filters, stem_oc=stem_oc)
        # self.redB_3 = ReductionB(in_channels=(16+16+stem_oc+32+32+48+32+32+48+32+32+48)*x_filters, x_filters=x_filters, stem_oc=stem_oc)

        self.bottleneck = Conv1DBN(in_channels=(16+16+stem_oc+32+32+48+32+32+48+32+32+48)*x_filters,
                                   out_channels=BOTTLENECK_CHANNELS, kernel_size=1, padding=0)

        self.redB_decoder_1 = ReductionBDecoder(in_channels=BOTTLENECK_CHANNELS, x_filters=x_filters, stem_oc=stem_oc)
        self.redB_decoder_2 = ReductionBDecoder(in_channels=3*(16+16+stem_oc)*x_filters + BOTTLENECK_CHANNELS,
                                                x_filters=x_filters, stem_oc=stem_oc)
        # self.redB_decoder_3 = ReductionBDecoder(in_channels=3*(16+16+stem_oc)*x_filters+3*(16+16+stem_oc)*x_filters + BOTTLENECK_CHANNELS,
        #                                         x_filters=x_filters, stem_oc=stem_oc)
        # Reduce filters
        self.decoder_fix = Conv1DBN(in_channels=3*(16+16+stem_oc)*x_filters+3*(16+16+stem_oc)*x_filters + BOTTLENECK_CHANNELS,
                                    out_channels=4*(16+16+stem_oc)*x_filters,
                                    kernel_size=1, stride=1, padding=0)
        # self.branch_1_decoder = NeuralNets.ConvTranspose1d(in_channels=2, out_channels=(16+16+stem_oc)*x_filters, kernel_size=2, stride=2,
        #                                            output_padding=0, padding=0)
        # # TODO READ MORE ABOUT OUTPUT PADDING
        # self.branch_2_decoder = NeuralNets.Sequential(
        #     NeuralNets.ConvTranspose1d(in_channels=2, out_channels=48*x_filters, kernel_size=3, stride=2, output_padding=1, padding=1),
        #     Conv1DBN(in_channels=48*x_filters, out_channels=(16+16+stem_oc)*x_filters, kernel_size=1, stride=1, padding=0)
        # )
        # self.branch_3_decoder = NeuralNets.Sequential(
        #     NeuralNets.ConvTranspose1d(in_channels=2, out_channels=32*x_filters, kernel_size=3, stride=2, output_padding=1, padding=1),
        #     Conv1DBN(in_channels=32*x_filters, out_channels=(16+16+stem_oc)*x_filters, kernel_size=1, stride=1, padding=0)
        # )
        # self.branch_4_decoder = NeuralNets.Sequential(
        #     NeuralNets.ConvTranspose1d(in_channels=2, out_channels=32*x_filters, kernel_size=3, stride=2, output_padding=1, padding=1),
        #     Conv1DBN(in_channels=32*x_filters, out_channels=32*x_filters, kernel_size=3, stride=1, padding=1),
        #     Conv1DBN(in_channels=32*x_filters, out_channels=(16+16+stem_oc)*x_filters, kernel_size=1, stride=1, padding=0)
        # )

    def forward(self, x):
        x0 = self.redB_1(x)
        x1 = self.redB_2(x0)
        # x2 = self.redB_3(x1)

        # x4 = torch.cat((x0, x1, x2), dim=1)
        x3 = self.bottleneck(x1)

        x4 = self.redB_decoder_1(x3)
        x5 = self.redB_decoder_2(x4)
        # x6 = self.redB_decoder_3(x5)

        x7 = self.decoder_fix(x5)

        # x10 = torch.cat((x6, x7, x8, x9), dim=1)

        return x7


class InceptionCDecoder(nn.Module):
    def __init__(self, x_filters, scale, stem_oc):
        super(InceptionCDecoder, self).__init__()
        self.scale = scale

        self.branch_1 = nn.Sequential(
            Conv1DBN(in_channels=4*(16+16+stem_oc)*x_filters, out_channels=(16+16+stem_oc+32+32+48)*x_filters, kernel_size=1)
        )
        self.branch_2 = nn.Sequential(
            Conv1DBN(in_channels=4*(16+16+stem_oc)*x_filters, out_channels=48*x_filters, kernel_size=3, padding=1),
            Conv1DBN(in_channels=48*x_filters, out_channels=(16+16+stem_oc+32+32+48)*x_filters, kernel_size=1)
        )
        self.conv1 = Conv1DBN(in_channels=2*(16+16+stem_oc+32+32+48)*x_filters, out_channels=4*(16+16+stem_oc)*x_filters, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch_1(x)
        x1 = self.branch_2(x)
        out = torch.cat((x0, x1), dim=1)
        out = self.conv1(out)
        out = out * self.scale + x
        out = self.relu(out)

        return out

# 4*(16+16+stem_oc)*x_filters
class ReductionBDecoder(nn.Module):
    def __init__(self, in_channels, x_filters, stem_oc):
        super(ReductionBDecoder, self).__init__()
        # NOTE: branch_1 only decodes the maxpool operation
        self.branch_1 = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2,
                                           output_padding=0, padding=0)
        # TODO READ MORE ABOUT OUTPUT PADDING
        self.branch_2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels, out_channels=48*x_filters, kernel_size=3, stride=2, output_padding=1, padding=1),
            Conv1DBN(in_channels=48*x_filters, out_channels=(16+16+stem_oc)*x_filters, kernel_size=1, stride=1, padding=0)
        )
        self.branch_3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels, out_channels=32*x_filters, kernel_size=3, stride=2, output_padding=1, padding=1),
            Conv1DBN(in_channels=32*x_filters, out_channels=(16+16+stem_oc)*x_filters, kernel_size=1, stride=1, padding=0)
        )
        self.branch_4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels, out_channels=32*x_filters, kernel_size=3, stride=2, output_padding=1, padding=1),
            Conv1DBN(in_channels=32*x_filters, out_channels=32*x_filters, kernel_size=3, stride=1, padding=1),
            Conv1DBN(in_channels=32*x_filters, out_channels=(16+16+stem_oc)*x_filters, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x0 = self.branch_1(x)
        x1 = self.branch_2(x)
        x2 = self.branch_3(x)
        x3 = self.branch_4(x)
        out = torch.cat((x0, x1, x2, x3), dim=1)

        return out


class InceptionBDecoder(nn.Module):
    def __init__(self, in_channels, x_filters, scale, stem_oc):
        super(InceptionBDecoder, self).__init__()
        self.scale = scale

        self.branch_1 = nn.Sequential(
            Conv1DBN(in_channels=in_channels, out_channels=(16+16+stem_oc)*x_filters, kernel_size=1)
        )
        self.branch_2 = nn.Sequential(
            Conv1DBN(in_channels=in_channels, out_channels=32*x_filters, kernel_size=7, padding=3),
            Conv1DBN(in_channels=32*x_filters, out_channels=(16+16+stem_oc)*x_filters, kernel_size=1)
        )
        self.conv1 = Conv1DBN(in_channels=2*(16+16+stem_oc)*x_filters, out_channels=in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch_1(x)
        x1 = self.branch_2(x)
        out = torch.cat((x0, x1), dim=1)
        out = self.conv1(out)
        out = out * self.scale + x
        out = self.relu(out)

        return out


class ReductionADecoder(nn.Module):
    def __init__(self, in_channels, x_filters, stem_oc):
        super(ReductionADecoder, self).__init__()
        self.branch_1 = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels,
                                           kernel_size=2, stride=2, output_padding=0, padding=0)
        self.branch_2 = nn.ConvTranspose1d(in_channels=in_channels, out_channels=stem_oc*x_filters,
                                           kernel_size=3, stride=2, output_padding=1, padding=1)
        self.branch_3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels, 16*x_filters, kernel_size=3, stride=2, output_padding=1, padding=1),
            Conv1DBN(in_channels=16*x_filters, out_channels=16*x_filters, kernel_size=3, stride=1, padding=1),
            Conv1DBN(in_channels=16*x_filters, out_channels=stem_oc*x_filters, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch_1(x)
        x1 = self.branch_2(x)
        x2 = self.branch_3(x)
        out = torch.cat((x0, x1, x2), dim=1)

        return out

# temp = ReductionADecoder(2, 32)
# temp.branch_1(t8).shape
# temp.branch_2(t8).shape
# temp.branch_3(t8).shape
class InceptionADecoder(nn.Module):
    def __init__(self, in_channels, x_filters, scale, stem_oc):
        super(InceptionADecoder, self).__init__()
        self.scale = scale
        self.branch_1 = nn.Sequential(
            Conv1DBN(in_channels=in_channels, out_channels=16*x_filters, kernel_size=3, padding=1),
            Conv1DBN(in_channels=16*x_filters, out_channels=stem_oc*x_filters, kernel_size=1)
        )
        self.branch_2 = nn.Sequential(
            Conv1DBN(in_channels=in_channels, out_channels=16*x_filters, kernel_size=5, padding=2),
            Conv1DBN(in_channels=16*x_filters, out_channels=16*x_filters, kernel_size=5, padding=2),
            Conv1DBN(in_channels=16*x_filters, out_channels=stem_oc*x_filters, kernel_size=1)
        )
        self.branch_3 = nn.Sequential(
            Conv1DBN(in_channels=in_channels, out_channels=stem_oc*x_filters, kernel_size=1, padding=0)
        )
        self.conv1 = Conv1DBN(in_channels=3*stem_oc*x_filters, out_channels=in_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch_1(x)
        x1 = self.branch_2(x)
        x2 = self.branch_3(x)
        out = torch.cat((x0, x1, x2), dim=1)
        out = self.conv1(out)
        out = out * self.scale + x
        out = self.relu(out)

        return out


class StemDecoder(nn.Module):
    def __init__(self, in_channels):
        super(StemDecoder, self).__init__()
        self.conv1 = Conv1DBN(in_channels=in_channels, out_channels=4, kernel_size=1)

    def forward(self, x):
        x0 = self.conv1(x)

        return x0


# TODO Set cudnn.benchmark=False
class IncResAutoEncoder(nn.Module):
    def __init__(self, x_filters, my_in_channels=4, stem_oc=32):
        super(IncResAutoEncoder, self).__init__()
        self.Stem = Stem(in_channels=my_in_channels, x_filters=x_filters, stem_oc=stem_oc)
        self.ModuleA = InceptionA(scale=0.3, x_filters=x_filters, stem_oc=stem_oc)
        self.ReductionA = ReductionA(x_filters=x_filters, stem_oc=stem_oc)
        self.ModuleB = InceptionB(scale=0.3, x_filters=x_filters, stem_oc=stem_oc)
        self.ReductionB = ReductionB(in_channels=(16+16+stem_oc)*x_filters,
                                     x_filters=x_filters, stem_oc=stem_oc)
        self.ModuleC = InceptionC(scale=0.3, x_filters=x_filters, stem_oc=stem_oc)
        self.Bottleneck = InceptionBottleneck(x_filters=x_filters, stem_oc=stem_oc)
        self.DecoderC = InceptionCDecoder(x_filters=x_filters, scale=0.3, stem_oc=stem_oc)
        self.DecoderReductionB = ReductionBDecoder(in_channels=4*(16+16+stem_oc)*x_filters,
                                                   x_filters=x_filters, stem_oc=stem_oc)
        self.DecoderB = InceptionBDecoder(in_channels=3*(16+16+stem_oc)*x_filters+4*(16+16+stem_oc)*x_filters,
                                          x_filters=x_filters, scale=0.3, stem_oc=stem_oc)
        self.DecoderReductionA = ReductionADecoder(in_channels=3*(16+16+stem_oc)*x_filters+4*(16+16+stem_oc)*x_filters,
                                                   x_filters=x_filters, stem_oc=stem_oc)
        self.DecoderA = InceptionADecoder(in_channels=2*stem_oc*x_filters+3*(16+16+stem_oc)*x_filters+4*(16+16+stem_oc)*x_filters,
                                          x_filters=x_filters, scale=0.3, stem_oc=stem_oc)
        self.StemDecoder = StemDecoder(in_channels=2*stem_oc*x_filters+3*(16+16+stem_oc)*x_filters+4*(16+16+stem_oc)*x_filters)

        # n_size = self._get_conv_output(input_shape)

    # generate input sample and forward to get shape
    # def _get_conv_output(self, shape):
    #     bs = 1
    #     my_input = Variable(torch.rand(bs, *shape))
    #     output_feat = self._forward_features(my_input)
    #     n_size = output_feat.data.view(bs, -1).size(1)
    #     return n_size
    #
    # def _forward_features(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #     return x
    def forward(self, my_input):
        x0 = self.Stem(my_input)
        x1 = self.ModuleA(x0)
        x2 = self.ReductionA(x1)
        x3 = self.ModuleB(x2)
        x4 = self.ReductionB(x3)
        x5 = self.ModuleC(x4)
        bottleneck = self.Bottleneck(x5)
        x6 = self.DecoderC(bottleneck)
        x7 = self.DecoderReductionB(x6)
        x8 = self.DecoderB(x7)
        x9 = self.DecoderReductionA(x8)
        x10 = self.DecoderA(x9)
        x11 = self.StemDecoder(x10)

        return x11


# autoencoder_model = IncResAutoEncoder(x_filters=2)
# autoencoder_model.Bottleneck.branch_1
class IncResEncoder(nn.Module):
    def __init__(self, autoencoder_model):
        super(IncResEncoder, self).__init__()
        self.features = nn.Sequential(
            autoencoder_model.Stem,
            autoencoder_model.ModuleA,
            autoencoder_model.ReductionA,
            autoencoder_model.ModuleB,
            autoencoder_model.ReductionB,
            autoencoder_model.ModuleC,
        )
        self.branch_1 = autoencoder_model.Bottleneck.branch_1
        self.branch_2 = autoencoder_model.Bottleneck.branch_2
        self.branch_3 = autoencoder_model.Bottleneck.branch_3
        self.branch_4 = autoencoder_model.Bottleneck.branch_4

        self.bottleneck = autoencoder_model.Bottleneck.bottleneck

    def forward(self, x):
        x0 = self.features(x)
        x1 = self.branch_1(x0)
        x2 = self.branch_2(x0)
        x3 = self.branch_3(x0)
        x4 = self.branch_4(x0)

        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        x6 = self.bottleneck(x5)

        return x6


def dna_to_onehot(seqs_list, channels_first=False):
    base_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0],
                 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    list_alt = [list(x) for x in seqs_list]
    # Use a dictionary to perform multiple replacements at once
    # ~4x faster than using list comprehension in succession
    encoded = [[base_dict.get(x, x) for x in y] for y in list_alt]
    if channels_first:
        # NOTE: PYTORCH IS CHANNELS FIRST BY DEFAULT
        # Make sure the lists do not get scrambled during reshaping; simply use transpose
        one_hot_alt = [np.transpose(np.array(x)).reshape((1, 4, len(x))) for x in encoded]
    else:
        # Here, reshaping alone results in intended format
        one_hot_alt = [np.array(x).reshape((1, len(x), 4)) for x in encoded]
    return one_hot_alt


def pad_for_divisibility(onehot_seqs, divisible_by=4, channels_first=False, num_channels=4):
    # TODO: Do we need the following lines if dna_to_onehot reshapes the sequences?
    if channels_first:
        onehot_seqs = onehot_seqs.reshape(onehot_seqs.shape[0], num_channels, onehot_seqs.shape[3])
    else:
        onehot_seqs = onehot_seqs.reshape(onehot_seqs.shape[0], onehot_seqs.shape[2], num_channels)
    # Make sure we don't have any NaN values
    assert not np.isnan(np.sum(onehot_seqs))
    # for i in range(len(one_hot_alt)):
    # Detect odd length input, pad by 1
    if channels_first:
        if (onehot_seqs.shape[2] & 1) == 1:
            onehot_seqs = np.concatenate((onehot_seqs,
                         np.tile(np.array([0]*num_channels),
                                 (onehot_seqs.shape[0], 1)).reshape((onehot_seqs.shape[0], num_channels, 1))), axis=2)

    else:
        if (onehot_seqs.shape[1] & 1) == 1:
            # Duplicate last base or just add a zero vector
            # x_train[i][len(x_train[i])-1]
            onehot_seqs = np.concatenate((onehot_seqs,
                                     np.tile(np.array([0]*num_channels),
                                             (onehot_seqs.shape[0], 1)).reshape((onehot_seqs.shape[0], 1, num_channels))), axis=1)
    # Keep padding with 2 0 vectors until divisible by given denominator
    if channels_first:
        while onehot_seqs.shape[2] % divisible_by != 0:
            onehot_seqs = np.concatenate((onehot_seqs,
                                      np.tile(np.array([0]*num_channels),
                                              (onehot_seqs.shape[0]*2, 1)).reshape((onehot_seqs.shape[0], num_channels, 2))), axis=2)
    else:
        while onehot_seqs.shape[1] % divisible_by != 0:
            onehot_seqs = np.concatenate((onehot_seqs,
                                      np.tile(np.array([0]*num_channels),
                                              (onehot_seqs.shape[0]*2, 1)).reshape((onehot_seqs.shape[0], 2, num_channels))), axis=1)
    # else:
    #     raise Exception('Padding for other denominators not yet implemented!')

    return onehot_seqs


# cur_transcripts = pd.read_csv(PATH+'all_transcript_seqs.txt', engine='c', sep=',', nrows=100)
def transcript_generator(transcript_path, chunk_size, per_yield_size, padding_div=4, channels_first=False):
    # while True:  # keras requires all generators to be infinite
        # base_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0],
        #              'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    for cur_transcripts in pd.read_csv(PATH+transcript_path, engine='c', sep=',', chunksize=chunk_size):
        cur_alt_lists = cur_transcripts['transcript'].tolist()
        # Convert LIST of dna sequences to their onehot encoded form
        cur_onehot = dna_to_onehot(cur_alt_lists, channels_first=channels_first)

        # Group sequences by the same length as batches need to be of the same size
        if channels_first:
            all_lengths = np.unique([x.shape[2] for x in cur_onehot])
        else:
            all_lengths = np.unique([x.shape[1] for x in cur_onehot])
        # Sort, descending
        all_lengths = -np.sort(-all_lengths)
        for length in all_lengths:
            if channels_first:
                cur_sub = np.array([x for x in cur_onehot if x.shape[2] == length])
            else:
                cur_sub = np.array([x for x in cur_onehot if x.shape[1] == length])

            # 3 Reduction by half, thus sequence must be divisible by 2^3
            cur_sub = pad_for_divisibility(cur_sub, divisible_by=padding_div, channels_first=channels_first)
            # TODO: Yield only one sample at a time? How to make sure batches are of the same length?
            # Limit the overall size of each released batch to fit on the VRAM
            cur_size = np.prod(cur_sub.shape)
            if cur_size > per_yield_size:
                num_parts = np.int(np.ceil(cur_size/per_yield_size))
                advance = np.int(np.ceil(cur_sub.shape[0]/num_parts))
                for i in range(0, cur_sub.shape[0], advance):
                    # This automatically ensures only valid indices are kept,
                    # so num_parts can be larger than what remains
                    cur_train = cur_sub[i:i+advance, :]
                    yield cur_train
            else:
                yield cur_sub

        # Garbage collection to reduce possibility of memory leak (?)
        gc.collect()

# temp = transcript_generator('all_transcript_seqs.txt', 100, 1e5, True)
# t = next(temp)
# t.shape

class MyDataset(data.Dataset):
    def __init__(self, transcript_path, chunk_size, per_yield_size, size, padding_div, channels_first=False):
        super(MyDataset, self).__init__()
        self.transcript_path = transcript_path
        self.chunk_size = chunk_size
        self.per_yield_size = per_yield_size
        self.size = size
        self.padding_div = padding_div
        self.transcript_generator = transcript_generator(self.transcript_path, self.chunk_size, self.per_yield_size,
                                                         self.padding_div, channels_first=channels_first)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        batch_x = next(self.transcript_generator)

        return batch_x


# def cross_entropy2d(input, target, weight=None, size_average=True):
#     # input: (n, c, h, w), target: (n, h, w)
#     n, c, h, w = input.size()
#     # log_p: (n, c, h, w)
#     # if LooseVersion(torch.__version__) < LooseVersion('0.3'):
#     #     # ==0.2.X
#     #     log_p = F.log_softmax(input)
#     # else:
#     # >=0.3
#     log_p = F.log_softmax(input, dim=1)
#     # log_p: (n*h*w, c)
#     log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
#     log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
#     log_p = log_p.view(-1, c)
#     # target: (n*h*w,)
#     mask = target >= 0
#     target = target[mask]
#     loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
#     if size_average:
#         loss /= mask.data.sum()
#     return loss
#
#
# def categorical_cross_entropy(input, target, size_average=True):
#     logsoftmax = NeuralNets.LogSoftmax()
#     if size_average:
#         return torch.mean(torch.sum(-target * logsoftmax(input), dim=0))
#     else:
#         return torch.sum(torch.sum(-target * logsoftmax(input), dim=0))

# a simple custom collate function, just to show the idea
def my_collate(batch):
    # data = [item for item in batch]
    # target = [item[1] for item in batch]
    # target = torch.LongTensor(target)
    return torch.tensor(batch)[0]


# def save_checkpoint(state, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, 'model_best.pth.tar')

# a simple custom collate function, just to show the idea
# def my_collate(batch):
#     data = [item[0] for item in batch]
#     target = [item[1] for item in batch]
#     target = torch.LongTensor(target)
#     return [data, target]

if __name__ == '__main__':
    cur_chunk_size = int(sys.argv[1])
    num_epochs = int(sys.argv[2])
    per_yield_size = int(sys.argv[3])
    action = sys.argv[4]
    opt_level = sys.argv[5]
    cur_model = IncResAutoEncoder(x_filters=2)
    # cur_model = cur_model.float()
    # cur_model = cur_model.to(device)
    # print('Loaded model onto', device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cur_model.parameters(), lr=0.005)

    training_set = MyDataset(transcript_path='all_transcript_seqs.txt', chunk_size=cur_chunk_size,
                             per_yield_size=per_yield_size,
                             size=int(100000000000), padding_div=16,
                             channels_first=True)
    training_loader = data.DataLoader(training_set, batch_size=1, num_workers=0, shuffle=False,
                                      collate_fn=my_collate)
    print('Created training loader')

    if action == 'resume':
        model_path = PATH+'incres_cae_pytorch.pth.tar'
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            cur_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_path, checkpoint['epoch']))
            cur_model = cur_model.float()
            cur_model = cur_model.to(device)
            print('(Re)loaded model onto', device)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            print('Loaded optimizer onto', device)
            cur_model, optimizer = amp.initialize(cur_model, optimizer, opt_level=opt_level)
            print('Initialized AMP (Automatic Mixed Precision)')

        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    elif action == 'encode':
        # Load model parameters and optimizer states
        model_path = 'incres_cae_pytorch.pth.tar'
        checkpoint = torch.load(model_path, map_location=device)
        start_epoch = checkpoint['epoch']
        cur_model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # print("=> loaded checkpoint '{}' (epoch {})"
        #       .format(model_path, checkpoint['epoch']))
        # print('Loaded model onto', device)
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(device)
        # print('Loaded optimizer onto', device)

        # dir(cur_model)
        # Load trained model, extract encoder part
        cur_encoder = IncResEncoder(autoencoder_model=cur_model)
        summary(cur_encoder, input_size=(4, 20000), device='cpu')
        cur_encoder = cur_encoder.float()
        cur_encoder = cur_encoder.to(device)

        # cur_encoder, optimizer = amp.initialize(cur_encoder, optimizer, opt_level=opt_level)
        # print('Initialized AMP (Automatic Mixed Precision)')

        all_outputs = []
        # Convert input sequences to features, no gradients required
        total_start = time.time()
        with torch.no_grad():
            batch_start = time.time()
            total_sequences = 0	
            for i, local_batch in enumerate(training_loader, 0):
                # batch_start = time.time()
                # local_batch = iter(training_generator).next()[0]
                # Transfer to GPU
                # local_batch = local_batch
                total_sequences += local_batch.shape[0]
                local_batch = local_batch.float()
                local_batch = local_batch.to(device)

                # forward propagation only (encode)
                # Move to cpu and convert to numpy for easier processing using pickle
                outputs = cur_encoder(local_batch).cpu()
                if outputs.shape[0] > 1:
                    for j in range(outputs.shape[0]):
                        # Append to list separately
                        all_outputs.append(outputs[j].numpy().reshape(1, BOTTLENECK_CHANNELS, outputs.shape[2]))
                else:
                    all_outputs.append(outputs.numpy())
                if i % 1000 == 999:
                    print('Encoded', str(total_sequences), 'sequences in', str(time.time() - batch_start), 'seconds', 'in', str(i+1), 'batches')
                    batch_start = time.time()
                if total_sequences > 411000:
                    print('Finished encoding 412000 sequences')
                    break

        file_path = 'encoded_transcripts.pkl'

        with open(PATH+file_path, 'wb') as f:
            pickle.dump(all_outputs, f)

        print('Encoded all sequences and saved to', file_path, 'in', str(time.time() - total_start))
        sys.exit('Finished encoding sequences!')

    else:
        cur_model = cur_model.float()
        from torchsummary import summary
        summary(cur_model, input_size=(4, 20000), device='cpu')
        cur_model = cur_model.to(device)

        cur_model, optimizer = amp.initialize(cur_model, optimizer, opt_level=opt_level)

    running_loss = 0.0
    # Loop over epochs
    for epoch in range(num_epochs):
        # Training
        start_time = time.time()
        i = 0
        total_sequences = 0
        batch_start = time.time()
        # An epoch is not finished until at least 410482 sequences are seen
        # while total_sequences < 410482:
        training_loader = data.DataLoader(training_set, batch_size=1, num_workers=0, shuffle=False,
                                          collate_fn=my_collate)
        for i, local_batch in enumerate(training_loader, 0):
            # local_batch = iter(training_generator).next()[0]
            # Transfer to GPU
            # local_batch = local_batch
            total_sequences += local_batch.shape[0]
            local_batch = local_batch.float()
            local_batch = local_batch.to(device)
            local_labels = local_batch.argmax(dim=1)
            local_labels = local_labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cur_model(local_batch)
            # loss = categorical_cross_entropy(outputs, local_batch)
            loss = criterion(outputs, local_labels)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            # loss.backward()
            optimizer.step()
            i += 1
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches, average loss
                print('[%d, %5d, %5d] loss: %.6f in %3d s, total %5d s' %
                      (epoch + 1, total_sequences, i + 1, running_loss / 100,
                       time.time() - batch_start, time.time() - start_time))
                running_loss = 0.0
                batch_start = time.time()
                # TODO REMOVE
                # break
        duration = start_time - time.time()
        print('Finished epoch', str(epoch+1), str(duration))
        # torch.save(cur_model.state_dict(), 'inceptionresnet_cae_pytorch.pth.tar')
        torch.save({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': cur_model.state_dict(),
            # 'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, PATH+'incres_cae_pytorch.pth.tar')
        print('Saved checkpoint successfully')


