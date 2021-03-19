# This script contains a model that takes in encoded expression and transcript data to predict the outcome of an
# RNAi experiment, i.e. LFC 

import pandas as pd
import numpy as np
import time
import sys
import gc
import logging
import threading
import pickle
import os
import re

# path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/RNAi/Train_Data/shRNA_by_index/1700001_1800000.txt"
# temp = pd.read_csv(path)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchsummary import summary


# temp_path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/RNAi/Train_Data/shRNA_by_index/1_100000.txt"
# temp_shrna = pd.read_csv(temp_path)
# temp_shrna.iloc[1:10,:]

# import pdb
from apex import amp
# torch.set_flush_denormal(True)

# torch.cuda.set_device
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

# sh_path = 'shRNA_by_line'
# cell_idx_path = 'Cell_Line_Indices'


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
    # if channels_first:
    #     onehot_seqs = onehot_seqs.reshape(onehot_seqs.shape[0], num_channels, onehot_seqs.shape[3])
    # else:
    #     onehot_seqs = onehot_seqs.reshape(onehot_seqs.shape[0], onehot_seqs.shape[2], num_channels)
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

    # if divisible_by == 8:
    #     # Detect odd length input
    #     if (onehot_seqs.shape[1] & 1) == 1:
    #         # Duplicate last base or just add a zero vector
    #         # x_train[i][len(x_train[i])-1]
    #         onehot_seqs = np.concatenate((onehot_seqs,
    #                                  np.tile(np.array([0,0,0,0]),
    #                                          (onehot_seqs.shape[0], 1)).reshape((onehot_seqs.shape[0], 1, 4))), axis=1)
    #         # Since we're doing 3 transposed convolutions, we're dividing by 8, so must add 2 more
    #     if onehot_seqs.shape[1] % 4 == 2:
    #         onehot_seqs = np.concatenate((onehot_seqs,
    #                                   np.tile(np.array([0,0,0,0]),
    #                                           (onehot_seqs.shape[0]*2, 1)).reshape((onehot_seqs.shape[0], 2, 4))), axis=1)
    #     if onehot_seqs.shape[1] % 8 == 4:
    #         onehot_seqs = np.concatenate((onehot_seqs,
    #                                   np.tile(np.array([0,0,0,0]),
    #                                           (onehot_seqs.shape[0]*4, 1)).reshape((onehot_seqs.shape[0], 4, 4))), axis=1)
    # else:
    #     raise Exception('Padding for other denominators not yet implemented!')

    return onehot_seqs


# encoded_transcript_path = 'encoded_transcripts.pkl'
# exp_path = 'encoded_expression.pkl'
# exp_index_path = 'ccle_name_index.txt'
def FullGenerator(encoded_transcript_path, sh_path, exp_path, exp_index_path, cell_idx_path):
    # Read all transcripts
    # all_transcripts = pd.read_csv(PATH+'all_transcript_seqs.txt', engine='c', sep=',')
    # Read all expression data
    # all_expression = pd.read_csv(PATH+exp_path, engine='c', sep=',')
    # Get cell line names
    seq_files = os.listdir(PATH+cell_idx_path)

    # Read encoded expression data
    with (open(PATH+exp_path, "rb")) as openfile:
        encoded_expression = pickle.load(openfile)

    # Read cell line index data
    exp_index = pd.read_csv(PATH+exp_index_path, engine='c', sep=',')

    # Get encoded transcripts
    with (open(PATH+encoded_transcript_path, "rb")) as openfile:
        encoded_transcripts = pickle.load(openfile)

    # For each cell line:
    for cur_file in seq_files:
        cur_line = re.sub('.txt', '', cur_file)
        # Read this cell line's shRNA, sequencing and expression data
        cur_shrna = pd.read_csv(PATH+'/'+sh_path+'/'+cur_line+'.txt', engine='c', sep=',')
        cur_shrna_seqs = dna_to_onehot(cur_shrna['shRNA'].values, channels_first=True)
        cur_shrna_lfcs = cur_shrna['lfc']

        cur_seq_idx = pd.read_csv(PATH+'/'+cell_idx_path+'/'+cur_file, engine='c', sep=',')
        assert cur_seq_idx.shape[0] == 20233
        # Subset all transcripts, don't forget python vs R indexing; subtract 1
        # cur_seq_sub = all_transcripts.iloc[cur_seq_idx.loc[:, 'ref_idx'].values-1, :]
        # Get this line's expression data:
        # Get index, then find in encoded list
        cur_exp_idx = exp_index.loc[exp_index.iloc[:, 1] == cur_line, :]['index']
        cur_exp = encoded_expression[int(cur_exp_idx)-1]
        cur_exp = cur_exp.reshape(256)
        # cur_exp = all_expression.loc[all_expression.iloc[:, 0] == cur_line, :]

        # Get encoded sequence data
        cur_encoded = [encoded_transcripts[i] for i in cur_seq_idx.loc[:, 'ref_idx'].values - 1]
        cur_encoded = np.concatenate(cur_encoded, axis=2)
        cur_encoded = pad_for_divisibility(cur_encoded, divisible_by=2048,
                                           channels_first=True, num_channels=2)
        cur_encoded = cur_encoded.reshape(cur_encoded.shape[1], cur_encoded.shape[2])
        # cur_encoded = torch.Tensor(cur_encoded)
        # cur_encoded = cur_encoded.to(device)

        for i in range(cur_shrna.shape[0]):
            cur_shrna_seq = cur_shrna_seqs[i]
            cur_shrna_seq = cur_shrna_seq.reshape(4, 21)
            cur_shrna_lfc = cur_shrna_lfcs[i]

            yield cur_shrna_seq, cur_exp, cur_encoded, cur_shrna_lfc


class AllData(data.Dataset):
    def __init__(self, encoded_transcript_path, sh_path, exp_path, exp_index_path, cell_idx_path):
        self.generator = FullGenerator(encoded_transcript_path, sh_path, exp_path,
                                       exp_index_path, cell_idx_path)

    def __len__(self):
        return 94453243

    def __getitem__(self, index):
        return next(self.generator)


class Conv1DBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, do_bn=True):
        super(Conv1DBN, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              bias=False, padding=padding, stride=stride)
        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm1d(num_features=out_channels, eps=0.001, momentum=0.1, affine=True,
                                 track_running_stats=False)
        self.do_bn = do_bn

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.relu(x0)
        if self.do_bn:
            return self.bn(x1)
        else:
            return x1


class TransConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, output_padding, stride=1, do_bn=True):
        super(TransConvBN, self).__init__()
        self.tran = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=kernel_size, stride=stride, padding=padding,
                                       output_padding=output_padding)
        self.relu = nn.ReLU(inplace=False)
        self.batchnorm = nn.BatchNorm1d(num_features=out_channels)
        self.do_bn = do_bn

    def forward(self, x):
        x0 = self.tran(x)
        x1 = self.relu(x0)
        if self.do_bn:
            return self.batchnorm(x1)
        else:
            return x1


class DenseBatchNorm(nn.Module):
    def __init__(self, in_features, out_features, do_bn=True):
        super(DenseBatchNorm, self).__init__()
        self.dense = nn.Linear(in_features=in_features, out_features=out_features)
        self.relu = nn.ReLU(inplace=False)
        self.batchnorm = nn.BatchNorm1d(num_features=out_features,
                                        eps=0.001, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.do_bn = do_bn

    def forward(self, x):
        x0 = self.dense(x)
        x1 = self.relu(x0)
        if self.do_bn:
            return self.batchnorm(x1)
        else:
            return x1


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ShAutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(ShAutoEncoder, self).__init__()
        # NeuralNets.Embedding(num_embeddings=4+1, embedding_dim=4)
        self.conv1d_1 = Conv1DBN(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv1d_2 = Conv1DBN(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.tran_1 = TransConvBN(in_channels=32, out_channels=64, kernel_size=3, output_padding=0, padding=1)
        self.tran_2 = TransConvBN(in_channels=64, out_channels=4, kernel_size=3, output_padding=0, padding=1)

    def forward(self, x):
        x0 = self.conv1d_1(x)
        x1 = self.conv1d_2(x0)

        x2 = self.tran_1(x1)
        x3 = self.tran_2(x2)

        return x3


class ShEncoder(nn.Module):
    def __init__(self, sh_autoencoder_model):
        super(ShEncoder, self).__init__()
        self.features = nn.Sequential(
            sh_autoencoder_model.conv1d_1,
            sh_autoencoder_model.conv1d_2
        )

    def forward(self, x):
        x0 = self.features(x)

        return x0


class ReductionModule(nn.Module):
    def __init__(self, in_channels, out_channels, x_filters, do_bn=True):
        super(ReductionModule, self).__init__()
        self.reduc_branch_1 = nn.MaxPool1d(kernel_size=2)
        self.reduc_branch_2 = Conv1DBN(in_channels=in_channels, out_channels=16*x_filters,
                                       kernel_size=3, stride=2, padding=1, do_bn=do_bn)
        self.reduc_branch_3 = nn.Sequential(
            Conv1DBN(in_channels=in_channels, out_channels=8*x_filters, kernel_size=1, stride=1, do_bn=do_bn),
            Conv1DBN(in_channels=8*x_filters, out_channels=16*x_filters, kernel_size=3, stride=1, padding=1, do_bn=do_bn),
            Conv1DBN(in_channels=16*x_filters, out_channels=16*x_filters, kernel_size=3, stride=2, padding=1, do_bn=do_bn)
        )
        self.conv1 = Conv1DBN(in_channels=in_channels+32*x_filters, out_channels=out_channels,
                              kernel_size=1, stride=1, padding=0, do_bn=do_bn)

    def forward(self, x):
        x0 = self.reduc_branch_1(x)
        x1 = self.reduc_branch_2(x)
        x2 = self.reduc_branch_3(x)
        x3 = torch.cat((x0, x1, x2), dim=1)
        x4 = self.conv1(x3)

        return x4


# TODO: MUST GO FROM 8237056 TO A FEW THOUSAND NEURONS
class SequenceHead(nn.Module):
    def __init__(self, x_filters, do_bn=True):
        super(SequenceHead, self).__init__()
        self.reduction = nn.Sequential(
            ReductionModule(in_channels=4, out_channels=4, x_filters=x_filters, do_bn=do_bn),
            ReductionModule(in_channels=4, out_channels=4, x_filters=x_filters, do_bn=do_bn),
            ReductionModule(in_channels=4, out_channels=4, x_filters=x_filters, do_bn=do_bn),
            # ReductionModule(in_channels=4, out_channels=4, x_filters=x_filters, do_bn=do_bn),
            # ReductionModule(in_channels=4, out_channels=4, x_filters=x_filters, do_bn=do_bn),
            # ReductionModule(in_channels=4, out_channels=4, x_filters=x_filters, do_bn=do_bn),
            # ReductionModule(in_channels=4, out_channels=4, x_filters=x_filters, do_bn=do_bn),
            # ReductionModule(in_channels=4, out_channels=4, x_filters=x_filters, do_bn=do_bn),
            # ReductionModule(in_channels=4, out_channels=4, x_filters=x_filters, do_bn=do_bn),
            # ReductionModule(in_channels=4, out_channels=4, x_filters=x_filters, do_bn=do_bn),
            ReductionModule(in_channels=4, out_channels=1, x_filters=x_filters, do_bn=do_bn)
        )
        # Use adaptive pooling to ensure the out put has a fixed size
        self.pool = nn.AdaptiveMaxPool1d(output_size=256)
        self.flatten = Flatten()
        self.dense = nn.Sequential(
            DenseBatchNorm(in_features=256, out_features=128, do_bn=do_bn),
            DenseBatchNorm(in_features=128, out_features=64, do_bn=do_bn),
            DenseBatchNorm(in_features=64, out_features=32, do_bn=do_bn)
        )

    def forward(self, x):
        x0 = self.reduction(x)
        x1 = self.pool(x0)
        x2 = self.flatten(x1)
        x3 = self.dense(x2)

        return x3


summary(SequenceHead(x_filters=1, do_bn=True), input_size=(4, 8237056), batch_size=2)
class shHead(nn.Module):
    def __init__(self, do_bn=True):
        super(shHead, self).__init__()
        self.encoder = nn.Sequential(
            # ShEncoder(sh_autoencoder_model=sh_autoencoder_model),
            Conv1DBN(in_channels=4, out_channels=32, kernel_size=3, padding=1, do_bn=do_bn),
            Conv1DBN(in_channels=32, out_channels=16, kernel_size=3, padding=1, do_bn=do_bn),
            Conv1DBN(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1, do_bn=do_bn),
            Conv1DBN(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0, do_bn=do_bn)
        )
        self.flatten = Flatten()

    def forward(self, x):
        x0 = self.encoder(x)
        x1 = self.flatten(x0)

        return x1


class ExpHead(nn.Module):
    def __init__(self, do_bn=True):
        super(ExpHead, self).__init__()
        self.head = nn.Sequential(
            DenseBatchNorm(in_features=256, out_features=128, do_bn=do_bn),
            DenseBatchNorm(in_features=128, out_features=64, do_bn=do_bn),
            DenseBatchNorm(in_features=64, out_features=32, do_bn=do_bn)
        )

    def forward(self, x):
        x0 = self.head(x)

        return x0


class FullModel(nn.Module):
    def __init__(self, split_gpus=False, x_filters=1, split_size=1, do_bn=True):
        super(FullModel, self).__init__()
        self.sh_head = shHead(do_bn=do_bn)
        self.exp_head = ExpHead(do_bn=do_bn)
        self.sequence_head = SequenceHead(x_filters=x_filters, do_bn=do_bn)
        self.dense = nn.Sequential(
            DenseBatchNorm(in_features=85, out_features=64, do_bn=do_bn),
            DenseBatchNorm(in_features=64, out_features=32, do_bn=do_bn),
            # NOTE: Cannot use ReLu as there are negative LFCs
            nn.Linear(in_features=32, out_features=1)
            # NeuralNets.ReLU
            # CustomDense(input_size=32, hidden_size=1)
        )
        self.split_size = split_size
        self.split_gpus = split_gpus
        if split_gpus:
            self.sh_head.cuda(0)
            self.exp_head.cuda(0)
            self.sequence_head.cuda(1)
            self.dense.cuda(1)

    def forward(self, sh, exp, seq):
        # sh_splits = iter(sh.split(self.split_size, dim=0))
        # exp_splits = iter(exp.split(self.split_size, dim=0))
        # seq_splits = iter(seq.split(self.split_size, dim=0))
        #
        # sh_next = next(sh_splits)
        # sh_prev = self.sh_head(sh_next).cuda(0)
        #
        # exp_next = next(exp_splits)
        # exp_prev = self.exp_head(exp_next).cuda(0)
        #
        # seq_next = next(seq_splits)
        # seq_prev = self.seq_head(seq_next).cuda(0)
        #
        # sh_ret = []
        # exp_ret = []
        # seq_ret = []
        # all_ret = []
        #
        # for sh_next in sh_splits:
        #     # A. s_prev runs on cuda:1
        #     sh_ret = self.sh_head(sh_prev)
        #     exp_ret = self.exp_head(exp_next)
        #     seq_ret = self.seq_head(seq_next)
        #
        #     all_ret.append(torch.cat((sh_ret, exp_ret, seq_ret), dim=1))
        #
        #     # ret.append(self.fc(sh_prev.view(sh_prev.size(0), -1)))
        #
        #     # B. s_next runs on cuda:0, which can run concurrently with A
        #     sh_prev = self.seq1(sh_next).to('cuda:1')
        #     sh_ret = self.sh_head(sh_prev).cuda(1)
        #     exp_ret = self.exp_head(exp_next).cuda(1)
        #     seq_ret = self.seq_head(seq_next).cuda(1)
        #
        #     all_ret.append(torch.cat((sh_ret, exp_ret, seq_ret), dim=1))
        #
        # s_prev = self.seq2(s_prev)
        # ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        x0 = self.sh_head(sh)  # 21 nodes
        x1 = self.exp_head(exp)  # 32 nodes
        x2 = self.sequence_head(seq)  # 32 nodes

        if self.split_gpus:
            # P2P GPU Transfer of sequence head results
            x0 = x0.cuda(1)
            x1 = x1.cuda(1)

        x3 = torch.cat((x0, x1, x2), dim=1)  # 85 nodes
        x3 = x3.cuda(1)
        x4 = self.dense(x3)  # 1 node

        return x4.cuda(1)


if __name__ == '__main__':
    # cur_chunk_size = int(sys.argv[1])
    num_epochs = int(sys.argv[1])
    # batch_size = 2
    batch_size = int(sys.argv[2])
    # per_yield_size = int(sys.argv[2])
    # action = sys.argv[2]
    opt_level = sys.argv[3]
    do_bn = bool(int(sys.argv[4]))
    split_gpus = bool(int(sys.argv[5]))

    # cur_model = cur_model.to('cpu')

    # temp = encoded_expression[0:2]
    # temp = np.stack(temp)
    # temp.shape
    # t1 = cur_model.sh_head(cur_sh)
    # sh_head = shHead(sh_autoencoder)
    # t1 = sh_head(cur_sh)
    # t1 = sh_autoencoder.conv1d_1(cur_sh)
    # t2 = sh_autoencoder.conv1d_2(t1)
    # t1.shape
    # t2.shape
    # cur_model.exp_head(cur_exp)
    # cur_model.sequence_head(cur_seq)
    # r1 = ReductionModule(in_channels=2, out_channels=8, x_filters=1)
    # r2 = ReductionModule(in_channels=8, out_channels=4, x_filters=1)
    # tt = r1(cur_seq)
    # tt2 = r2(tt)
    # cur_model.exp_head.head(torch.Tensor(cur_exp))
    # t1 = cur_model.sequence_head.reduction(torch.Tensor(cur_encoded))
    # t2 = cur_model.sequence_head.pool(t1)
    # t1.shape
    # t2.shape
    # t3 = cur_model.sequence_head.flatten(t2)
    # t3.shape
    # t4 = cur_model.sequence_head.dense(t3)
    # d1 = CustomDense(input_size=256, hidden_size=128)
    # t4 = d1.dense(t3)
    # t5 = d1.relu(t4)
    # t6 = d1.batchnorm_list(t5)
    #
    # t5 = d1(t3)
    # CustomDense(input_size=128, hidden_size=64),
    # CustomDense(input_size=64, hidden_size=32)

    # summary(cur_model, input_size=[(2, 4, 21), (2, 1, 256), (2, 9023195)], device='cpu')

    training_set = AllData(encoded_transcript_path='encoded_transcripts.pkl', sh_path='shRNA_by_line',
                           exp_path='encoded_expression.pkl', exp_index_path='ccle_name_index.txt',
                           cell_idx_path='Cell_Line_Indices')
    training_loader = data.DataLoader(training_set, batch_size=batch_size, num_workers=0, shuffle=True)

    criterion = nn.MSELoss()
    if do_bn:
        print("Model will have batch normalization!")
    # sh_autoencoder = ShAutoEncoder(input_dim=4)
    if use_cuda & split_gpus:
        print("Creating multi-gpu model!")
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        cur_model = FullModel(split_gpus=False, do_bn=do_bn)
        cur_model = nn.DataParallel(cur_model)
        cur_model = cur_model.float()

        optimizer = optim.RMSprop(cur_model.parameters())
        # optimizer = optim.Adam(cur_model.parameters(), lr=0.001)
        # cur_model = cur_model.cuda()
        cur_model, optimizer = amp.initialize(cur_model, optimizer, opt_level=opt_level)
    elif use_cuda:
        cur_model = FullModel(do_bn=do_bn)
        cur_model = cur_model.float()
        cur_model.to(device)
        optimizer = optim.RMSprop(cur_model.parameters())
        # optimizer = optim.Adam(cur_model.parameters(), lr=0.001)
        # Difference with the else statement is whether we use AMP
        cur_model, optimizer = amp.initialize(cur_model, optimizer, opt_level=opt_level)
    else:
        cur_model = FullModel(do_bn=do_bn)
        cur_model = cur_model.float()
        optimizer = optim.RMSprop(cur_model.parameters())
        # optimizer = optim.Adam(cur_model.parameters(), lr=0.001)

    running_loss = 0.0
    # Loop over epochs
    # num_epochs = 1
    total_time = time.time()
    for epoch in range(num_epochs):
        # Training
        start_time = time.time()
        i = 0
        batch_start = time.time()
        for i, local_batch in enumerate(training_loader, 0):
            cur_sh, cur_exp, cur_seq, cur_lfc = local_batch
            # local_batch = iter(training_loader).next()
            # Transfer to GPU
            # local_batch = local_batch
            cur_sh = cur_sh.float()
            cur_exp = cur_exp.float()
            cur_seq = cur_seq.float()
            cur_lfc = cur_lfc.reshape(batch_size, 1)
            cur_lfc = cur_lfc.float()

            if use_cuda & split_gpus:
                cur_sh = cur_sh.cuda(0)
                cur_exp = cur_exp.cuda(0)
                cur_seq = cur_seq.cuda(1)
                cur_lfc = cur_lfc.cuda(1)
            else:
                cur_sh = cur_sh.to(device)
                cur_exp = cur_exp.to(device)
                cur_seq = cur_seq.to(device)
                cur_lfc = cur_lfc.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            output = cur_model(cur_sh, cur_exp, cur_seq)
            print("Calculated output:")
            print(str(output))
            loss = criterion(output, cur_lfc)
            print("Calculated loss")
            print(str(loss.item()))

            if use_cuda & split_gpus:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    # loss.cuda(1)
                    # print("Moved loss to cuda 1")
                    loss.backward()
                    print("Backpropagated loss")
            else:
                loss.backward()

            optimizer.step()
            print("Stepped optimizer")
            i += 1
            # print statistics
            running_loss += loss.item()

            if i % 2 == 1:    # print every 5 mini-batches, average loss
                print('[%d, %5d] loss: %.6f in %3d seconds' %
                      (epoch + 1, i + 1, running_loss / 5, time.time() - batch_start))
                running_loss = 0.0
                batch_start = time.time()
        duration = start_time - time.time()
        print('Finished epoch', str(epoch+1), str(duration))
        torch.save({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': cur_model.state_dict(),
            # 'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, 'full_rnai_model.pth.tar')
        print('Saved checkpoint successfully')
    print('Total time was', str(time.time() - total_time), 'seconds')
