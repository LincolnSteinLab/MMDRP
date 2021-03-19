import pandas as pd
import numpy as np
import time
import sys

# PATH = "/content/gdrive/My Drive/Drug_Response/RNAi/"
PATH = "~/anaconda3/envs/Drug_Response/Data/RNAi/Train_Data/"
# PATH = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/RNAi/Train_Data/"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn
from torchsummary import summary
from apex import amp

use_cuda = torch.cuda.is_available()
torch.cuda.device_count()
if use_cuda:
    print('CUDA is available!')
device = torch.device("cuda:0" if use_cuda else "cpu")
# Turning off benchmarking makes highly dynamic models faster
cudnn.benchmark = False
cudnn.deterministic = True


# SH_Input = Input(shape=(21, ), name='shRNA_Input')
# # SH_Embedding = Embedding(output_dim=4, input_dim=4+1,
# #                          input_length=21, name='shRNA_Embedding')(SH_Input)
# SH = Conv1D(filters=64, kernel_size=4, padding='valid', activation='relu')(SH_Embedding)
# SH = BatchNormalization()(SH)
# SH = Conv1D(filters=32, kernel_size=2, padding='valid', activation='relu')(SH)
# SH = BatchNormalization()(SH)
#
# sh_encoded = Flatten()(SH)
# SH = RepeatVector(21)(sh_encoded)
# SH = GRU(units=128, activation='tanh', return_sequences=True)(SH)
# SH = GRU(units=128, activation='tanh', return_sequences=True)(SH)
# SH_Output = GRU(units=128, activation='softmax', return_sequences=True, name='shRNA_Output')(SH)
#
# sh_model = Model(inputs=SH_Input, outputs=SH_Output)
# sh_model.compile(optimizer=Adam(lr=0.001),
#                  loss='sparse_categorical_crossentropy')

# cur_tokenizer = Tokenizer(num_words=5, filters="", lower=False, split=' ')
# cur_tokenizer.fit_on_texts('ATCG ')

# Read shRNA Data
# Have more shRNAs than cell lines, thus, for each shRNA, we get the corresponding cell line and its associated data


# sh_file_name = 'all_shrna_seqs.txt'
class ShData(data.Dataset):
    def __init__(self, path, sh_file_name):
        super(ShData, self).__init__()
        all_shrna = pd.read_csv(PATH+sh_file_name, engine='c', sep=',')['shRNA'].tolist()
        all_shrna = dna_to_onehot(all_shrna, channels_first=True)
        all_shrna = np.array(all_shrna)
        self.size = all_shrna.shape[0]
        self.all_shrna = all_shrna.reshape((all_shrna.shape[0], all_shrna.shape[2], all_shrna.shape[3]))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.all_shrna[index]


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


class TransConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, output_padding, stride=1):
        super(TransConvBN, self).__init__()
        self.tran = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=kernel_size, stride=stride, padding=padding,
                                       output_padding=output_padding)
        self.relu = nn.ReLU(inplace=False)
        self.batchnorm = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        x0 = self.tran(x)
        x1 = self.relu(x0)
        x2 = self.batchnorm(x1)

        return x2


class ShAutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(ShEncoder, self).__init__()
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
    def __init__(self, autoencoder_model):
        super(ShEncoder, self).__init__()
        self.features = nn.Sequential(
            autoencoder_model.conv1d_1,
            autoencoder_model.conv1d_2
        )

    def forward(self, x):
        x0 = self.features(x)

        return x0


if __name__ == '__main__':
    cur_batch_size = int(sys.argv[1])
    num_epochs = int(sys.argv[2])
    # per_yield_size = int(sys.argv[2])
    # action = sys.argv[2]
    opt_level = sys.argv[3]

    training_set = ShData(path=PATH, sh_file_name='all_shrna_seqs.txt')
    training_loader = data.DataLoader(training_set, batch_size=cur_batch_size, num_workers=0, shuffle=True)

    cur_model = ShEncoder(input_dim=4)
    cur_model = cur_model.float()
    cur_model = cur_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cur_model.parameters(), lr=0.001)

    cur_model, optimizer = amp.initialize(cur_model, optimizer, opt_level=opt_level)

    running_loss = 0.0

    total_time = time.time()
    for epoch in range(num_epochs):
        # Training
        start_time = time.time()
        i = 0
        batch_start = time.time()
        for i, local_batch in enumerate(training_loader, 0):
            cur_shrna = local_batch
            # if i == 1:
            #     break
            # TODO REMOVE
            # cur_shrna = iter(training_loader).next()
            # Transfer to GPU
            # local_batch = local_batch
            cur_shrna = cur_shrna.float()
            cur_shrna = cur_shrna.to(device)
            cur_labels = cur_shrna.argmax(dim=1)
            cur_labels = cur_labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = cur_model(cur_shrna)

            loss = criterion(output, cur_labels)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            # loss.backward()
            optimizer.step()

            i += 1
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 5 mini-batches, average loss
                print('[%d, %5d] loss: %.6f in %3d seconds' %
                      (epoch + 1, i + 1, running_loss / 100, time.time() - batch_start))
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
    }, 'shrna_embed_pytorch.pth.tar')
    print('Saved checkpoint successfully')
    print('Total time was', str(time.time() - total_time), 'seconds')



