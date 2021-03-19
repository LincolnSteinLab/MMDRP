# This script contains an encoder of expression data either for an auto-encoder or to predict
# secondary labels such as tumor types

import pandas as pd
import sys
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn
# To install into conda environment, install pip in conda, then install using the /bin/ path for pip
from torchsummary import summary
# from apex import amp
import pickle
# torch.set_flush_denormal(True)

use_cuda = torch.cuda.is_available()
torch.cuda.device_count()
if use_cuda:
    print('CUDA is available!')
device = torch.device("cuda:1" if use_cuda else "cpu")
# Turning off benchmarking makes highly dynamic models faster
cudnn.benchmark = False
cudnn.deterministic = True

# path = "/content/gdrive/My Drive/Python/RNAi/Train_Data/"
PATH = "/u/ftaj/anaconda3/envs/Python/Data/DRP_Training_Data/"
# PATH = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"

exp_file_name = 'DepMap_20Q2_Expression.csv'
tum_file_name = 'DepMap_20Q2_Line_Info.csv'
path = PATH
class ExpData(data.Dataset):
    """
    Reads tumor expression as well as tumor classification data from files and
    creates Pandas data frames. The tumor classification data is converted to
    one-hot labels using the LabelEncoder() function from sklearn
    """
    def __init__(self, path, exp_file_name, tum_file_name):
        # Using 'self' makes these elements accessible after instantiation
        self.path = path
        self.exp_file_name = exp_file_name
        self.tum_file_name = tum_file_name
        # Read from file paths
        self.tum_data = pd.read_csv(path+tum_file_name, engine='c', sep=',')
        # For expression data, the first column is the stripped_cell_line_name, which we don't need
        tum_data = pd.read_csv(path+tum_file_name, engine='c', sep=',')
        exp_train = pd.read_csv(path+exp_file_name, engine='c', sep=',')
        tum_train = tum_data['primary_disease'].values
        self.full_train = pd.merge(exp_train, self.tum_data[['DepMap_ID', 'primary_disease']], how='left', on='DepMap_ID')
        # Delete exp_train since it takes too much RAM
        del exp_train
        # full_train = pd.merge(exp_train, tum_data[['DepMap_ID', 'primary_disease']], how='left', on='DepMap_ID')
        self.tum_train = self.full_train['primary_disease'].values
        # Create one-hot labels for tumor classes, based on the order in the table above (for pairing with DepMap_ID)
        self.le = LabelEncoder()
        self.le.fit(self.tum_train)
        self.tum_labels = self.le.transform(self.tum_train)

        # Now separate expression from class data, while discarding stripped cell line name and DepMap ID
        self.full_train = self.full_train.iloc[:, 2:-1]
        # full_train = full_train.iloc[:, 2:-1]

    def __len__(self):
        # Return the number of rows in the training data
        return self.full_train.shape[0]

    def width(self):
        return self.full_train.shape[1]

    def __getitem__(self, idx):
        # skip_idx = np.delete(np.arange(0, 1165), idx)
        # cur_exp = pd.read_csv(self.path+self.exp_file_name, engine='c', sep=',',
        #                       skiprows=skip_idx.tolist()).iloc[:, 1:].values
        # Return paired expression and classification data
        return self.full_train.values[idx], self.tum_labels[idx]


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
        self.noise = self.noise.to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class DenseBatchNorm(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseBatchNorm, self).__init__()
        self.dense = nn.Linear(in_features=in_features, out_features=out_features)
        self.relu = nn.ReLU(inplace=False)
        self.batchnorm = nn.BatchNorm1d(num_features=out_features,
                                        eps=0.001, momentum=0.1, affine=True,
                                        track_running_stats=False)

    def forward(self, x):
        x0 = self.dense(x)
        x1 = self.relu(x0)
        x2 = self.batchnorm(x1)

        return x2


class ExpressionAutoEncoder(nn.Module):
    def __init__(self, input_dim, tum_dim):
        super(ExpressionAutoEncoder, self).__init__()
        self.noise = GaussianNoise(sigma=1, is_relative_detach=True)
        self.dense1 = DenseBatchNorm(in_features=input_dim, out_features=2048)
        self.dense2 = DenseBatchNorm(in_features=2048, out_features=512)
        self.dense3 = DenseBatchNorm(in_features=512, out_features=256)
        self.dense4 = DenseBatchNorm(in_features=256, out_features=512)
        self.dense5 = DenseBatchNorm(in_features=512, out_features=2048)
        self.dense6 = DenseBatchNorm(in_features=2048, out_features=input_dim)

        self.dense_tum1 = DenseBatchNorm(in_features=256, out_features=64)
        self.dense_tum2 = DenseBatchNorm(in_features=64, out_features=tum_dim)
        # self.softmax_tum = NeuralNets.Softmax(dim=tum_dim)

    def forward(self, x):
        # Expression Encoder
        # exp0 = self.noise(x)
        exp1 = self.dense1(x)
        exp2 = self.dense2(exp1)
        exp_encoded = self.dense3(exp2)
        exp4 = self.dense4(exp_encoded)
        exp5 = self.dense5(exp4)
        exp6 = self.dense6(exp5)

        # Tumor type predictor
        # tum0 = self.dense_tum1(exp_encoded)
        # tum1 = self.dense_tum2(tum0)
        # tum1 = self.softmax_tum(tum0)

        return exp6


class ExpressionEncoder(nn.Module):
    def __init__(self, expression_autoencoder):
        super(ExpressionEncoder, self).__init__()
        self.encoder = nn.Sequential(
            expression_autoencoder.noise,
            expression_autoencoder.dense1,
            expression_autoencoder.dense2,
            expression_autoencoder.dense3
        )

    def forward(self, x):
        x0 = self.encoder(x)

        return x0

# EXP_Input = Input(shape=(57820, ), name='EXP_Input')
# EXP = GaussianNoise(stddev=1)(EXP_Input)
# EXP = Dense(units=2048, activation='relu')(EXP)
# EXP = BatchNormalization()(EXP)
# EXP = Dense(units=512, activation='relu')(EXP)
# EXP = BatchNormalization()(EXP)
# EXP = Dense(units=256, activation='relu')(EXP)
# exp_encoded = BatchNormalization(name='EXP_Embedding')(EXP)
#
# EXP = Dense(units=512, activation='relu')(exp_encoded)
# EXP = BatchNormalization()(EXP)
# EXP = Dense(units=2048, activation='relu')(EXP)
# EXP = BatchNormalization()(EXP)
# EXP_Output = Dense(units=57820, activation='relu', name='EXP_Output')(EXP)
#
# TUM = Dense(64, activation='relu', name='TUM_Input')(exp_encoded)
# TUM = BatchNormalization()(TUM)
# TUM_Output = Dense(33, activation='softmax', name='TUM_Output')(TUM)
# exp_model = Model(inputs=EXP_Input, outputs=[EXP_Output, TUM_Output])

# batch_size=4
if __name__ == '__main__':
    # cur_chunk_size = int(sys.argv[1])
    num_epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    # per_yield_size = int(sys.argv[2])
    action = sys.argv[3]
    # opt_level = sys.argv[4]

    training_set = ExpData(path=PATH, exp_file_name='DepMap_20Q2_Expression.csv',
                           tum_file_name='DepMap_20Q2_Line_Info.csv')
    training_loader = data.DataLoader(training_set, batch_size=batch_size, num_workers=0, shuffle=True)

    cur_model = ExpressionAutoEncoder(input_dim=training_set.width(),
                                      tum_dim=training_set.le.classes_.shape[0])

    cur_model = cur_model.float()
    cur_model = cur_model.to(device)
    # print('Loaded model onto', device)
    # summary(cur_model, input_size=(1, 57820), batch_size=128)
    #
    # x0 = cur_model.noise(cur_exp)
    # x0 = cur_model.dense1(cur_exp)
    # x1 = cur_model.dense2(x0)
    # x2 = cur_model.dense3(x1)
    # x3 = cur_model.dense4(x2)
    # x4 = cur_model.dense5(x3)
    # x5 = cur_model.dense6(x4)
    # x5.shape
    #
    # t0 = cur_model.dense_tum1(x2)
    # t1 = cur_model.dense_tum2(t0)
    # t0.shape
    # t1.shape

    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cur_model.parameters(), lr=0.001)

    if action == 'encode':
        model_path = 'exp_embed_pytorch.pth.tar'
        checkpoint = torch.load(model_path, map_location=device)
        start_epoch = checkpoint['epoch']
        cur_model.load_state_dict(checkpoint['state_dict'])
        # Use the autoencoder's weights
        cur_encoder = ExpressionEncoder(expression_autoencoder=cur_model)

        # summary(cur_encoder, input_size=(1, 57820), device='cpu')
        cur_encoder = cur_encoder.float()
        cur_encoder = cur_encoder.to(device)

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
                cur_exp = local_batch[0]
                total_sequences += cur_exp.shape[0]
                cur_exp = cur_exp.float()
                cur_exp = cur_exp.to(device)

                # forward propagation only (encode)
                # Move to cpu and convert to numpy for easier processing using pickle
                outputs = cur_encoder(cur_exp).cpu()
                if outputs.shape[0] > 1:
                    for j in range(outputs.shape[0]):
                        # Append to list separately
                        all_outputs.append(outputs[j].numpy())
                else:
                    all_outputs.append(outputs.numpy())
                if i % 1000 == 999:
                    print('Encoded', str(total_sequences), 'sequences in', str(time.time() - batch_start), 'seconds', 'in', str(i+1), 'batches')
                    batch_start = time.time()
                # We know we have less than 1200 cell lines
                if total_sequences > 1200:
                    print('Finished encoding 1165 sequences')
                    break

        file_path = 'encoded_expression.pkl'

        with open(PATH+file_path, 'wb') as f:
            pickle.dump(all_outputs, f)

        print('Encoded all sequences and saved to', file_path, 'in', str(time.time() - total_start))
        sys.exit('Finished encoding sequences!')

    # cur_model, optimizer = amp.initialize(cur_model, optimizer, opt_level=opt_level)
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
            cur_exp = local_batch[0]
            # cur_tum = local_batch[1]
            # if i == 1:
            #     break

            # local_batch = iter(training_loader).next()[0]
            # Transfer to GPU
            # local_batch = local_batch
            cur_exp = cur_exp.float()
            cur_exp = cur_exp.to(device)
            # cur_tum = cur_tum.to(device)

            # local_labels = local_batch.argmax(dim=1)
            # local_labels = local_labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output1, output2 = cur_model(cur_exp)
            # loss = categorical_cross_entropy(outputs, local_batch)
            loss1 = criterion1(output1, cur_exp)
            # loss2 = criterion2(output2, cur_tum)

            # TODO: Can weigh losses differently
            total_loss = loss1

            # with amp.scale_loss(total_loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()

            total_loss.backward()
            optimizer.step()
            i += 1
            # print statistics
            running_loss += total_loss.item()
            # print('Current total loss:', str(total_loss.item()))
            # print('Current exp loss:', str(loss1.item()))
            # print('Current tum loss:', str(loss2.item()))
            if i % 5 == 1:    # print every 5 mini-batches, average loss
                print('[%d, %5d] loss: %.6f in %3d seconds' %
                      (epoch + 1, i + 1, running_loss / 5, time.time() - batch_start))
                running_loss = 0.0
                batch_start = time.time()
        duration = start_time - time.time()
        print('Finished epoch', str(epoch+1), 'in', str(duration), 'seconds')
        # torch.save(cur_model.state_dict(), 'inceptionresnet_cae_pytorch.pth.tar')
        torch.save({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': cur_model.state_dict(),
            # 'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, 'exp_embed_pytorch.pth.tar')
        print('Saved checkpoint successfully')
    print('Total time was', str(time.time() - total_time), 'seconds')

