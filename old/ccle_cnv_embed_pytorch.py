# This script contains an auto-encoder for copy number data either with an optional auxiliary tumor classifier.

import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from CustomPytorchLayers import CustomCoder
# To install into conda environment, install pip in conda, then install using the /bin/ path for pip
# from torchsummary import summary
# from apex import amp
# torch.set_flush_denormal(True)
from DRP.src.DataImportModules import CNVData

use_cuda = torch.cuda.is_available()
torch.cuda.device_count()
device = torch.device("cuda:0" if use_cuda else "cpu")
# Turning off benchmarking makes highly dynamic models faster
cudnn.benchmark = True
cudnn.deterministic = True

 # if tied:
        #     # Tie/Mirror the weights
        #     # Initialize weights for each layer
        #     torch.NeuralNets.init.xavier_uniform_(self.encoder1.dense.weight)
        #     self.decoder3.dense.weight = NeuralNets.Parameter(self.encoder1.dense.weight.transpose(0, 1))
        #
        #     torch.NeuralNets.init.xavier_uniform_(self.encoder2.dense.weight)
        #     self.decoder2.dense.weight = NeuralNets.Parameter(self.encoder2.dense.weight.transpose(0, 1))
        #
        #     torch.NeuralNets.init.xavier_uniform_(self.encoder3.dense.weight)
        #     self.decoder1.dense.weight = NeuralNets.Parameter(self.encoder3.dense.weight.transpose(0, 1))

        # self.dense_tum1 = CustomDense(input_size=256, hidden_size=64)
        # self.dense_tum2 = CustomDense(input_size=64, hidden_size=tum_dim)
        # self.softmax_tum = NeuralNets.Softmax(dim=tum_dim)


class CNVAutoEncoder(nn.Module):
    def __init__(self, input_dim, tied=True):
        super(CNVAutoEncoder, self).__init__()
        self.encoder = CustomCoder(input_size=input_dim, first_layer_size=8192, code_layer_size=1024,
                                   num_layers=3, encode=True)
        self.decoder = CustomCoder(input_size=input_dim, first_layer_size=8192, code_layer_size=1024,
                                   num_layers=3, encode=False)

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)

        return dec


if __name__ == '__main__':
    # path = "/content/gdrive/My Drive/Python/RNAi/Train_Data/"
    # PATH = "/u/ftaj/anaconda3/envs/Python/Data/DRP_Training_Data/"
    # PATH = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"

    machine = sys.argv[1]
    num_epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    action = sys.argv[4]
    # opt_level = sys.argv[4]

    if machine == "cluster":
        PATH = "/u/ftaj/anaconda3/envs/Python/Data/DRP_Training_Data/"
    else:
        PATH = "/Data/DRP_Training_Data/"

    cnv_file_name = 'DepMap_20Q2_CopyNumber.csv'
    tum_file_name = 'DepMap_20Q2_Line_Info.csv'

    training_set = CNVData(path=PATH, cnv_file_name='DepMap_20Q2_CopyNumber.hdf')
    training_loader = data.DataLoader(training_set, batch_size=batch_size, num_workers=2, shuffle=True)

    cur_model = CNVAutoEncoder(input_dim=training_set.width(),
                               tied=True)
    cur_model = cur_model.float()
    cur_model = cur_model.to(device)

    last_epoch = 0

    criterion1 = nn.MSELoss()
    # criterion2 = NeuralNets.CrossEntropyLoss()
    optimizer = optim.Adam(cur_model.parameters(), lr=0.001)

    # if action == 'encode':
    #     model_path = 'exp_embed_pytorch.pth.tar'
    #     checkpoint = torch.load(model_path, map_location=device)
    #     start_epoch = checkpoint['epoch']
    #     cur_model.load_state_dict(checkpoint['state_dict'])
    #     # Use the autoencoder's weights
    #     cur_encoder = ExpressionEncoder(expression_autoencoder=cur_model)
    #
    #     # summary(cur_encoder, input_size=(1, 57820), device='cpu')
    #     cur_encoder = cur_encoder.float()
    #     cur_encoder = cur_encoder.to(device)
    #
    #     all_outputs = []
    #     # Convert input sequences to features, no gradients required
    #     total_start = time.time()
    #     with torch.no_grad():
    #         batch_start = time.time()
    #         total_sequences = 0
    #         for i, local_batch in enumerate(training_loader, 0):
    #             # batch_start = time.time()
    #             # local_batch = iter(training_generator).next()[0]
    #             # Transfer to GPU
    #             # local_batch = local_batch
    #             cur_exp = local_batch[0]
    #             total_sequences += cur_exp.shape[0]
    #             cur_exp = cur_exp.float()
    #             cur_exp = cur_exp.to(device)
    #
    #             # forward propagation only (encode)
    #             # Move to cpu and convert to numpy for easier processing using pickle
    #             outputs = cur_encoder(cur_exp).cpu()
    #             if outputs.shape[0] > 1:
    #                 for j in range(outputs.shape[0]):
    #                     # Append to list separately
    #                     all_outputs.append(outputs[j].numpy())
    #             else:
    #                 all_outputs.append(outputs.numpy())
    #             if i % 1000 == 999:
    #                 print('Encoded', str(total_sequences), 'sequences in', str(time.time() - batch_start), 'seconds', 'in', str(i+1), 'batches')
    #                 batch_start = time.time()
    #             # We know we have less than 1200 cell lines
    #             if total_sequences > 1200:
    #                 print('Finished encoding 1165 sequences')
    #                 break
    #
    #     file_path = 'encoded_expression.pkl'
    #
    #     with open(PATH+file_path, 'wb') as f:
    #         pickle.dump(all_outputs, f)
    #
    #     print('Encoded all sequences and saved to', file_path, 'in', str(time.time() - total_start))
    #     sys.exit('Finished encoding sequences!')

    if action == "continue":
        # Load model checkpoint, optimizer state, last epoch and loss
        model_path = 'cnv_embed_pytorch.pth.tar'
        checkpoint = torch.load(model_path, map_location=device)
        cur_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['epoch']
        # loss = checkpoint['loss']

    # cur_model, optimizer = amp.initialize(cur_model, optimizer, opt_level=opt_level)
    running_loss = 0.0
    # Loop over epochs
    # num_epochs = 1
    total_time = time.time()
    for epoch in range(last_epoch, num_epochs):
        # Training
        start_time = time.time()
        i = 0
        batch_start = time.time()
        for i, local_batch in enumerate(training_loader, 0):
            cur_cnv = local_batch
            # cur_tum = local_batch[1]
            # if i == 1:
            #     break

            # local_batch = iter(training_loader).next()[0]
            # Transfer to GPU
            # local_batch = local_batch
            cur_cnv = cur_cnv.float()
            cur_cnv = cur_cnv.to(device)
            # cur_tum = cur_tum.to(device)

            # local_labels = local_batch.argmax(dim=1)
            # local_labels = local_labels.to(device)

            # forward + backward + optimize
            output = cur_model(cur_cnv)
            # loss = categorical_cross_entropy(outputs, local_batch)
            loss = criterion1(output, cur_cnv)
            # loss2 = criterion2(output2, cur_tum)

            # TODO: Can weigh losses differently
            total_loss = loss #+ loss2

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()

            # with amp.scale_loss(total_loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()

            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model. Internally, the parameters of each Module are stored
            # in Tensors with requires_grad=True, so this call will compute gradients for
            # all learnable parameters in the model.
            total_loss.backward()
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
            i += 1
            # print statistics
            # running_loss += total_loss.item()
            # print('Current total loss:', str(total_loss.item()))
            # print('Current exp loss:', str(loss1.item()))
            # print('Current tum loss:', str(loss2.item()))
            if i % 100 == 1:    # print every 5 mini-batches, average loss
                print('[%d, %5d] loss: %.6f in %3d seconds' %
                      (epoch + 1, i + 1, total_loss.item(), time.time() - batch_start))
                # running_loss = 0.0
            batch_start = time.time()
        duration = start_time - time.time()
        print('Finished epoch', str(epoch+1), str(duration))
        # torch.save(cur_model.state_dict(), 'inceptionresnet_cae_pytorch.pth.tar')
    torch.save({
        'epoch': epoch + 1,
        # 'arch': args.arch,
        'state_dict': cur_model.state_dict(),
        # 'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
        # 'loss': total_loss,
    }, 'cnv_embed_pytorch.pth.tar')
    # print('Saved checkpoint successfully')
    print('Total time was', str(time.time() - total_time), 'seconds')
