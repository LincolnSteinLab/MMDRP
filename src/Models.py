from typing import List

import torch
import torch.nn as nn
# from torch.autograd import Variable
from torch.autograd.grad_mode import F
from torch.nn import Parameter
from torch.nn.init import xavier_normal_, kaiming_normal_
from torch_geometric.nn import GCNConv

from CustomPytorchLayers import CustomCoder, CustomDense, CustomCNN


# activation_dict = NeuralNets.ModuleDict({
#                 'lrelu': NeuralNets.LeakyReLU(),
#                 'prelu': NeuralNets.PReLU(),
#                 'relu': NeuralNets.ReLU(),
#                 'none': NeuralNets.Identity()
#         })

# from torchinfo import summary
# temp = OmicAutoEncoder(input_dim=19100, first_layer_size=512,
#                                 code_layer_size=2048, num_layers=5,
#                                 batchnorm_list=[False]*5)
#
# summary(temp, input_size=(19100,), batch_dim=None)
# temp = MultiHeadCNNAutoEncoder(num_branch=3, code_layer_size=None, kernel_size_list=[3, 5, 11], in_channels=1,
#                       out_channels_list=[1, 1, 1], batchnorm_list=[False, False, False], stride_list=[2, 2, 2],
#                       act_fun_list=NeuralNets.ReLU(), dropout_list=[0., 0., 0.], encode=None)


# temp = MultiHeadCNNAutoEncoder(input_size=27000, first_layer_size=8196, code_layer_size=1024, code_layer=True, num_layers=4, batchnorm_list=[True]*4,
#                    act_fun_list=[NeuralNets.ReLU(), NeuralNets.ReLU(),NeuralNets.ReLU(),NeuralNets.ReLU()], encode=True)
# temp = MultiHeadCNNAutoEncoder(num_branch=3, code_layer_size=None, kernel_size_list=[3, 5, 11], in_channels=1,
#                       out_channels_list=[1, 1, 1], batchnorm_list=[False, False, False], stride_list=[2, 2, 2],
#                       act_fun_list=NeuralNets.ReLU(), dropout_list=[0., 0., 0.], encode=None)
# torchsummary.summary(temp, input_size=(1, 512))
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed


class DNNAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, first_layer_size: int, code_layer_size: int, num_layers: int,
                 batchnorm_list=None,
                 act_fun_list=None, dropout_list=0.0, name=""):
        super(DNNAutoEncoder, self).__init__()
        # Determine batch norm as a whole for this DNN
        if batchnorm_list is None or batchnorm_list is False:
            batchnorm_list = [False] * num_layers
        elif batchnorm_list is True:
            batchnorm_list = [True] * num_layers
        else:
            Warning("Incorrect batchnorm_list argument, defaulting to False for all layers")
            batchnorm_list = [False] * num_layers

        if dropout_list is None or dropout_list == "none" or dropout_list == 0.0:
            dropout_list = [0.0] * num_layers
        elif isinstance(dropout_list, float):
            dropout_list = [dropout_list] * num_layers

        # Make Encoder including the code layer itself
        self.encoder = CustomCoder(input_size=input_dim, first_layer_size=first_layer_size,
                                   code_layer_size=code_layer_size, num_layers=num_layers, encode=True,
                                   code_layer=True, act_fun_list=act_fun_list, batchnorm_list=batchnorm_list,
                                   dropout_list=dropout_list, name=name)
        # Make Decoder excluding the code layer
        self.decoder = CustomCoder(input_size=input_dim, first_layer_size=first_layer_size,
                                   code_layer_size=code_layer_size, num_layers=num_layers, encode=False,
                                   code_layer=False, act_fun_list=act_fun_list, batchnorm_list=batchnorm_list,
                                   dropout_list=dropout_list, name=name)

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)

        return dec


def calculate_padding(input_size, stride, kernel_size):
    p = ((input_size - 1) * stride - input_size + kernel_size) // 2
    return p


class MultiHeadCNNAutoEncoder(nn.Module):
    """
    This class implements a CNN autoencoder (single-layer in encoder) that has multiple heads each with a different filter size that
    scan the same layers, similar to the Inception CNN architecture.
    """

    def __init__(self, input_width: int, num_branch: int = 3, stride: int = None, in_channels: int = 1,
                 out_channels_list: [] = None, batchnorm_list: [] = None, act_fun_list: [] = None,
                 dropout_list: [] = None):
        super(MultiHeadCNNAutoEncoder, self).__init__()

        # Formula for creating odd-sized kernels from 3 based on given num_branch
        kernel_size_list = list(range(3, 3 + (num_branch * 2), 2))

        self.input_width = input_width

        # Check layer-wise parameters
        if batchnorm_list is None or batchnorm_list is False:
            batchnorm_list = [False] * num_branch
        else:
            assert len(batchnorm_list) == num_branch, "batchnorm_list length should be the same as num_branch"
        self.batchnorm_list = batchnorm_list

        if act_fun_list is None:
            act_fun_list = [None] * num_branch
        else:
            assert len(act_fun_list) == num_branch, "act_fun_list length should be the same as num_branch"
        self.act_fun_list = act_fun_list

        if dropout_list is None:
            dropout_list = [0.] * num_branch
        else:
            assert len(dropout_list) == num_branch, "dropout_list length should be the same as num_branch"
        self.dropout_list = dropout_list

        if out_channels_list is None:
            out_channels_list = [64] * num_branch
        self.out_channels_list = out_channels_list

        if stride is None:
            stride_list = [1] * num_branch
        else:
            stride_list = [stride] * num_branch
        self.stride_list = stride_list

        # Calculate padding for each branch
        self.padding_list = [calculate_padding(self.input_width, stride_list[i], kernel_size_list[i]) for i in
                             range(num_branch)]

        self.encoder_branches = nn.ModuleList([
            CustomCNN(in_channels, out_channels_list[i],
                      kernel_size_list[i], act_fun=act_fun_list[i], batch_norm=batchnorm_list[i], stride=stride_list[i],
                      dropout=dropout_list[i], padding=self.padding_list[i], name="cnn_encoder_" + str(i)) for i in
            range(num_branch)
        ])
        # Stack and reduce channels from each branch down to 1 using 1x1 convolution, to concatenate later
        self.code_layer_in = nn.ModuleList([
            CustomCNN(out_channels_list[i], out_channels=1, kernel_size=1, stride=1, act_fun=act_fun_list[0],
                      # padding=self.padding_list[i],
                      name="code_in_" + str(i)) for _ in
            range(num_branch) for i in range(num_branch)
        ])
        self.code_layer_out = nn.ModuleList([
            CustomCNN(1, out_channels=out_channels_list[i], kernel_size=1, stride=1, act_fun=act_fun_list[0],
                      # padding=self.padding_list[i],
                      name="code_out_" + str(i)) for _ in
            range(num_branch) for i in range(num_branch)
        ])

        # Mirror everything for the decoder
        out_channels_list_rev = out_channels_list[::-1]
        act_fun_list_rev = act_fun_list[::-1]
        batchnorm_list_rev = batchnorm_list[::-1]
        dropout_list_rev = dropout_list[::-1]
        kernel_size_list_rev = kernel_size_list[::-1]
        padding_list_rev = self.padding_list[::-1]
        self.decoder_branches = nn.ModuleList([
            CustomCNN(in_channels=out_channels_list_rev[i], out_channels=1, transpose=True, stride=stride_list[i],
                      kernel_size=kernel_size_list_rev[i], act_fun=act_fun_list_rev[i],
                      batch_norm=batchnorm_list_rev[i],
                      dropout=dropout_list_rev[i],
                      padding=padding_list_rev[i],
                      name="cnn_decoder_" + str(i)) for i in range(num_branch)
        ])

        self.depth_reduce = CustomCNN(in_channels=num_branch, out_channels=1, kernel_size=1,
                                      act_fun=self.act_fun_list[0],
                                      stride=1, name="depth_reduce")

    def forward(self, x):
        # Apply each head (branch) to the input
        y = [self.encoder_branches[i](x) for i in range(len(self.encoder_branches))]
        # 1x1 convolution on each multichannel output -> 1 channel
        y = [self.code_layer_in[i](y[i]) for i in range(len(y))]
        # Concatenate 1-channel on the same row, this is the true code layer
        # y = torch.cat(y, dim=1)
        # 1x1 convolution on each 1 channel output -> multichannel
        y = [self.code_layer_out[i](y[i]) for i in range(len(y))]
        # Apply each head on the multi-channel output of 1x1 conv -> 1 channel
        y = [self.decoder_branches[i](y[i]) for i in range(len(self.decoder_branches))]
        # Depth concatenation
        y = torch.cat(y, dim=1)
        # Depth reduction
        y = self.depth_reduce(y)
        # Each of these 1 channel outputs should be the same as the input
        return y

        # if encode is True:
        #     # np.linspace allows for the creation of an array with equidistant numbers from start to end
        #     if num_branch is None:
        #         # TODO how does this work in the context of CNN
        #         num_branch = np.linspace(first_layer_size, code_layer_size, num_layers).astype(int)
        #
        #     # Create and add CNN layers TODO
        #     self.coder = NeuralNets.ModuleList([CustomCNN(in_channels_list[0], out_channels_list[0],
        #                                           kernel_size_list[0], act_fun_list=act_fun_list[0],
        #                                           batch_norm=batchnorm_list[0], dropout_list=dropout_list[0])])
        #     self.coder.extend([CustomCNN(in_channels_list[i], out_channels_list[i],
        #                                  kernel_size_list[i], act_fun_list=act_fun_list[i], batch_norm=batchnorm_list[i],
        #                                  dropout_list=dropout_list[i]) for i in range(len(num_branch) - 1)])
        #
        # else:
        #     self.coder = NeuralNets.ModuleList([CustomCNN(in_channels_list[i], out_channels_list[i],
        #                                           kernel_size_list[i], act_fun_list=act_fun_list[i],
        #                                           batch_norm=batchnorm_list[i],
        #                                           dropout_list=dropout_list[i]) for i in range(len(num_branch) - 1)])
        #     self.coder.extend([CustomCNN(in_channels_list[0], out_channels_list[0],
        #                                  kernel_size_list[0], act_fun_list=act_fun_list[0],
        #                                  batch_norm=batchnorm_list[0], dropout_list=dropout_list[0])])


class DrugResponsePredictor(nn.Module):
    """
    This class takes in an arbitrary number of encoder modules with pre-assigned weights, freezes them and
    connects them to an MLP for drug response prediction.
    Optional: User can determine the GPU each module will be trained on.
    Optional: User can convert the encoder modules to half (FP16) precision
    """

    def __init__(self,
                 layer_sizes=None,
                 encoder_requires_grad: bool = False,
                 gnn_info: List = (False, None),
                 act_fun_list=None,
                 batchnorm_list=None, dropout_list=0.0, merge_method: str = "concat", *encoders):

        super(DrugResponsePredictor, self).__init__()
        self.merge_method = merge_method
        self.gnn_info = gnn_info

        if self.merge_method not in ["concat", "sum"]:
            print("Current given merge method is:", self.merge_method)
            exit("Current implemented merge methods are: concat, sum")

        # if gpu_locs is not None:
        #     assert len(gpu_locs) == len(encoders), "gpu_locs:" + str(len(gpu_locs)) + ", encoders:" + str(len(encoders))
        if batchnorm_list is None or batchnorm_list is False:
            batchnorm_list = [False] * len(layer_sizes)
        elif batchnorm_list is True:
            batchnorm_list = [True] * len(layer_sizes)
        else:
            Warning("Incorrect batchnorm_list argument, defaulting to False")
            batchnorm_list = [False] * len(layer_sizes)

        if dropout_list is None or dropout_list == "none" or dropout_list == 0.0:
            dropout_list = [0.0] * len(layer_sizes)

        if act_fun_list is None:
            act_fun_list = [None] * len(layer_sizes)

        self.encoders = encoders
        self.len_encoder_list = len(self.encoders)
        # self.gpu_locs = gpu_locs
        self.layer_sizes = layer_sizes
        self.encoder_requires_grad = encoder_requires_grad

        # Ensure encoders' weights are frozen (for batchnorm_list to work properly)
        for encoder in self.encoders:
            for param in encoder.parameters():
                param.requires_grad = self.encoder_requires_grad

        if self.encoder_requires_grad is True:
            self.encoders = [encoder.train() for encoder in self.encoders]
        else:
            self.encoders = [encoder.eval() for encoder in self.encoders]

        # Transfer each module to requested GPU
        # if self.gpu_locs is not None:
            # print("Transferring encoder modules to GPU...")
            # self.encoders = [encoder.to("cuda:" + str(self.gpu_locs[i]), non_blocking=True) for i, encoder in
            #                  zip(range(len(self.gpu_locs)), self.encoders)]

        # Generate the outputs using a random tensor (similar to a dry-run)
        if gnn_info[0] is False:
            encoder_outputs = [
                encoder(torch.FloatTensor(2, encoder.input_size).to("cpu")) for
                encoder in self.encoders]
            # The size of the input to the first layer is the sum of the sizes of the outputs of the encoder layers
            self.encoder_out_sizes = [output.shape[1] for output in encoder_outputs]

        else:
            # GNN output size should be given in gnn_info, otherwise it's difficult to do a dry run
            encoder_outputs = [
                encoder(torch.FloatTensor(2, encoder.input_size).to("cpu")) for
                encoder in self.encoders[1:]]
            self.encoder_out_sizes = [gnn_info[1]] + [output.shape[1] for output in encoder_outputs]
            # print("self.encoder_out_sizes:", self.encoder_out_sizes)
        # else:
        #     raise NotImplementedError
        #     # print("Transferring encoder modules to CPU...")
        #     # Transfer everything to default GPU: 0
        #     self.encoders = [encoder.to("cpu") for encoder in self.encoders]
        #     encoder_outputs = [encoder(torch.Tensor(2, encoder.input_size).to("cpu")) for encoder in
        #                        self.encoders]

        # Convert Python list to torch.ModuleList so that torch can 'see' the parameters of each module
        self.encoders = nn.ModuleList(self.encoders)

        # The size of the input to the first layer is the sum of the sizes of the outputs of the encoder layers
        # self.encoder_out_sizes = [output.shape[1] for output in encoder_outputs]
        if self.merge_method == "sum" or self.merge_method == "multiply":
            assert len(set(self.encoder_out_sizes)) == 1, \
                "Encoder outputs have different sizes, cannot sum or multiply! Have: " + str(self.encoder_out_sizes)

        if self.merge_method == 'concat':
            self.drp_input_size = sum(self.encoder_out_sizes)
            self.layer_sizes = [self.drp_input_size] + layer_sizes
        else:
            self.drp_input_size = encoder_outputs[0].shape[1]  # code size is the same for all encoders
            self.layer_sizes = [self.drp_input_size] + layer_sizes

        print("Size of DRP input is:", self.drp_input_size)

        drp_layers = [CustomDense(input_size=self.layer_sizes[i],
                                  hidden_size=self.layer_sizes[i + 1],
                                  act_fun=act_fun_list[i],
                                  batch_norm=batchnorm_list[i],
                                  dropout=dropout_list[i],
                                  name="drp_" + str(i)) for i in
                      range(len(self.layer_sizes) - 1)]
        self.drp_module = nn.Sequential(*drp_layers)
        # if self.gpu_locs is not None:
        #     TODO: setup model-parallel multi-gpu here
            # Note: NeuralNets.ModuleList modules are not connected together in any way, so cannot use instead of NeuralNets.Sequential!
        # else:
            # drp_layers = [CustomDense(input_size=self.layer_sizes[i],
            #                           hidden_size=self.layer_sizes[i + 1]) for i in
            #               range(len(self.layer_sizes) - 1)]
            # self.drp_module = nn.Sequential(*drp_layers).to("cpu")

    def forward(self, *inputs):
        # TODO this "is.instance" might cause a slow down
        # We have an arbitrarily sized list of inputs matching the number of given encoders
        if not len(inputs) == self.len_encoder_list:
            print("Number of inputs should match the number of encoders and must be in the same order, got")
            raise AssertionError("len(inputs):", len(inputs), "!=", "len_encoder_list", self.len_encoder_list)

        # print("encoder input data sizes are:", print(inputs[0].shape), print(inputs[1].shape))
        # print(self.encoders[0])
        # print(self.encoders[1])

        if self.gnn_info[0] is False:
            encoder_outs = [self.encoders[i](inputs[i]) for i in
                            range(len(self.encoders))]
            # for i in range(len(self.encoders)):
            #     print("i in self.encoders:", i)
            #     print("shape of inputs[i]:", inputs[i].shape)
            #     print("cur encoder info:")
            #     for name, param in self.encoders[i].named_parameters():
            #         # if param.requires_grad:
            #         print("Layer Name:", name, "\nLayer Shape:", param.data.shape)
            #         self.encoders[i](inputs[i])

        else:
            # GNN input must be handled differently
            gnn_output = self.encoders[0](inputs[0].x, inputs[0].edge_index, inputs[0].edge_attr, inputs[0].batch)

            # encoder_outs = [gnn_output] + [self.encoders[i](inputs[0].omic_data[i]) for i in
            #                                range(len(inputs[0].omic_data))]
            encoder_outs = [gnn_output] + [self.encoders[i](inputs[1][i - 1]) for i in
                                           range(1, len(self.encoders))]

        # print("encoder output sizes are:", print(encoder_outs[0].shape), print(encoder_outs[1].shape))

        # print("Drug output shape is:", encoder_outs[0].shape)

        if self.merge_method == "sum":
            drp_input = torch.sum(torch.stack(encoder_outs, dim=1), dim=1)
        else:
            drp_input = torch.cat(encoder_outs, 1)

        # print("DRP Input shape is:", drp_input.shape)
        # exit()
        # if self.merge_method == "multiply":
        #     drp_input = torch.sum(torch.vstack(encoder_outs), dim=0)
        # Concatenate the outputs by rows

        return self.drp_module(drp_input)


class FullDrugResponsePredictorTest(nn.Module):
    """
    This class is only for testing with the torchinfo package.
    """

    def __init__(self, layer_sizes=None, gpu_locs=None, encoder_requires_grad=False, *encoders):
        super(FullDrugResponsePredictorTest, self).__init__()
        if gpu_locs is not None:
            assert len(gpu_locs) == len(encoders)
        self.encoders = encoders
        self.len_encoder_list = len(self.encoders)
        self.gpu_locs = gpu_locs
        self.layer_sizes = layer_sizes
        self.encoder_requires_grad = encoder_requires_grad

        # Ensure encoders' weights are frozen (for batchnorm_list to work properly)
        for encoder in self.encoders:
            for param in encoder.parameters():
                param.requires_grad = self.encoder_requires_grad
        if self.encoder_requires_grad is True:
            self.encoders = [encoder.train() for encoder in self.encoders]
        else:
            self.encoders = [encoder.eval() for encoder in self.encoders]

        # Transfer each module to requested GPU
        if self.gpu_locs is not None:
            # print("Transferring encoder modules to GPU...")
            self.encoders = [encoder.to("cuda:" + str(self.gpu_locs[i]), non_blocking=True) for i, encoder in
                             zip(range(len(self.gpu_locs)), self.encoders)]
            # Generate the outputs using a random tensor (a kind of dry-run)
            encoder_outputs = [
                encoder(torch.FloatTensor(2, encoder.input_size).to("cuda:" + str(self.gpu_locs[i]),
                                                                    non_blocking=True)) for
                i, encoder in zip(range(len(self.gpu_locs)), self.encoders)]
        else:
            # print("Transferring encoder modules to CPU...")
            # TODO: What if user wants to use CPU?
            # Transfer everything to default GPU: 0
            self.encoders = [encoder.to("cpu") for encoder in self.encoders]
            # Generate the outputs using a random tensor (a kind of dry-run)
            encoder_outputs = [encoder(torch.Tensor(2, encoder.input_size).to("cpu")) for encoder in
                               self.encoders]

        # Convert Python list to torch.ModuleList so that torch can 'see' the parameters of each module
        # self.encoder_list = NeuralNets.ModuleList(self.encoder_list)

        # The size of the input to the first layer is the sum of the sizes of the outputs of the encoder layers
        self.encoder_out_sizes = [output.shape[1] for output in encoder_outputs]
        self.drp_input_size = sum(self.encoder_out_sizes)
        print("Sum width of the encoder outputs is:", self.drp_input_size)
        self.layer_sizes = [self.drp_input_size] + layer_sizes

        if self.gpu_locs is not None:
            # self.drp_module = NeuralNets.ModuleList([CustomDense(input_size=self.layer_sizes[i],
            #                                              hidden_size=self.layer_sizes[i + 1]) for i in
            #                                  range(len(self.layer_sizes) - 1)])

            # Note: NeuralNets.ModuleList modules are not connected together in any way, so cannot use instead of NeuralNets.Sequential!
            drp_layers = [CustomDense(input_size=self.layer_sizes[i],
                                      hidden_size=self.layer_sizes[i + 1], name="drp_layer_" + str(i)) for i in
                          range(len(self.layer_sizes) - 1)]
            self.drp_module = nn.Sequential(*drp_layers).cuda()

        else:
            drp_layers = [CustomDense(input_size=self.layer_sizes[i],
                                      hidden_size=self.layer_sizes[i + 1], name="drp_layer_" + str(i)) for i in
                          range(len(self.layer_sizes) - 1)]
            self.drp_module = nn.Sequential(*drp_layers).to("cpu")
            # self.drp_module = NeuralNets.ModuleList([CustomDense(input_size=self.layer_sizes[i],
            #                                              hidden_size=self.layer_sizes[i + 1]) for i in
            #                                  range(len(self.layer_sizes) - 1)])

        self.encoders = nn.ModuleList(self.encoders)
        print("self.drp_module length:", len(self.drp_module))
        print("self.encoders length:", len(self.encoders))
        print("self.encoder_out_sizes:", self.encoder_out_sizes)

    def forward(self, *inputs):
        # TODO this "is.instance" might cause a slow down
        # We have an arbitrarily sized list of inputs matching the number of given encoders
        assert len(inputs) == self.len_encoder_list, \
            "Number of inputs should match the number of encoders and must be in the same order"
        # inputs = [drug, mut, cnv, exp, prot]
        # TODO This part is for the case where different GPUs handle different inputs
        # if self.gpu_locs is not None:
        #     # Move inputs to the same GPU as the respective encoder (assuming same order)
        #     # TODO: Does this back and forth transfer result in speed loss?
        #     # Use non_blocking=True results in an asynchronous transfer, potentially faster. .forward() is async by default
        #     encoder_outs = [
        #         self.encoder_list[j].forward(inputs[j].to("cuda:" + str(self.gpu_locs[i]), non_blocking=True)) for i, j
        #         in
        #         zip(range(len(self.gpu_locs)), range(len(self.encoder_list)))]
        #     # Then bring each result back to default GPU: 0
        #     encoder_outs = [out.cuda(non_blocking=True) for out in encoder_outs]
        # else:
        #     encoder_outs = [self.encoder_list[i].forward(inputs[i].to("cpu")) for i in range(len(self.encoder_list))]
        # encoder_outs = [self.encoder_list[i](inputs[i].cuda(non_blocking=True)) for i in
        #                 range(len(self.encoder_list))]
        encoder_outs = [self.encoders[i](inputs[i]) for i in
                        range(len(self.encoders))]
        # Concatenate the outputs by rows
        drp_input = torch.cat(encoder_outs, 1)
        # Pass on to DRP module, which is on default GPU: 0
        # for i in range(len(self.drp_module)):
        #     drp_input = self.drp_module[i](drp_input)
        return self.drp_module(drp_input)
        # return drp_input


class LMF(nn.Module):
    """
    Low-rank Multimodal Fusion
    """

    def __init__(self, drp_layer_sizes: List,
                 encoder_requires_grad: bool = True,
                 gnn_info: List = (False, None),
                 output_dim: int = 256,
                 rank: int = 4,
                 act_fun_list=None,
                 batchnorm_list=None,
                 dropout_list=0.0,
                 mode: str = "train",
                 *encoders):
        """
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        """
        super(LMF, self).__init__()
        self.output_dim = output_dim
        self.rank = rank
        self.gnn_info = gnn_info
        self.mode = mode

        if batchnorm_list is None or batchnorm_list is False:
            batchnorm_list = [False] * len(drp_layer_sizes)
        elif batchnorm_list is True:
            batchnorm_list = [True] * len(drp_layer_sizes)
        else:
            Warning("Incorrect batchnorm_list argument, defaulting to False")
            batchnorm_list = [False] * len(drp_layer_sizes)

        if dropout_list is None or dropout_list == "none" or dropout_list == 0.0:
            dropout_list = [0.0] * len(drp_layer_sizes)

        if act_fun_list is None:
            act_fun_list = [None] * len(drp_layer_sizes)

        self.encoders = encoders
        self.len_encoder_list = len(self.encoders)
        self.layer_sizes = [self.output_dim] + drp_layer_sizes
        self.encoder_requires_grad = encoder_requires_grad

        # Ensure encoders' weights are frozen (for batchnorm_list to work properly)
        # TODO REMOVE
        # for encoder in self.encoders:
        #     for param in encoder.parameters():
        #         param.requires_grad = self.encoder_requires_grad
        # if self.encoder_requires_grad is True:
        #     self.encoders = [encoder.train() for encoder in self.encoders]
        # else:
        #     self.encoders = [encoder.eval() for encoder in self.encoders]

        # Transfer each module to requested GPU
        # self.encoders = [encoder.to("cuda:0", non_blocking=True) for encoder in self.encoders]

        # Generate the outputs using a random tensor (a kind of dry-run)
        if self.gnn_info[0] is False:
            # encoder_outputs = [
            #     encoder(torch.FloatTensor(2, encoder.input_size).to("cuda:0",
            #                                                         non_blocking=True)) for encoder in self.encoders]
            encoder_outputs = [
                encoder(torch.FloatTensor(2, encoder.input_size)) for encoder in self.encoders]
            # The size of the input to the first layer is the sum of the sizes of the outputs of the encoder layers
            self.encoder_out_sizes = [output.shape[1] for output in encoder_outputs]
        else:
            # GNN output size should be given in gnn_info, otherwise it's difficult to do a dry run
            # encoder_outputs = [
            encoder_outputs = [
                encoder(torch.FloatTensor(2, encoder.input_size)) for
                encoder in self.encoders[1:]]
            self.encoder_out_sizes = [gnn_info[1]] + [output.shape[1] for output in encoder_outputs]

        # Convert Python list to torch.ModuleList so that torch can 'see' the parameters of each module
        # self.encoder_list = NeuralNets.ModuleList(self.encoder_list)

        # self.encoder_in_sizes = [encoder.input_size for encoder in self.encoders]

        self.drp_input_size = sum(self.encoder_out_sizes)
        print("Sum width of the encoder outputs is:", self.drp_input_size)
        # self.layer_sizes = [self.drp_input_size] + layer_sizes

        # Note: NeuralNets.ModuleList modules are not connected together in any way, so cannot use instead of NeuralNets.Sequential!
        drp_layers = [CustomDense(input_size=self.layer_sizes[i],
                                  hidden_size=self.layer_sizes[i + 1],
                                  act_fun=act_fun_list[i],
                                  batch_norm=batchnorm_list[i],
                                  dropout=dropout_list[i],
                                  name="drp_" + str(i)) for i in
                      range(len(self.layer_sizes) - 1)]
        # self.drp_module = nn.Sequential(*drp_layers).cuda()
        self.drp_module = nn.Sequential(*drp_layers)

        self.encoders = nn.ModuleList(self.encoders)
        # print("self.drp_module length:", len(self.drp_module))
        print("self.encoders length:", len(self.encoders))
        print("self.encoder_out_sizes:", self.encoder_out_sizes)

        # dimensions are specified in the order of audio, video and text
        # self.audio_in = input_dims[0]
        # self.video_in = input_dims[1]
        # self.text_in = input_dims[2]
        #
        # self.audio_hidden = hidden_dims[0]
        # self.video_hidden = hidden_dims[1]
        # self.text_hidden = hidden_dims[2]

        # self.text_out = text_out

        # define the pre-fusion subnetworks
        # self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        # self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        # self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        # self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        # self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)

        self.all_factors = nn.ParameterList()
        for i in range(len(self.encoders)):
            cur_factor = Parameter(torch.Tensor(self.rank, self.encoder_out_sizes[i] + 1, self.output_dim))
            # init the factors
            xavier_normal_(cur_factor)
            self.all_factors.append(cur_factor)

        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        xavier_normal_(self.fusion_weights)

        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))
        self.fusion_bias.data.fill_(0)

    def forward(self, *inputs):
        """
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        """

        if self.gnn_info[0] is False:
            assert len(inputs) == self.len_encoder_list, \
                "Number of inputs should match the number of encoders and must be in the same order. len(inputs): " +\
                str(len(inputs)) + ", len_encoder_list: " + str(self.len_encoder_list)

            encoder_outs = [self.encoders[i](inputs[i]) for i in
                            range(len(self.encoders))]
        else:
            # GNN data input must be handled differently
            if self.mode == "test":
                gnn_output = self.encoders[0](inputs[0][0], inputs[0][1], inputs[0][2], inputs[0][3])
            else:
                gnn_output = self.encoders[0](inputs[0].x, inputs[0].edge_index, inputs[0].edge_attr, inputs[0].batch)
            # gnn_output = self.encoders[0](inputs[0][0], inputs[0][1], inputs[0][2], inputs[0][3])
            # TODO Re-order manually until torch_geometric Dataloader solution is found
            # cur_omics = inputs[0].omic_data
            # len_enc = len(inputs[0].omic_data[0])
            # len_batch = len(inputs[0].omic_data)
            # correct_omics = []
            # for i in range(len_enc):
            #     correct_omics.append(torch.vstack([cur_omics[j][i] for j in range(len_batch)]))
            # for i in range(1, len(self.encoders)):
            #     print("i in self.encoders:", i)
            #     print("shape of inputs[1][i-1]:", inputs[1][i - 1].shape)
            #     print("cur encoder info:")
            #     for name, param in self.encoders[i].named_parameters():
            #         # if param.requires_grad:
            #         print("Layer Name:", name, "\nLayer Shape:", param.data.shape)
                    # self.encoders[i](inputs[1][i - 1])
            encoder_outs = [gnn_output] + [self.encoders[i](inputs[1][i - 1]) for i in
                                           range(1, len(self.encoders))]

        # TODO REMOVE
        # print(encoder_outs)

        batch_size = encoder_outs[1].shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and element-wise product
        if encoder_outs[0].is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _encoder_outs = []
        for i in range(len(encoder_outs)):
            _encoder_outs.append(
                torch.cat((torch.ones((batch_size, 1), requires_grad=False).type(DTYPE), encoder_outs[i]),
                          dim=1))

        # _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        # _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        # _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)

        fusions = []
        for i in range(len(encoder_outs)):
            fusions.append(torch.matmul(_encoder_outs[i], self.all_factors[i]))
        # fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        # fusion_video = torch.matmul(_video_h, self.video_factor)
        # fusion_text = torch.matmul(_text_h, self.text_factor)

        fusion_zy = fusions[0]
        for i in range(1, len(fusions)):
            fusion_zy = fusion_zy * fusions[i]
        # fusion_zy = fusion_audio * fusion_video * fusion_text

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # print("fusion_zy shape:", fusion_zy.shape)
        # print("fusion_weights shape:", self.fusion_weights.shape)
        # use linear transformation instead of simple summation, more flexibility
        fusion_output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        fusion_output = fusion_output.view(-1, self.output_dim)
        # print("fusion_output shape:", fusion_output.shape)
        # if self.use_softmax:
        #     output = F.softmax(output)

        # TODO REMOVE
        # print(fusion_output)
        # Pass the output to the DRP layers
        output = self.drp_module(fusion_output)

        return output


class LMFTest(nn.Module):
    """
    Low-rank Multimodal Fusion
    """

    def __init__(self, drp_layer_sizes: List,
                 encoder_requires_grad: bool = True,
                 gnn_info: List = (False, None),
                 output_dim: int = 256,
                 rank: int = 4,
                 act_fun_list=None,
                 batchnorm_list=None,
                 dropout_list=0.0,
                 *encoders):
        """
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        """
        super(LMFTest, self).__init__()
        self.output_dim = output_dim
        self.rank = rank
        self.gnn_info = gnn_info

        if batchnorm_list is None or batchnorm_list is False:
            batchnorm_list = [False] * len(drp_layer_sizes)
        elif batchnorm_list is True:
            batchnorm_list = [True] * len(drp_layer_sizes)
        else:
            Warning("Incorrect batchnorm_list argument, defaulting to False")
            batchnorm_list = [False] * len(drp_layer_sizes)

        if dropout_list is None or dropout_list == "none" or dropout_list == 0.0:
            dropout_list = [0.0] * len(drp_layer_sizes)

        if act_fun_list is None:
            act_fun_list = [None] * len(drp_layer_sizes)

        self.encoders = encoders
        self.len_encoder_list = len(self.encoders)
        self.layer_sizes = [self.output_dim] + drp_layer_sizes
        self.encoder_requires_grad = encoder_requires_grad

        # Ensure encoders' weights are frozen (for batchnorm_list to work properly)
        # TODO REMOVE
        # for encoder in self.encoders:
        #     for param in encoder.parameters():
        #         param.requires_grad = self.encoder_requires_grad
        # if self.encoder_requires_grad is True:
        #     self.encoders = [encoder.train() for encoder in self.encoders]
        # else:
        #     self.encoders = [encoder.eval() for encoder in self.encoders]

        # Transfer each module to requested GPU
        # self.encoders = [encoder.to("cuda:0", non_blocking=True) for encoder in self.encoders]

        # Generate the outputs using a random tensor (a kind of dry-run)
        if self.gnn_info[0] is False:
            # encoder_outputs = [
            #     encoder(torch.FloatTensor(2, encoder.input_size).to("cuda:0",
            #                                                         non_blocking=True)) for encoder in self.encoders]
            encoder_outputs = [
                encoder(torch.FloatTensor(2, encoder.input_size)) for encoder in self.encoders]
            # The size of the input to the first layer is the sum of the sizes of the outputs of the encoder layers
            self.encoder_out_sizes = [output.shape[1] for output in encoder_outputs]
        else:
            # GNN output size should be given in gnn_info, otherwise it's difficult to do a dry run
            # encoder_outputs = [
                # encoder(torch.FloatTensor(2, encoder.input_size).to("cuda:0",
                #                                                     non_blocking=True)) for
                # encoder in self.encoders[1:]]
            encoder_outputs = [
                encoder(torch.FloatTensor(2, encoder.input_size)) for
                encoder in self.encoders[1:]]
            self.encoder_out_sizes = [gnn_info[1]] + [output.shape[1] for output in encoder_outputs]

        # Convert Python list to torch.ModuleList so that torch can 'see' the parameters of each module
        # self.encoder_list = NeuralNets.ModuleList(self.encoder_list)

        # self.encoder_in_sizes = [encoder.input_size for encoder in self.encoders]

        self.drp_input_size = sum(self.encoder_out_sizes)
        print("Sum width of the encoder outputs is:", self.drp_input_size)
        # self.layer_sizes = [self.drp_input_size] + layer_sizes

        # Note: NeuralNets.ModuleList modules are not connected together in any way, so cannot use instead of NeuralNets.Sequential!
        drp_layers = [CustomDense(input_size=self.layer_sizes[i],
                                  hidden_size=self.layer_sizes[i + 1],
                                  act_fun=act_fun_list[i],
                                  batch_norm=batchnorm_list[i],
                                  dropout=dropout_list[i],
                                  name="drp_" + str(i)) for i in
                      range(len(self.layer_sizes) - 1)]
        # self.drp_module = nn.Sequential(*drp_layers).cuda()
        self.drp_module = nn.Sequential(*drp_layers)

        self.encoders = nn.ModuleList(self.encoders)
        # print("self.drp_module length:", len(self.drp_module))
        print("self.encoders length:", len(self.encoders))
        print("self.encoder_out_sizes:", self.encoder_out_sizes)

        # dimensions are specified in the order of audio, video and text
        # self.audio_in = input_dims[0]
        # self.video_in = input_dims[1]
        # self.text_in = input_dims[2]
        #
        # self.audio_hidden = hidden_dims[0]
        # self.video_hidden = hidden_dims[1]
        # self.text_hidden = hidden_dims[2]

        # self.text_out = text_out

        # define the pre-fusion subnetworks
        # self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        # self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        # self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        # self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        # self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)

        self.all_factors = nn.ParameterList()
        for i in range(len(self.encoders)):
            cur_factor = Parameter(torch.Tensor(self.rank, self.encoder_out_sizes[i] + 1, self.output_dim))
            # init the factors
            xavier_normal_(cur_factor)
            self.all_factors.append(cur_factor)

        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        xavier_normal_(self.fusion_weights)

        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))
        self.fusion_bias.data.fill_(0)

    def forward(self, *inputs):
        """
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        """

        if self.gnn_info[0] is False:
            assert len(inputs) == self.len_encoder_list, \
                "Number of inputs should match the number of encoders and must be in the same order. len(inputs): " +\
                str(len(inputs)) + ", len_encoder_list: " + str(self.len_encoder_list)

            encoder_outs = [self.encoders[i](inputs[i]) for i in
                            range(len(self.encoders))]
        else:
            # GNN data input must be handled differently
            # gnn_output = self.encoders[0](inputs[0].x, inputs[0].edge_index, inputs[0].edge_attr, inputs[0].batch)
            gnn_output = self.encoders[0](inputs[0][0], inputs[0][1], inputs[0][2], inputs[0][3])
            # TODO Re-order manually until torch_geometric Dataloader solution is found
            # cur_omics = inputs[0].omic_data
            # len_enc = len(inputs[0].omic_data[0])
            # len_batch = len(inputs[0].omic_data)
            # correct_omics = []
            # for i in range(len_enc):
            #     correct_omics.append(torch.vstack([cur_omics[j][i] for j in range(len_batch)]))

            encoder_outs = [gnn_output] + [self.encoders[i](inputs[1][i - 1]) for i in
                                           range(1, len(self.encoders))]

        # TODO REMOVE
        # print(encoder_outs)

        batch_size = encoder_outs[1].shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and element-wise product
        if encoder_outs[0].is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _encoder_outs = []
        for i in range(len(encoder_outs)):
            _encoder_outs.append(
                torch.cat((torch.ones((batch_size, 1), requires_grad=False).type(DTYPE), encoder_outs[i]),
                          dim=1))

        # _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        # _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        # _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)

        fusions = []
        for i in range(len(encoder_outs)):
            fusions.append(torch.matmul(_encoder_outs[i], self.all_factors[i]))
        # fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        # fusion_video = torch.matmul(_video_h, self.video_factor)
        # fusion_text = torch.matmul(_text_h, self.text_factor)

        fusion_zy = fusions[0]
        for i in range(1, len(fusions)):
            fusion_zy = fusion_zy * fusions[i]
        # fusion_zy = fusion_audio * fusion_video * fusion_text

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # print("fusion_zy shape:", fusion_zy.shape)
        # print("fusion_weights shape:", self.fusion_weights.shape)
        # use linear transformation instead of simple summation, more flexibility
        fusion_output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        fusion_output = fusion_output.view(-1, self.output_dim)
        # print("fusion_output shape:", fusion_output.shape)
        # if self.use_softmax:
        #     output = F.softmax(output)

        # TODO REMOVE
        # print(fusion_output)
        # Pass the output to the DRP layers
        output = self.drp_module(fusion_output)

        return output


class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class GraphDiscriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphDiscriminator, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
