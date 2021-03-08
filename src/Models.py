import torch
import torch.nn as nn

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
#                       act_fun=NeuralNets.ReLU(), dropout_list=[0., 0., 0.], encode=None)


# temp = MultiHeadCNNAutoEncoder(input_size=27000, first_layer_size=8196, code_layer_size=1024, code_layer=True, num_layers=4, batchnorm_list=[True]*4,
#                    act_fun=[NeuralNets.ReLU(), NeuralNets.ReLU(),NeuralNets.ReLU(),NeuralNets.ReLU()], encode=True)
# temp = MultiHeadCNNAutoEncoder(num_branch=3, code_layer_size=None, kernel_size_list=[3, 5, 11], in_channels=1,
#                       out_channels_list=[1, 1, 1], batchnorm_list=[False, False, False], stride_list=[2, 2, 2],
#                       act_fun=NeuralNets.ReLU(), dropout_list=[0., 0., 0.], encode=None)
# torchsummary.summary(temp, input_size=(1, 512))


class DNNAutoEncoder(nn.Module):
    def __init__(self, input_dim, first_layer_size, code_layer_size, num_layers, batchnorm=None,
                 act_fun=None, dropout=0.0, name=""):
        super(DNNAutoEncoder, self).__init__()
        # Determine batch norm as a whole for this DNN
        if batchnorm is None or batchnorm is False:
            batchnorm_list = [False] * num_layers
        elif batchnorm is True:
            batchnorm_list = [True] * num_layers
        else:
            Warning("Incorrect batchnorm argument, defaulting to False")
            batchnorm_list = [False] * num_layers

        # Determine the activation function as a whole for this DNN
        # if act_fun is None or act_fun == "none":
        #     act_fun = activation_dict['none']
        # elif act_fun in ['relu', 'prelu', 'lrelu']:
        #     act_fun = activation_dict[act_fun]
        # else:
        #     Warning("Incorrect act_fun argument, defaulting to ReLU")
        #     act_fun = activation_dict['relu']

        if dropout is None or dropout == "none" or dropout == 0.0:
            dropout_list = [0.0] * num_layers
        else:
            dropout_list = [dropout] * num_layers

        self.encoder = CustomCoder(input_size=input_dim, first_layer_size=first_layer_size,
                                   code_layer_size=code_layer_size, num_layers=num_layers, encode=True,
                                   code_layer=True, act_fun=act_fun, batchnorm_list=batchnorm_list,
                                   dropout_list=dropout_list, name=name)
        self.decoder = CustomCoder(input_size=input_dim, first_layer_size=first_layer_size,
                                   code_layer_size=code_layer_size, num_layers=num_layers, encode=False,
                                   code_layer=False, act_fun=act_fun, batchnorm_list=batchnorm_list,
                                   dropout_list=dropout_list, name=name)
        # self.encoder.coder[0].dense.weight
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

    def __init__(self, input_width, num_branch=3, stride=None, in_channels=1,
                 out_channels_list=None, batchnorm=None, act_fun=None, dropout=None):
        super(MultiHeadCNNAutoEncoder, self).__init__()

        # Formula for creating odd-sized kernels from 3 based on given num_branch
        kernel_size_list = list(range(3, 3 + (num_branch * 2), 2))

        self.input_width = input_width

        # Check layer-wise parameters
        if batchnorm is None or batchnorm is False:
            batchnorm = [False] * num_branch
        else:
            batchnorm = [True] * num_branch
        self.batchnorm_list = batchnorm

        act_fun_list = [act_fun] * num_branch
        self.act_fun_list = act_fun_list

        if dropout is None:
            dropout_list = [0.] * num_branch
        else:
            dropout_list = [dropout] * num_branch
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
                      kernel_size_list[i], act_fun=act_fun_list[i], batch_norm=batchnorm[i], stride=stride_list[i],
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
        # self.code_layer.append(torch.cat())

        # Mirror everything for the decoder
        out_channels_list_rev = out_channels_list[::-1]
        act_fun_list_rev = act_fun_list[::-1]
        batchnorm_list_rev = batchnorm[::-1]
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
        #                                           kernel_size_list[0], act_fun=act_fun[0],
        #                                           batch_norm=batchnorm_list[0], dropout=dropout_list[0])])
        #     self.coder.extend([CustomCNN(in_channels_list[i], out_channels_list[i],
        #                                  kernel_size_list[i], act_fun=act_fun[i], batch_norm=batchnorm_list[i],
        #                                  dropout=dropout_list[i]) for i in range(len(num_branch) - 1)])
        #
        # else:
        #     self.coder = NeuralNets.ModuleList([CustomCNN(in_channels_list[i], out_channels_list[i],
        #                                           kernel_size_list[i], act_fun=act_fun[i],
        #                                           batch_norm=batchnorm_list[i],
        #                                           dropout=dropout_list[i]) for i in range(len(num_branch) - 1)])
        #     self.coder.extend([CustomCNN(in_channels_list[0], out_channels_list[0],
        #                                  kernel_size_list[0], act_fun=act_fun[0],
        #                                  batch_norm=batchnorm_list[0], dropout=dropout_list[0])])


class DrugResponsePredictor(nn.Module):
    """
    This class takes in an arbitrary number of encoder modules with pre-assigned weights, freezes them and
    connects them to an MLP for drug response prediction.
    Optional: User can determine the GPU each module will be trained on.
    Optional: User can convert the encoder modules to half (FP16) precision
    """

    def __init__(self, layer_sizes=None, gpu_locs=None, encoder_requires_grad=False, act_fun=None,
                 batchnorm=None, dropout=0.0, *encoders):
        super(DrugResponsePredictor, self).__init__()
        if gpu_locs is not None:
            assert len(gpu_locs) == len(encoders), "gpu_locs:" + str(len(gpu_locs)) + ", encoders:" + str(len(encoders))
        if batchnorm is None or batchnorm is False:
            batchnorm_list = [False] * len(layer_sizes)
        elif batchnorm is True:
            batchnorm_list = [True] * len(layer_sizes)
        else:
            Warning("Incorrect batchnorm argument, defaulting to False")
            batchnorm_list = [False] * len(layer_sizes)

        self.encoders = encoders
        self.len_encoder_list = len(self.encoders)
        self.gpu_locs = gpu_locs
        self.layer_sizes = layer_sizes
        self.encoder_requires_grad = encoder_requires_grad

        # Ensure encoders' weights are frozen (for batchnorm to work properly)
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
            # Generate the outputs using a random tensor (similar to a dry-run)
            encoder_outputs = [
                encoder(torch.FloatTensor(2, encoder.input_size).to("cuda:" + str(self.gpu_locs[i]),
                                                                    non_blocking=True)) for
                i, encoder in zip(range(len(self.gpu_locs)), self.encoders)]
        else:
            # print("Transferring encoder modules to CPU...")
            # TODO: What if user wants to use CPU?
            # Transfer everything to default GPU: 0
            self.encoders = [encoder.to("cpu") for encoder in self.encoders]
            encoder_outputs = [encoder(torch.Tensor(2, encoder.input_size).to("cpu")) for encoder in
                               self.encoders]

        # Convert Python list to torch.ModuleList so that torch can 'see' the parameters of each module
        self.encoders = nn.ModuleList(self.encoders)

        # The size of the input to the first layer is the sum of the sizes of the outputs of the encoder layers
        self.encoder_out_sizes = [output.shape[1] for output in encoder_outputs]
        self.drp_input_size = sum(self.encoder_out_sizes)
        # print("Sum width of the encoder outputs is:", self.drp_input_size)
        self.layer_sizes = [self.drp_input_size] + layer_sizes

        drp_layers = [CustomDense(input_size=self.layer_sizes[i],
                                  hidden_size=self.layer_sizes[i + 1],
                                  act_fun=act_fun,
                                  batch_norm=batchnorm_list[i],
                                  dropout=dropout,
                                  name="drp_" + str(i)) for i in
                      range(len(self.layer_sizes) - 1)]
        if self.gpu_locs is not None:
            # TODO: setup model-parallel multi-gpu here
            # Note: NeuralNets.ModuleList modules are not connected together in any way, so cannot use instead of NeuralNets.Sequential!
            self.drp_module = nn.Sequential(*drp_layers).cuda()
        else:
            # drp_layers = [CustomDense(input_size=self.layer_sizes[i],
            #                           hidden_size=self.layer_sizes[i + 1]) for i in
            #               range(len(self.layer_sizes) - 1)]
            self.drp_module = nn.Sequential(*drp_layers).to("cpu")

    def forward(self, *inputs):
        # TODO this "is.instance" might cause a slow down
        # We have an arbitrarily sized list of inputs matching the number of given encoders
        assert len(inputs) == self.len_encoder_list, \
            "Number of inputs should match the number of encoders and must be in the same order"
        # if len(inputs) != self.len_encoder_list:
        #     print("Number of inputs should match the number of encoders and must be in the same order")
        #     print("Input Length:", len(inputs), ", expected:", self.len_encoder_list)
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
        # TODO how to make this operation async? or is it inherently async?
        encoder_outs = [self.encoders[i](inputs[i]) for i in
                        range(len(self.encoders))]
        # Concatenate the outputs by rows
        drp_input = torch.cat(encoder_outs, 1)
        # Pass on to DRP module, which is on default GPU: 0
        # for i in range(len(self.drp_module)):
        #     drp_input = self.drp_module[i](drp_input)
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

        # Ensure encoders' weights are frozen (for batchnorm to work properly)
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
