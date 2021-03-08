# The purpose of this file is to load parts of modules, e.g. the encoder portion of the protein auto-encoder
import torch
import torch.nn as nn

# use_cuda = torch.cuda.is_available()
# torch.cuda.device_count()
# if use_cuda:
#     print('CUDA is available!')
# device = torch.device("cuda:0" if use_cuda else "cpu")
# Turning off benchmarking makes highly dynamic models faster
# cudnn.benchmark = True
# cudnn.deterministic = True


class ExtractEncoder(nn.Module):
    """
    This class takes in an auto-encoder and the path to its pickled state dict.
    If no path is given, then the encoder portion is created as-is.
    """
    def __init__(self, autoencoder, state_path=None):
        super(ExtractEncoder, self).__init__()
        self.autoencoder = autoencoder
        self.input_size = autoencoder.encoder.input_size
        self.state_path = state_path

        if state_path:
            # Load model and optimizer states
            self.checkpoint = torch.load(self.state_path, map_location="cpu")
            self.autoencoder.load_state_dict(self.checkpoint['model_state_dict'])
            # Extract encoder module
            self.encoder = self.autoencoder.encoder
            # Must indicate inference time, so that batch-norm is turned off
            # self.encoder.eval()
        else:
            self.encoder = self.autoencoder.encoder
            # self.encoder.eval()

    def forward(self, x):
        y = self.encoder(x)

        return y
