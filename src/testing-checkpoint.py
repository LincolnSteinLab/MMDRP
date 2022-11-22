import numpy as np
import torch
from Models import DNNAutoEncoder, MorganAutoEncoder, FullDrugResponsePredictorTest
from DRP.src.ModuleLoader import ExtractEncoder
from torchinfo import summary
import hiddenlayer as hl

morgan_model = MorganAutoEncoder(input_dim=4096,
                                 first_layer_size=1024,
                                 code_layer_size=1024,
                                 num_layers=2)
summary(morgan_model, (4096,))
mut_model = DNNAutoEncoder(input_dim=691,
                           first_layer_size=512,
                           code_layer_size=128,
                           num_layers=2)
summary(mut_model, (691,))

cnv_model = DNNAutoEncoder(input_dim=27000,
                           first_layer_size=8192,
                           code_layer_size=1024,
                           num_layers=2)
summary(cnv_model, (27000,))
exp_model = DNNAutoEncoder(input_dim=22000,
                           first_layer_size=8192,
                           code_layer_size=2048,
                           num_layers=2)
summary(exp_model, (22000,))
prot_model = DNNAutoEncoder(input_dim=5500,
                            first_layer_size=2048,
                            code_layer_size=1024,
                            num_layers=3)
summary(prot_model, (5500,))
cur_autoencoders = [morgan_model, mut_model, cnv_model, exp_model, prot_model]
cur_encoders = [ExtractEncoder(autoencoder) for autoencoder in cur_autoencoders]
# Determine layer sizes, add final target layer
cur_layer_sizes = list(np.linspace(4096, 512, 3).astype(int))
cur_layer_sizes.append(1)
cur_model = FullDrugResponsePredictorTest(encoder_list=cur_encoders, layer_sizes=cur_layer_sizes,
                                  encoder_requires_grad=True)
summary(cur_model, [(4096,), (691,), (27000,), (22000,), (5500,)], device="cpu")
hl.build_graph(cur_model, (torch.zeros([1, 4096]),
                           torch.zeros([1, 691]),
                           torch.zeros([1, 27000]),
                           torch.zeros([1, 22000]),
                           torch.zeros([1, 5500])))
