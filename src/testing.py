import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from Models import DNNAutoEncoder, FullDrugResponsePredictorTest, MultiHeadCNNAutoEncoder



activation_dict = nn.ModuleDict({
                'lrelu': nn.LeakyReLU(),
                'prelu': nn.PReLU(),
                'relu': nn.ReLU(),
                'none': nn.Identity()
        })

cnn_model = MultiHeadCNNAutoEncoder(
    input_width=4096,
    num_branch=3,
    stride=2,
    batchnorm_list=True,
    act_fun_list='prelu',
    dropout_list=0.163313
)

writer = SummaryWriter('runs/cnn_model')
writer.add_graph(cnn_model, torch.zeros([1, 1, 4096]))

morgan_model = DNNAutoEncoder(input_dim=4096,
                              first_layer_size=1024,
                              code_layer_size=1024,
                              num_layers=2, name="morgan")
writer = SummaryWriter('runs/morgan_model')
writer.add_graph(morgan_model, torch.zeros([1, 4096]))
writer.close()

# summary(morgan_model, (4096,))
mut_model = DNNAutoEncoder(input_dim=691,
                           first_layer_size=512,
                           code_layer_size=128,
                           num_layers=2, name="mut")
writer = SummaryWriter('runs/mut_model')
writer.add_graph(mut_model, torch.zeros([1, 691]))
writer.close()
# summary(mut_model, (691,))

cnv_model = DNNAutoEncoder(input_dim=27000,
                           first_layer_size=8192,
                           code_layer_size=1024,
                           num_layers=2, name="cnv")
writer = SummaryWriter('runs/cnv_model')
writer.add_graph(cnv_model, torch.zeros([1, 27000]))
writer.close()

# summary(cnv_model, (27000,))
exp_model = DNNAutoEncoder(input_dim=22000,
                           first_layer_size=8192,
                           code_layer_size=2048,
                           num_layers=2, name="exp")
writer = SummaryWriter('runs/exp_model')
writer.add_graph(exp_model, torch.zeros([1, 22000]))
writer.close()

# summary(exp_model, (22000,))
prot_model = DNNAutoEncoder(input_dim=5500,
                            first_layer_size=2048,
                            code_layer_size=1024,
                            num_layers=3, name="prot")
writer = SummaryWriter('runs/prot_model')
writer.add_graph(prot_model, torch.zeros([1, 5500]))
writer.close()

# summary(prot_model, (5500,))
# cur_autoencoders = NeuralNets.ModuleList([morgan_model, mut_model, cnv_model, exp_model, prot_model])
# cur_encoders = NeuralNets.ModuleList([autoencoder.encoder for autoencoder in cur_autoencoders])

# list(cur_encoders[0].named_children())
# list(cur_encoders[1].named_children())
# list(cur_encoders[2].named_children())

# Determine layer sizes, add final target layer
cur_layer_sizes = list(np.linspace(4096, 512, 3).astype(int))
cur_layer_sizes.append(1)
cur_model = FullDrugResponsePredictorTest(*[cur_layer_sizes, None, True,
                                            morgan_model.encoder,
                                            cnv_model.encoder,
                                            mut_model.encoder,
                                            exp_model.encoder,
                                            prot_model.encoder])
summary(cur_model, [(1,4096), (1,27000), (1,691), (1,22000), (1,5500)], device="cpu")
# hl.build_graph(cur_model, (torch.zeros([1, 4096]),
#                            torch.zeros([1, 691]),
#                            torch.zeros([1, 27000]),
#                            torch.zeros([1, 22000]),
#                            torch.zeros([1, 5500])))

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/full_drp_experiment')
writer.add_graph(cur_model, [torch.zeros([1, 4096]),
                           torch.zeros([1, 27000]),
                           torch.zeros([1, 691]),
                           torch.zeros([1, 22000]),
                           torch.zeros([1, 5500])])
writer.close()


# =============== PairData Testing =====================
from DRPPreparation import drp_main_prep, drp_create_datasets

path = "/Data/DRP_Training_Data/"
bottleneck = True
cur_modules = ['drug', 'mut', 'cnv', 'exp', 'prot']
prep_gen = drp_main_prep(module_list=cur_modules, train_file="CTRP_AAC_MORGAN_512.hdf", path=path, device="cpu")
prep_list = next(prep_gen)
_, final_address, subset_data, subset_keys, subset_encoders, \
    data_list, key_columns, required_data_indices = prep_list

train_data, train_sampler, valid_sampler, \
    train_idx, valid_idx = drp_create_datasets(data_list,
                                               key_columns,
                                               drug_index=0,
                                               drug_dr_column="area_above_curve",
                                               test_drug_data=None,
                                               bottleneck=bottleneck,
                                               required_data_indices=[0, 1, 2, 3, 4])

temp = train_data.get_subset()
set(list(train_data.drug_names))
# Test cell line subsetting
temp1 = train_data.get_subset(cell_lines=["MONOMAC6"])
temp1[0].shape
temp2 = train_data.get_subset(cell_lines=["MONOMAC6", "LS513"])
temp2[0].shape

temp[0].iloc[:, 1]
temp4 = train_data.get_subset(cpd_names=["triptolide"], partial_match=True)
set(list(temp4[0].iloc[:, 1]))
temp3 = train_data.get_subset(cpd_names=["vorinostat:navitoclax (4:1 mol/mol)"])
set(list(temp3[0].iloc[:, 1]))
temp5 = train_data.get_subset(cpd_names=["triptolide", "navitoclax"], partial_match=True)
set(list(temp5[0].iloc[:, 1]))

temp6 = train_data.get_subset(cell_lines=["MONOMAC6", "LS513"], cpd_names=["navitoclax"])
temp7 = train_data.get_subset(cell_lines=["MONOMAC6", "LS513"], cpd_names=["navitoclax"], partial_match=True)

temp8 = train_data.get_subset(min_target=0.5, max_target=0.8)
temp8[0].shape
temp9 = train_data.get_subset(min_target=0.5, max_target=0.8, cell_lines=["MONOMAC6", "LS513"], cpd_names=["navitoclax"], partial_match=True)
temp9[0].shape

temp10 = train_data.get_subset(min_target=0.5, max_target=0.8, cell_lines=["MONOMAC6", "LS513"],
                               cpd_names=["navitoclax"], partial_match=True, make_main=True)
