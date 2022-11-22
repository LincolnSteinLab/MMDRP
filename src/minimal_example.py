import torch
import torch_geometric
from torch.utils import data
from torch_geometric.data import Data


class MultiModalDataSet(data.Dataset):
    def __init__(self):
        super(MultiModalDataSet, self).__init__()
        self.modality_1 = torch.rand((16, 32))
        self.modality_2 = torch.rand((16, 64))
        self.modality_3 = torch.rand((16, 48))

        self.targets = torch.rand((16, 1))

    def __len__(self):
        return self.modality_1.shape[0]

    def __getitem__(self, idx):
        return [self.modality_1[idx, ], self.modality_2[idx, ], self.modality_3[idx,]], self.targets


# Pytorch DataLoader batching/collation
my_data = MultiModalDataSet()
my_loader = data.DataLoader(my_data, batch_size=4)
my_iter = iter(my_loader)
torch_dataloader_sample = next(my_iter)

len(torch_dataloader_sample)  # 2, for data + targets
len(torch_dataloader_sample[0])  # 3, for 3 modalities
torch_dataloader_sample[0][0].shape  # modality 1 batched together
torch_dataloader_sample[0][1].shape  # modality 2 batched together
torch_dataloader_sample[0][2].shape  # modality 3 batched together


class MyGNNData(Data):
    def __cat_dim__(self, key, item):
        if key in ['modality_1', 'modality_2', 'modality_3', 'target']:
            return None
        else:
            return super().__cat_dim__(key, item)


class MultiModalGraphDataSet(data.Dataset):
    def __init__(self):
        super(MultiModalGraphDataSet, self).__init__()
        self.data = []
        for i in range(16):
            modality_1 = torch.rand((1, 32))
            modality_2 = torch.rand((1, 64))
            modality_3 = torch.rand((1, 48))
            target = torch.rand((1, 1))
            edge_index = torch.tensor([
               [0, 1, 1, 2],
               [1, 0, 2, 1],
            ])

            data = MyGNNData(edge_index=edge_index,
                             modalities=[modality_1, modality_2, modality_3],
                             target=target)
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MultiModalGraphDataSet2(data.Dataset):
    def __init__(self):
        super(MultiModalGraphDataSet2, self).__init__()
        self.modality_1 = torch.rand((16, 32))
        self.modality_2 = torch.rand((16, 64))
        self.modality_3 = torch.rand((16, 48))
        self.target = torch.rand((16, 1))
        self.data = []
        for i in range(16):
            edge_index = torch.tensor([
               [0, 1, 1, 2],
               [1, 0, 2, 1],
            ])

            data = MyGNNData(edge_index=edge_index)
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], [self.modality_1[idx, ], self.modality_2[idx, ], self.modality_3[idx, ]], self.target[idx, ]

# Torch Geometric DataLoader batching/collation
my_data = MultiModalGraphDataSet2()
my_loader = torch_geometric.data.DataLoader(my_data, batch_size=4)
my_iter = iter(my_loader)
torchgeo_dataloader_sample = next(my_iter)

len(torchgeo_dataloader_sample.modalities)  # 4, 1 for each sample/datapoint
len(torchgeo_dataloader_sample.modalities[0])  # 3, for 3 modalities
torchgeo_dataloader_sample.modalities[0][0].shape  # modality 1,2,3 batched together as 1 sample
