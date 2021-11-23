'''
将数据与labels 进行 dataset 化
'''

from torch.utils.data import Dataset
import torch

class YeastDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.labels[idx])

    def __len__(self):
        return len(self.labels)