import torch.utils.data as Data
import torch

import config


class MyDataset(Data.Dataset):

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input = self.inputs[idx]
        target = self.targets[idx]
        return input, target


class SentimentDataset(Data.Dataset):

    def __init__(self, inputs, targets, config=None):
        self.inputs = inputs
        self.targets = targets
        self.config = config
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input = self.inputs[idx]
        target = self.targets[idx]
        max_seq_len = config.max_seq_len if self.config else 30
        if len(input) < max_seq_len:
            input = input + [0] * (max_seq_len - len(input))
        else:
            input = input[:max_seq_len]
        return torch.LongTensor(input), torch.LongTensor([target])