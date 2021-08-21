import torch.utils.data as Data


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