import torch.nn as nn


class LinearModel(nn.Module):

    def __init__(self, *args):
        super(LinearModel, self).__init__()
        self.linear_layer1 = nn.Linear(103, 500)
        self.linear_layer2 = nn.Linear(500, 100)
        self.output_layer = nn.Linear(100, 14)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        l1_output = self.relu(self.linear_layer1(x))
        l2_output = self.relu(self.linear_layer2(l1_output))
        output = self.sigmoid(self.output_layer(l2_output))
        return output
        

if __name__ == '__main__':
    model = LinearModel()