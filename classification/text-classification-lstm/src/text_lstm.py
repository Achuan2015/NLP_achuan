import torch
import torch.nn as nn



class LSTM(nn.Module):

    def __init__(self, config):
        pass

input_size = 10
output_size = 20
num_layers = 2
batch_size = 5
seq_len = 3
num_direction = 1

rnn = nn.LSTM(input_size, output_size, num_layers)

input = torch.randn(seq_len, batch_size, input_size)

h0 = torch.randn(num_layers * num_direction, batch_size, output_size)
c0 = torch.randn(num_layers * num_direction, batch_size, output_size)

output, (hn, cn) = rnn(input, (h0, c0))

print(output.shape)

print(hn.shape)

print(cn.shape)