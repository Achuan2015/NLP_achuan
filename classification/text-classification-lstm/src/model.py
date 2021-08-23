import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """
    需要改进的点：（1）pack_padded_sequence 不再直接用padding的sentence做计算
                (2) 
    """

    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.config = config
        # 初始化一层 双向LSTM
        self.lstm = nn.LSTM(input_size=config.num_classes, hidden_size=config.hidden_size, num_layers=config.num_layer, bidirectional=True)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
    
    def forward(self, input):
        # 获得 batch_size ，准备生成 t0 时刻的 hidden_state 和 cell_state
        batch_size = input.size(0)
        hidden_state = torch.randn(2 * self.config.num_layer, batch_size, self.config.hidden_size)
        cell_state = torch.randn(2 * self.config.num_layer, batch_size, self.config.hidden_size)

        x = input.transpose(0, 1)
        # outputs 实际上lstm 所有时刻的 hidden_state
        outputs, (_, _) = self.lstm(x, (hidden_state, cell_state))
        # 取最后时刻的输出，也就是最后时刻的 hidden_state 来进行最后的预测
        output = outputs[-1]
        output = self.fc(output)
        return output


class TextBiLSTM(nn.Module):

    def __init__(self, config):
        super(TextBiLSTM, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.feature_size)
        self.lstm = nn.LSTM(input_size=config.feature_size, hidden_size=config.hidden_size, num_layers=config.num_layer, bidirectional=True)
        self.fc = nn.Linear(config.hidden_size * 2,  config.num_classes)
    
    def forward(self, input, device):
        batch_size = input.size(0)
        hidden_state = torch.randn(self.config.num_layer * 2, batch_size, self.config.hidden_size).to(device)
        cell_state = torch.randn(self.config.num_layer * 2, batch_size, self.config.hidden_size).to(device)
        
        emb_output = self.embedding(input).transpose(0, 1)
        outputs, (_, _) = self.lstm(emb_output, (hidden_state, cell_state))
        output = outputs[-1]
        output = self.fc(output)
        return output
        
