import torch
import torch.nn as nn
import config


class TextCNN(nn.Module):

    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(256, config.feature_size, kernel_size=h),
            nn.ReLU(),nn.MaxPool1d(config.max_text_len - h + 1)) for h in config.window_sizes]
        )
        self.fc = nn.Linear(in_features=config.feature_size * len(config.window_sizes), out_features=config.num_classes)
    
    def forward(self, x):
        embed = self.embedding(x)  # embed -> 32 * 35 * 256
        embed = embed.transpose(1, 2) # embed -> 32 * 256 * 35
        out = [conv1d(embed) for conv1d in self.convs] # 32 * 100 * 1
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))
        out = self.fc(out)
        return out

if __name__ == "__main__":
    model = TextCNN(config)
    fake_inputs = torch.randint(0, config.vocab_size, (32, config.max_text_len))
    out = model(fake_inputs)
    print(out.shape)
        