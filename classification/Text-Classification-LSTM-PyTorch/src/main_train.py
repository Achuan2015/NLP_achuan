import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils import make_data
from dataset import MyDataset
from torch.utils.data import DataLoader
import config
from model import BiLSTM


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentence = (
    'GitHub Actions makes it easy to automate all your software workflows '
    'from continuous integration and delivery to issue triage and more'
    )
    word2idx = {w:i + 1 for i, w in enumerate(list(set(sentence.split(' '))))}
    idx2word = {i + 1:w for i, w in enumerate(list(set(sentence.split(' '))))}
    word2idx['<PAD>'] = 0
    idx2word[0] = '<PAD>'
    n_class = len(word2idx)
    # print('nclass', n_class)
    max_len = len(sentence.split(" "))
    # print('max_len', max_len)
    n_hidden = 5
    input_batch, target_batch = make_data(sentence, word2idx)
    # print(input_batch)
    # print(target_batch)
    dataset = MyDataset(input_batch, target_batch)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = BiLSTM(config).to(device)
    criteron = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(config.num_epoch):
        print('epoch:', epoch)
        for x, y in dataloader:
            output = model(x)
            loss = criteron(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    model.eval()
    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(sentence)
    print([idx2word[n.item()] for n in predict.squeeze()])

    
    


if __name__ == "__main__":
    run()