import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from sklearn.metrics import accuracy_score

from utils import make_test_data, make_data
from dataset import MyDataset, SentimentDataset
from torch.utils.data import DataLoader
import config
from engine import train_fn, eval_fn
from model import BiLSTM, TextBiLSTM


def run_test():
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
    input_batch, target_batch = make_test_data(sentence, word2idx)
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

    
def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = "/Users/hrtps/Projects/momian/NLP_achuan/classification/data/weibo_hrtps_senti_corpus.csv"
    eval_data = "/Users/hrtps/Projects/momian/NLP_achuan/classification/data/hrtps_sentiment_2k.csv"
    dfs_train = pd.read_csv(train_data, sep='\t')
    dfs_eval = pd.read_csv(eval_data, sep='\t')
    sentences_train, labels_train = dfs_train['review_clean'].tolist(), dfs_train['label'].tolist()
    sentences_eval, labels_eval = dfs_eval['review_clean'].tolist(), dfs_eval['label'].tolist()

    inputs_train, word2idx = make_data(sentences_train)
    inputs_eval, word2idx = make_data(sentences_eval, word2idx)

    dataset_train = SentimentDataset(inputs_train, labels_train)
    dataset_eval = SentimentDataset(inputs_eval, labels_eval)

    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    dataloader_eval = DataLoader(dataset_eval, batch_size=config.batch_size, shuffle=True)

    model = TextBiLSTM(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)

    best_accuracy = 0.95
    for epoch in range(config.num_epoch):
        train_loss = train_fn(dataloader_train, model, device, optimizer, criterion)
        fin_outputs, fin_targets, eval_loss = eval_fn(dataloader_eval, model, device, criterion)
        output_indices = np.argmax(np.array(fin_outputs), axis=1)
        accurate = accuracy_score(output_indices, fin_targets)
        print(f'{epoch}:train_loss: {train_loss} | eval_loss: {eval_loss}')
        print(f'{epoch} accurate: {accurate}')
        if accurate > best_accuracy:
            best_accuracy = accurate
            save_path = f"_{round(accurate, 3)}.".join(config.model_path.split("."))
            torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    run()