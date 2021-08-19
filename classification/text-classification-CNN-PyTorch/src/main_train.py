"""
使用textCNN 网络构造一个分类器，进行情感分类
"""


import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn import metrics

import config
from utils import make_data
from dataset import SentimentDataset
from engine import train_fn
from engine import eval_fn
from text_cnn import TextCNN


def run():
    dtype = torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #sentences = ["我 喜欢 你", "他 喜欢 我", "他 喜欢 篮球", "他 讨厌 你", "我 很 抱歉", "我 的 老天爷"]
    #labels = [1, 1, 1, 0, 0, 0]

    dfs_train = pd.read_csv("data/weibo_hrtps_senti_corpus.csv", sep="\t")
    sentences_train, labels_train = dfs_train['review_clean'].tolist(), dfs_train['label'].tolist()

    dfs_test = pd.read_csv("data/hrtps_sentiment_2k.csv", sep="\t")
    sentences_test, labels_test = dfs_test['review_clean'].tolist(), dfs_train['label'].tolist()

    inputs_train, word2idx = make_data(sentences_train)
    inputs_test, word2idx = make_data(sentences_test, word2idx)

    dataset_train = SentimentDataset(inputs_train, labels_train)
    dataloader_train = Data.DataLoader(dataset_train, config.batch_size, True)

    dataset_test = SentimentDataset(inputs_test, labels_test)
    dataloader_test = Data.DataLoader(dataset_test, config.batch_size, True)

    model = TextCNN(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=10e-3)
    criterion = nn.CrossEntropyLoss().to(device)

    best_accuracy = 0.75
    for epoch in range(config.num_epoch):
        train_loss = train_fn(dataloader_train, model, device, optimizer, criterion)
        fin_outputs, fin_targets, eval_loss = eval_fn(dataloader_test, model, device, criterion)
        output_indices = np.argmax(np.array(fin_outputs), axis=1)
        accuracy = metrics.accuracy_score(output_indices, np.array(fin_targets))
        print('epoch: ', epoch)
        print(f'train_loss: {train_loss} | eval_loss: {eval_loss}')
        print('accuracy: ', accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), config.model_path)


if __name__ == "__main__":
    run()