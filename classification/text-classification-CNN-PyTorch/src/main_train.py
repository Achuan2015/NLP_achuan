"""
使用textCNN 网络构造一个分类器，进行情感分类
"""


import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import pandas as pd

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

    dfs_train = pd.read_csv("data/sentiment_train_weibo_80k.csv", sep="\t")
    sentences_train, labels_train = dfs_train['review_clean'].tolist(), dfs_train['label'].tolist()

    dfs_test = pd.read_csv("data/sentiment_test_2k.csv", sep="\t")

    inputs_train, word2idx, idx2word = make_data(sentences_train)

    dataset_train = SentimentDataset(inputs_train, labels_train)
    dataloader_train = Data.DataLoader(dataset_train, config.batch_size, True)

    model = TextCNN(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=10e-3)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(config.num_epoch):
        train_fn(dataloader_train, model, device, optimizer, criterion)
        # fin_outputs, fin_targets = eval_fn()


if __name__ == "__main__":
    run()