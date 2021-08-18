"""
使用textCNN 网络构造一个分类器，进行情感分类
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import pandas as pd

import config
from utils import make_data


dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#sentences = ["我 喜欢 你", "他 喜欢 我", "他 喜欢 篮球", "他 讨厌 你", "我 很 抱歉", "我 的 老天爷"]
#labels = [1, 1, 1, 0, 0, 0]

dfs_train = pd.read_csv("data/sentiment_train_weibo_80k.csv", sep="\t")
sentences_train, labels_train = dfs_train['review_clean'].tolist(), dfs_train['label'].tolist()
# dfs_test = pd.read_test("classification/text-classification-CNN-PyTorch/data/sentiment_test_2k.csv", sep="\t")

inputs, word2idx, idx2word, vocab, vocab_size = make_data(sentences_train)

print(vocab_size)
print(word2idx)