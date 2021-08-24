from sklearn.utils.multiclass import type_of_target
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import BertConfig, BertForSequenceClassification

import config
from dataset import MyDataset
from engine import train_fn, eval_fn


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_train = "../data/weibo_hrtps_senti_corpus.csv"
    path_eval = "../data/hrtps_sentiment_2k.csv"

    dfs_train = pd.read_csv(path_train, sep="\t")
    dfs_eval = pd.read_csv(path_eval, sep="\t")

    sentences_train, labels_train = dfs_train['review_clean'].tolist(), dfs_train['label'].tolist()
    sentences_eval, labels_eval = dfs_eval['review_clean'].tolist(), dfs_eval['label'].tolist()

    dataset_train = MyDataset(sentences_train, labels_train)
    dataset_eval = MyDataset(sentences_eval, labels_eval)

    datalaoder_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    dataloader_eval = DataLoader(dataset_eval, batch_size=config.batch_size, shuffle=False)

    # 载入预训练模型的BertConfig基于BertForSequenceClassification
    model_path = config.model_path
    config_bert = BertConfig.from_pretrained(model_path, num_labels=config.num_labels, hidden_dropout_prob=config.hidden_dropout_prob)
    model = BertForSequenceClassification.from_pretrained(model_path, config=config_bert).to(device)
    # 定义优化器 optimizer ， 损失函数 已经在 model里面了
    ## method 1： 简单定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    ## method 2：详细定义优化指定 一些模块不进行权重衰减 weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    weight_decay = config.weight_decay
    learning_rate = config.learning_rate
    optimizer_grouped_parameters = [
        {'params':[p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params':[p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    for epoch in range(config.epochs):
        train_loss = train_fn(datalaoder_train, model, device, optimizer)
        eval_loss, fin_outputs = eval_fn(dataloader_eval, model, device)
        fin_indices = np.argmax(np.array(fin_outputs), axis=1)
        accuracy = accuracy_score(fin_indices, np.array(labels_eval))
        print(f'epoch: {epoch + 1} | train_loss: {train_loss} | eval_loss: {eval_loss} |accurate: {accuracy}')
    

if __name__ == "__main__":
    run()