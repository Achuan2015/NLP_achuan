"""
区别于 main_train.py 的目的，在 train_dataset 上进行split验证效果
"""
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import model_selection
from sklearn import metrics
import pandas as pd
from dataset import MyDataset
from transformers import get_linear_schedule_with_warmup

from model import BertForSequenceClassification
import config
from engine import train_fn_second, eval_fn_second


# Set the seed value all over the place to make this reproducible.
seed_val = 42

np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_train = "../data/weibo_hrtps_senti_corpus.csv"
    dfs = pd.read_csv(path_train, sep="\t")

    df_train, df_valid = model_selection.train_test_split(
        dfs,
        test_size=0.1,
        random_state=42,
        stratify=dfs.label.values
    )

    dataset_train = MyDataset(
        inputs=df_train.review_clean.values,
        targets=df_train.label.values
    )

    dataset_valid = MyDataset(
        inputs=df_valid.review_clean.values,
        targets=df_valid.label.values
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True
    )

    dataloader_valid = DataLoader(
        dataset_valid,
        batch_size=config.batch_size,
        shuffle=False
    )
    # 初始化 BERT model
    model = BertForSequenceClassification(config).to(device)
    # 构造分组参数优化器
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0
        }
    ]
    optimizer = optim.AdamW(optimizer_parameters, lr=config.learning_rate)
    
    # 设置训练的 scheduler
    num_training_step = (len(df_train) / config.batch_size) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps= num_training_step
    )
    # 设置训练的 目标函数
    critriion = nn.CrossEntropyLoss().to(device)
    best_accuracy=0.8
    for epoch in range(config.epochs):
        train_loss = train_fn_second(dataloader_train, model, device, optimizer, critriion, scheduler)
        eval_loss, fin_output, fin_target = eval_fn_second(dataloader_valid, model, device, critriion)
        predict_target = fin_output.argmax(axis=1)
        accuracy = metrics.accuracy_score(predict_target, fin_target)
        print(f'epoch:{epoch + 1} train_loss:{train_loss} eval_loss:{eval_loss} accuracy:{accuracy}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), config.model_output)


if __name__ == "__main__":
    run()