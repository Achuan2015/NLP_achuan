import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch.optim as optim
from sklearn import model_selection
from transformers import BertConfig
from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from dataset import SiameseDataset, CrossEncodeDataset
from model import BertCNNForClassification
from engine import train_mse_fn, eval_mse_fn, train_fn, eval_fn
import config_nezha as config


# Set the seed value all over the place to make this reproducible.
seed_val = 42

np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path_train = "../data/50_kmeans_train_data.csv"
    dfs = pd.read_csv(path_train, sep="\t")
    df_train, df_valid = model_selection.train_test_split(
        dfs,
        test_size=0.1,
        random_state=42,
        stratify=dfs.label.values
    )

    dataset_train = CrossEncodeDataset(
        querys=df_train['query'].values,
        candidates=df_train['candidate'].values,
        labels=df_train['label'].values
    )

    dataset_valid = CrossEncodeDataset(
        querys=df_valid['query'].values,
        candidates=df_valid['candidate'].values,
        labels=df_valid['label'].values
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
    model = BertForSequenceClassification.from_pretrained(config.model_path, 
            num_labels=config.num_labels, 
            hidden_dropout_prob=config.hidden_dropout_prob
    )
    model = nn.DataParallel(model, device_ids=[0,1]).to(device)
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
    # loop the epochs
    best_accuracy = 0.9
    for epoch in range(config.epochs):
        train_loss = train_fn(dataloader_train, model, device, optimizer, scheduler)
        eval_loss, eval_accu = eval_fn(dataloader_valid, model, device)
        print(f'epoch:{epoch+1} | train_loss:{train_loss} | eval_loss:{eval_loss} | accuracy:{eval_accu}')
        if eval_accu > best_accuracy:
            best_accuracy = eval_accu
            model.module.save_pretrained(config.output_path)

def run_second():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path_train = "../data/50_kmeans_train_data.csv"
    dfs = pd.read_csv(path_train, sep="\t")
    df_train, df_valid = model_selection.train_test_split(
        dfs,
        test_size=0.1,
        random_state=42,
        stratify=dfs.label.values
    )

    dataset_train = CrossEncodeDataset(
        querys=df_train['query'].values,
        candidates=df_train['candidate'].values,
        labels=df_train['label'].values
    )

    dataset_valid = CrossEncodeDataset(
        querys=df_valid['query'].values,
        candidates=df_valid['candidate'].values,
        labels=df_valid['label'].values
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
    config_bert = BertConfig.from_pretrained(config.model_path)
    model = BertCNNForClassification(config_bert, config)

    model = nn.DataParallel(model, device_ids=[0,1]).to(device)
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
    # loop the epochs
    best_accuracy = 0.9
    for epoch in range(config.epochs):
        train_loss = train_mse_fn(dataloader_train, model, device, optimizer, scheduler)
        eval_loss, eval_accu = eval_mse_fn(dataloader_valid, model, device)
        print(f'epoch:{epoch+1} | train_loss:{train_loss} | eval_loss:{eval_loss} | accuracy:{eval_accu}')
        if eval_accu > best_accuracy:
            best_accuracy = eval_accu
            model.module.save_pretrained(config.output_path)


if __name__ == "__main__":
    run_second()