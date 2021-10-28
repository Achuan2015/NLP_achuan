'''
基于SimCSE的思路fine-tuning sentence-embedding
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim  as optim
from transformers import BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import csv

from dataset import SimCSEDataset
from dataset import collate_simCSE
from engine import train_simCSE_fn
from model import BertForPoolingNetwork
import config


# Set the seed value all over the place to make this reproducible.
seed_val = 42

np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path_train = "../data/question_total.csv"
    corpus = []
    with open(path_train, 'r') as f:
        files = csv.DictReader(f, delimiter='\t')
        for item in files:
            corpus.append(item['question'])
    # 生成训练集
    dataset_train = SimCSEDataset(
        sents1=corpus,
        sents2=corpus
    )
    # 生成数据 生成器
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_simCSE
    )
    # 初始化 BERT Model
    bert_config = BertConfig.from_pretrained(config.model_path)
    model = BertForPoolingNetwork(bert_config, config.model_path).to(device)
    # 构造分组参数优化器，LayerNorm
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
    num_training_step = (len(corpus) // config.batch_size) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps= num_training_step
    )
    # loop the epochs
    best_accuracy = 0.9
    for epoch in range(config.epochs):
        train_loss = train_simCSE_fn(dataloader_train, model, device, optimizer, scheduler)
        # eval_loss, eval_accu = eval_simCSE_fn(dataloader_eval, model, device)
        print(f'epoch:{epoch+1} | train_loss:{train_loss}')
        # if eval_accu > best_accuracy:
        #     best_accuracy = eval_accu
        #     model.save_pretrained(config.output_path)


if __name__ == '__main__':
    run()