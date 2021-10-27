'''
基于SimCSE的思路fine-tuning sentence-embedding
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataloader
import numpy as np
import csv

from dataset import SimCSEDataset
from dataset import collate_simCSE
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
    dataset_train = SimCSEDataset(
        sents1=corpus,
        sents2=corpus
    )
    dataloader_train = Dataloader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_simCSE
    ) 


if __name__ == '__main__':
    run()