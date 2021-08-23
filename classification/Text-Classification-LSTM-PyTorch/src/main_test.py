import sys
import torch
import pandas as pd
import joblib
from torch._C import device
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
import sys

from model import TextBiLSTM
import config
from dataset import SentimentDataset
from utils import get_bestmodel_path, make_data
from engine import predict_fn
    

def run():    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "output/"
    model_path = get_bestmodel_path(model_dir)
    if not model_path:
        sys.exit()
    eval_data = "../data/hrtps_sentiment_2k.csv"

    dfs_eval = pd.read_csv(eval_data, sep='\t')

    sentences_eval, labels_eval = dfs_eval['review_clean'].tolist(), dfs_eval['label'].tolist()

    if os.path.exists("output/word2idx.pkl"):
        word2idx = joblib.load("output/word2idx.pkl")
    inputs_eval, word2idx = make_data(sentences_eval, word2idx)

    dataset_eval = SentimentDataset(inputs_eval, labels_eval)

    dataloader_eval = DataLoader(dataset_eval, batch_size=config.batch_size, shuffle=False)

    model = TextBiLSTM(config).to(device)
    model.load_state_dict(torch.load(model_path))
    
    fin_outputs = predict_fn(dataloader_eval, model, device)
    fin_indices = np.argmax(np.array(fin_outputs), axis=1)
    dfs_eval['predict'] = list(fin_indices)
    dfs_eval.to_csv("output/predict_result.csv", sep='\t', index=False)


if __name__ == "__main__":
    run()