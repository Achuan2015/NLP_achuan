import joblib

import torch
import pandas as pd
import torch.utils.data as Data
import numpy as np

import config
from utils import make_data
from dataset import SentimentDataset
from engine import predict_fn


def run():
    dtype = torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #sentences = ["我 喜欢 你", "他 喜欢 我", "他 喜欢 篮球", "他 讨厌 你", "我 很 抱歉", "我 的 老天爷"]
    #labels = [1, 1, 1, 0, 0, 0]

    dfs_test = pd.read_csv("data/hrtps_sentiment_2k.csv", sep="\t")
    sentences_test, labels_test = dfs_test['review_clean'].tolist(), dfs_test['label'].tolist()

    # 载入训练师保存的 word2idx 
    word2idx = joblib.load('output/word2idx.pkl')
    inputs_test, word2idx = make_data(sentences_test, word2idx)

    model = torch.load(config.model_path)

    dataset_test = SentimentDataset(inputs_test, labels_test)
    dataloader_test = Data.DataLoader(dataset_test, config.batch_size, True)

    fin_outputs = predict_fn(dataloader_test, model, device)
    output_indices = np.argmax(np.array(fin_outputs), axis=1)


    dfs = pd.DataFrame({"review_clean": sentences_test, "label": labels_test, "predict": list(output_indices)})
    dfs.to_csv('output/test_result.csv', sep='\t', index=False)


if __name__ == "__main__":
    run()