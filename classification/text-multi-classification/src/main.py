'''
模型训练的main函数
'''

import torch
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim import Adam

from dataset import YeastDataset
from torch.utils.data import DataLoader

from model import LinearModel
from engine import eval_linear_fn, train_linear_fn


seed_val = 42
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, _ = arff.loadarff('data/yeast/yeast-train.arff')
    # 将data进行索引
    dfs = pd.DataFrame(data)
    x = dfs.iloc[:, :103].values
    y = dfs.iloc[:, 103:].values
    X_train, X_eval, y_train, y_eval = train_test_split(x, y, test_size=0.2)
    X_train, y_train = X_train.astype(np.float64), y_train.astype(np.float64)
    X_eval, y_eval = X_eval.astype(np.float64), y_eval.astype(np.float64)
    dataset_train = YeastDataset(X_train, y_train)
    dataset_eval = YeastDataset(X_eval, y_eval)
    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
    dataloader_eval = DataLoader(dataset_eval, batch_size=16, shuffle=True)
    model = LinearModel().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    for num in range(num_epochs):
        train_loss = train_linear_fn(model, dataloader_train, optimizer, device)
        eval_loss, eval_accu = eval_linear_fn(model, dataloader_eval, device)
        print(f'num_epoch: {num + 1}, train_loss: {train_loss}, eval_loss: {eval_loss}, eval_accu:{eval_accu}')
    

if __name__ == '__main__':
    main()