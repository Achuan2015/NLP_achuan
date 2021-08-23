from tqdm.auto import tqdm
import torch
import numpy as np


def train_fn(dataloader, model, device, optimizer, criterion):
    model.train()

    train_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(input, device)
        loss = criterion(output, target.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * input.size(0)
    train_loss = train_loss / len(dataloader.sampler)
    return train_loss    


def eval_fn(dataloader, model, device, criterion):
    """
    输出: 
    """
    model.eval()
    fin_outputs, fin_targets = [], []
    eval_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input, target = data[0].to(device), data[1].to(device)
        output = model(input, device)
        loss = criterion(output, target.view(-1))
        eval_loss += loss.item() * input.size(0)
        fin_outputs.extend(output.cpu().detach().numpy().tolist())
        fin_targets.extend(target.view(-1).cpu().detach().numpy().tolist())
    eval_loss = eval_loss / len(dataloader.sampler)
    return fin_outputs, fin_targets, eval_loss
