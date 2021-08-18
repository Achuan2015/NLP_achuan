import torch
from tqdm.auto import tqdm


def train_fn(dataloader, model, device, optimizer, criterion):
    model.train()
    
    for i, data in tqdm(enumerate(dataloader),total=len(dataloader)):
        print("bi", i)
        input, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target.view(-1))
        loss.backward()
        optimizer.step()


def eval_fn(dataloader, model, device):
    model.eval()
    
    fin_outputs = []
    fin_targets = []
    for bi, data in tqdm(enumerate(dataloader),total=len(dataloader)):
        input, target = data[0].to(device), target.to(device)
        output = model(input)
        fin_outputs.append(output)
        fin_targets.append(target)
    return fin_outputs, fin_targets