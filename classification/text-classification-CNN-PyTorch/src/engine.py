import torch
from tqdm.auto import tqdm


def train_fn(dataloader, model, device, optimizer, criterion):
    model.train()
    
    train_loss = 0.0
    for i, data in tqdm(enumerate(dataloader),total=len(dataloader)):
        input, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * input.size(0)
    train_loss = train_loss / len(dataloader.sampler)
    return train_loss


def eval_fn(dataloader, model, device, criterion):
    model.eval()
    
    fin_outputs = []
    fin_targets = []
    valid_loss = 0.0
    for _, data in tqdm(enumerate(dataloader),total=len(dataloader)):
        input, target = data[0].to(device), data[1].to(device)
        output = model(input)
        fin_outputs.extend(output.cpu().detach().numpy().tolist())
        fin_targets.extend(target.view(-1).cpu().detach().numpy().tolist())
        loss = criterion(output, target.view(-1))
        valid_loss += loss.item() * input.size(0)
    valid_loss = valid_loss / len(dataloader.sampler)
    return fin_outputs, fin_targets, valid_loss