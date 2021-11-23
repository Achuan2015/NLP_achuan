from tqdm.auto import tqdm
from torch.nn import BCELoss
from sklearn.metrics import hamming_loss
import torch
    

def train_linear_fn(model, dataloader, optimizer, device):
    model.train()

    loss_fn = BCELoss()
    train_loss = 0
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        x, y = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = train_loss/len(dataloader)
    return train_loss
        

def eval_linear_fn(model, dataloader, device):
    model.eval()
    loss_fn = BCELoss()
    eval_loss = 0
    eval_accu = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, y = data[0].to(device), data[1].to(device)
            output = model(x)
            loss= loss_fn(output, y)
            eval_loss += loss.item()
            hl_val = hamming_loss(torch.round(output).cpu(), y.cpu())
            eval_accu += 1 - hl_val
        eval_loss = eval_loss / len(dataloader)
        eval_accu = eval_accu / len(dataloader)
    return eval_loss, eval_accu