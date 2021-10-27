import torch.nn as nn
import torch
from tqdm.auto import tqdm


def cosine_loss_fn(q_emb, c_emb, label):
    # 因为 labels的取值是[0, 1], 而CosineEmbeddingLoss中要求label取值范围在[-1, 1]之间
    # y = 2 * label - 1
    return nn.CosineEmbeddingLoss(margin=0.5, reduction='mean')(q_emb, c_emb, 2 * label - 1)


def ce_loss_fn(device):
    return nn.CrossEntropyLoss().to(device)


def mse_loss_fn(output, label):
    return nn.MSELoss()(output.view(-1), label.view(-1))


def convert2device(x, device):
    return {key:value.to(device) for key, value in x.items()}



def train_cosine_fn(dataloader, model, device, optimizer, scheduler):
    model.train()

    train_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        query, candidate, label = data['query'], data['candidate'], data['label']
        query, candidate, label = convert2device(query, device), convert2device(candidate, device), label.to(device)
        optimizer.zero_grad()
        query_emb, candidate_emb = model(query, candidate)
        loss = cosine_loss_fn(query_emb, candidate_emb, label.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
    train_loss = train_loss / len(dataloader)
    return train_loss

def eval_cosine_fn(dataloader, model, device):
    model.eval()

    eval_loss = 0.0
    eval_accu = 0.0
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    threshold = 0.5
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            query, candidate, label = data['query'], data['candidate'], data['label']
            query, candidate, label = convert2device(query, device), convert2device(candidate, device), label.to(device)
            query_emb, candidate_emb = model(query, candidate)
            loss = cosine_loss_fn(query_emb, candidate_emb, label.view(-1))
            
            cos_output = cos(query_emb, candidate_emb)
            cos_output = (cos_output > threshold).long()
            
            accuracy = torch.sum(cos_output == label.view(-1))/ cos_output.size(0)
            eval_accu += accuracy.item()
            eval_loss += loss.item()
    eval_loss = eval_loss / len(dataloader)
    eval_accu = eval_accu / len(dataloader)
    return eval_loss, eval_accu

def train_mse_fn(dataloader, model, device, optimizer, scheduler):
    model.train()

    train_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input, label = data['input'], data['label']
        input, label = convert2device(input, device), label.to(device)
        optimizer.zero_grad()
        outputs = model(input)
        loss = mse_loss_fn(outputs, label)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
    train_loss = train_loss / len(dataloader)
    return train_loss

def eval_mse_fn(dataloader, model, device):
    model.eval()

    eval_loss = 0.0
    eval_accu = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            input, label = data['input'], data['label']
            input, label = convert2device(input, device), label.to(device)
            logits = model(input)
            loss = mse_loss_fn(logits, label)
            loss = loss.mean()
            accuracy = torch.sum(label.view(-1) == (logits.view(-1) > 0.5).long()) / label.size(0)
            eval_accu += accuracy.item()
            eval_loss += loss.mean().item()
    eval_loss = eval_loss / len(dataloader)
    eval_accu = eval_accu / len(dataloader)
    return eval_loss, eval_accu

def train_fn(dataloader, model, device, optimizer, scheduler):
    model.train()
    
    train_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input, label = data['input'], data['label']
        input, label = convert2device(input, device), label.to(device)
        optimizer.zero_grad()
        outputs = model(**input, labels=label)
        loss = outputs[0].mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
    train_loss = train_loss / len(dataloader)
    return train_loss

def train_simCSE_fn(dataloader, model, device, optimizer, scheduler):
    model.train()

    train_loss = 0.0


def eval_simCSE_fn(dataloader, model, device):
    model.eval()
    
    eval_loss = 0.0

def eval_fn(dataloader, model, device):
    model.eval()

    eval_loss = 0.0
    eval_accu = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            input, label = data['input'], data['label']
            input, label = convert2device(input, device), label.to(device)
            outputs = model(**input, labels=label)
            loss, logits = outputs[0], outputs[1]
            accuracy =  torch.sum(label.view(-1) == (logits.view(-1) > 0.5).long()) / label.size(0)
            eval_accu += accuracy.item()
            eval_loss += loss.mean().item()
    eval_loss = eval_loss / len(dataloader)
    eval_accu = eval_accu / len(dataloader)
    return eval_loss, eval_accu