from tqdm.auto import tqdm
import torch



def train_fn_second(dataloader, model, device, optimizer, criterion, scheduler):
    model.train()

    train_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_ids, mask_attention, token_type_ids, targets = data['input_ids'].to(device),\
            data['attention_mask'].to(device), data['token_type_id'].to(device), data['targets'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, mask_attention, token_type_ids)
        loss = criterion(outputs, targets.view(-1))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    train_loss = train_loss / len(dataloader)
    return train_loss

def eval_fn_second(dataloader, model, device, criterion):
    model.eval()

    eval_loss = 0.0
    fin_output = []
    fin_target = []
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_ids, mask_attention, token_type_ids, targets = data['input_ids'].to(device),\
            data['attention_mask'].to(device), data['token_type_id'].to(device), data['targets'].to(device)
        outputs = model(input_ids, mask_attention, token_type_ids)
        loss = criterion(outputs, targets.view(-1))
        eval_loss += loss.item()
        fin_output.extend(outputs.cpu().detach().numpu())
        fin_target.extend(targets.view(-1).cpu().detach().numpy())
    eval_loss = eval_loss / len(dataloader)
    return eval_loss, fin_output, fin_target


def train_fn(dataloader, model, device, optimizer):
    model.train()

    train_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_ids, token_type_id, attention_mask, target = data['input_ids'].to(device), data['token_type_id'].to(device), \
            data['attention_mask'].to(device), data['targets'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_id, labels=target)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * input_ids.size(0)
    train_loss = train_loss / len(dataloader.sampler)
    return train_loss


def eval_fn(dataloader, model, device):
    model.eval()

    eval_loss = 0.0
    fin_output = []
    fin_target = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids, token_type_id, attention_mask, target = data['input_ids'].to(device), data['token_type_id'].to(device), \
                data['attention_mask'].to(device), data['targets'].to(device)
            outputs =  model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_id, labels=target)
            loss = outputs[0]
            logits = outputs[1]
            fin_output.extend(logits.cpu().detach().numpy().tolist())
            fin_target.extend(target.view(-1).cpu().detach().numpy().tolist())
            eval_loss += loss.item() * input_ids.size(0)
    eval_loss = eval_loss / len(dataloader.sampler)
    return eval_loss, fin_output, fin_target