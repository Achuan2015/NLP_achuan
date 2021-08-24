from tqdm.auto import tqdm


def train_fn(dataloader, model, device, optimizer):
    model.train()

    train_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_ids, token_type_id, attention_mask, target = data['input_ids'].to(device), data['token_type_id'].to(device), \
            data['attention_mask'].to(device), data['targets'].to(device)
        print('input_ids', input_ids.shape)
        print('token_type_id', token_type_id.shape)
        print('attention_mask', attention_mask.shape)
        print('target', target.shape)
        optimizer.zero_grad()
        outputs =  model(input_ids, attention_mask, token_type_id, labels=target)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * input_ids.size(0)
    train_loss = train_loss / len(dataloader.sampler)
    return train_loss


def eval_fn(dataloader, model, device, optimzier):
    pass