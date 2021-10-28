from torch.utils.data import Dataset
import torch
import config


def collate_simCSE(data):
    input_encoded = {'input_ids':[], 'token_type_ids': [], 'attention_mask': []}
    for i in data:
        input_encoded['input_ids'].append(i['input_ids'])
        input_encoded['token_type_ids'].append(i['token_type_ids'])
        input_encoded['attention_mask'].append(i['attention_mask'])
    ids = torch.arange(len(input_encoded['input_ids']))
    labels = ids + 1 - ids % 2 * 2
    input_encoded = {key: torch.LongTensor(torch.stack(val, 0)) for key, val in input_encoded.items()}
    return input_encoded, torch.LongTensor(labels)

class SiameseDataset(Dataset):
    """
    reference:https://huggingface.co/transformers/custom_datasets.html?highlight=datasets
    """

    def __init__(self, querys, candidates, labels):
        self.querys = querys
        self.candidates = candidates
        self.labels = labels

    def __getitem__(self, idx):
        query, candidate = self.querys[idx], self.candidates[idx]
        query_encoded = config.tokenizer.encode_plus(
            query,
            add_special_tokens=True,
            max_length=config.max_seq_len,
            padding="max_length",
            truncation="longest_first"
        )
        candidate_encoded = config.tokenizer.encode_plus(
            candidate,
            add_special_tokens=True,
            max_length=config.max_seq_len,
            padding="max_length",
            truncation="longest_first"
        )
        
        return {
            "query": {key:torch.LongTensor(val) for key, val in query_encoded.items()},
            "candidate": {key:torch.LongTensor(val) for key, val in candidate_encoded.items()},
            "label": torch.LongTensor([self.labels[idx]])
        }

    def __len__(self):
        return len(self.labels)

class SimCSEDataset(Dataset):

    def __init__(self, sents1, sents2):
        self.sents1 = sents1
        self.sents2 = sents2
    
    def __getitem__(self, idx):
        sent1 = self.sents1[idx]
        sent2 = self.sents2[idx]
        input_encoded = config.tokenizer.encode_plus(
            sent1, sent2,
            add_special_tokens=True,
            max_length=config.max_seq_len,
            padding="max_length",
            truncation="longest_first"
        )
        input_encoded = {key:torch.LongTensor(val) for key, val in input_encoded.items()}
        return input_encoded
    
    def __len__(self):
        return len(self.sents1)
    

class CrossEncodeDataset(Dataset):

    def __init__(self, querys, candidates, labels):
        self.querys = querys
        self.candidates = candidates
        self.labels = labels

    def __getitem__(self, idx):
        query = self.querys[idx]
        candidate = self.candidates[idx]
        input_encoded = config.tokenizer.encode_plus(
            query, candidate,
            add_special_tokens=True,
            max_length=config.max_seq_len,
            padding="max_length",
            truncation="longest_first"
        )
        return {
            "input": {key:torch.LongTensor(val) for key, val in input_encoded.items()},
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }

    def __len__(self):
        return len(self.labels)