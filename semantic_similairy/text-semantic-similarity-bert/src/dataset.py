from torch.utils.data import Dataset
import torch
import config


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