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

    def __init__(self, encoded_data, labels):
        self.encoded_data = encoded_data
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encoded_data.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)