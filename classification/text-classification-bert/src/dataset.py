import torch
import config
from torch.utils.data import Dataset, dataset


class MyDataset(Dataset):

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input = self.inputs[idx]
        target = self.targets[idx]
        encode_output = config.tokenizer.encode_plus(
            input,
            add_special_tokens=True,
            max_length=config.max_seq_len,
            padding="max_length",
            truncation="longest_first"
        )
        input_ids = encode_output["input_ids"]
        attentin_mask = encode_output["attention_mask"]
        token_type_ids = encode_output["token_type_ids"]

        return {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attentin_mask),
            "token_type_id": torch.LongTensor(token_type_ids),
            "targets": torch.LongTensor([target])
        }


if __name__ == "__main__":
    inputs = ['今天天气很好']
    targets = [1]
    dataset = MyDataset(inputs, targets)
    print(len(dataset))
    print(dataset[0])
    