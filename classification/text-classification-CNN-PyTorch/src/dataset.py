import torch.utils.data as Data
import torch
import config


class SentimentDataset(Data.Dataset):
    """
    增加对sentence 的 cut-off 和 padding 处理
    """

    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
    
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        if len(sentence) > 30:
            sentence = sentence[:30]
        else:
            sentence += [0] * (config.max_seq_len - len(sentence))
        return torch.LongTensor(sentence), torch.LongTensor([label])