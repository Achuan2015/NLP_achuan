import numpy as np
import torch
import config


def make_data(sentence, word2idx):
    max_seq_len = config.max_seq_len
    num_classes = config.num_classes
    input_batch, target_batch = [], []

    words = sentence.split()
    for i in range(max_seq_len - 1):
        input = [word2idx[words[n]] for n in range(i + 1)]
        input = input + [0] * (max_seq_len - len(input))
        target = word2idx[words[i + 1]]
        input_batch.append(np.eye(num_classes)[input])
        target_batch.append(target)
    return torch.Tensor(input_batch), torch.LongTensor(target_batch)