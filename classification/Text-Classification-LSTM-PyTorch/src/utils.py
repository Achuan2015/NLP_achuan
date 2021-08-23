import numpy as np
import jieba
import torch

import config


stopwords = []

def make_test_data(sentence, word2idx):
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


def seg_word(sentence):
    seg_sentences = []
    for sent in sentence:
        sent_seg = [word for word in jieba.cut(sent) if word not in stopwords]
        seg_sentences.append(" ".join(sent_seg))
    return seg_sentences


def make_data(sentences, word2idx=None):
    inputs_id = []

    seg_sentences = seg_word(sentences)

    if word2idx is None:
        vocab_list = " ".join(seg_sentences).split(" ")
        vocab = list(set(vocab_list))
        vocab_size = len(vocab)
        word2idx = dict(zip(vocab, range(1, vocab_size + 1)))

    for sentence in seg_sentences:
        # sent_ids = [word2idx[word] for word in sentence.split(" ")]
        sent_ids = []
        for word in sentence.split(" "):
            if word not in word2idx:
                word2idx[word] = len(word2idx) + 1
            sent_ids.append(word2idx[word])
        inputs_id.append(sent_ids)
    return inputs_id, word2idx