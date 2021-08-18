import jieba


stopwords = []


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