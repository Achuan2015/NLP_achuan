import jieba

stopwords = []


def seg_word(sentence):
    seg_sentences = []
    for sent in sentence:
        sent_seg = [word for word in jieba.cut(sent) if word not in stopwords]
        seg_sentences.append(" ".join(sent_seg))
    return seg_sentences


def make_data(sentences):
    inputs_id = []

    seg_sentences = seg_word(sentences)
    vocab_list = " ".join(seg_sentences).split(" ")
    vocab = list(set(vocab_list))
    vocab_size = len(vocab)
    word2idx = dict(zip(vocab, range(vocab_size)))
    idx2word = dict(zip(range(vocab_size), vocab))

    for sentence in seg_sentences:
        sent_ids = [word2idx[word] for word in sentence.split(" ")]
        inputs_id.append(sent_ids)
    return inputs_id, word2idx, idx2word, vocab, vocab_size
    