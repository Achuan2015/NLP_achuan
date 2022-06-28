import pandas as pd
import jieba
from sklearn.model_selection import train_test_split

import csv
import logging


jieba.setLogLevel(logging.ERROR)


def convert2fasttext_format(path):
    '''
    path: data file relative location
    output: fasttext train file
    __label__chocolate American equivalent for British chocolate terms
    __label__baking __label__oven __label__convection Fan bake vs bake
    __label__sauce __label__storage-lifetime __label__acidity __label__mayonnaise Regulation and balancing of readymade pac
    '''
    dfs = pd.read_csv(path, sep='\t')
    dfs['label'] = dfs['intent_name'].apply(lambda x: convert2fasttext_label(x))
    dfs['text_token'] = dfs['text'].apply(lambda x: text_token(x, counter, stopwords))
    dfs = dfs[['label', 'text_token']]
    dfs_train, dfs_test = train_test_split(dfs, test_size=0.1, random_state=42, shuffle=True)
    dfs_train.to_csv('fasttext-tips/data/bot_corpus.train_v2', sep=',', header=False, index=False)
    dfs_test.to_csv('fasttext-tips/data/bot_corpus.test_v2', sep=',', header=False, index=False)


def convert2fasttext_label(text):
    if not text:
        return
    if isinstance(text, str):
        if text.startswith('('):
            return ' '.join(f'__label__{l}' for l in eval(text))
        return f'__label__{text}'

def text_token(text, counter, stopwords):
    items = [w for w in jieba.cut(text) if w not in stopwords]
    counter.update(items)
    return ' '.join(items)

def read_stopwords(path):
    stopwords = []
    with open(path, 'r') as f:
        for line in f:
            w = line.strip()
            if w:
                stopwords.append(w)
    return stopwords
        

if __name__ == '__main__':
    from collections import Counter
    path = 'fasttext-tips/data/data_label_corpus_20220627_v1.csv'
    stopword_path = 'fasttext-tips/data/stopword_cn.txt'
    word_freq_path = 'fasttext-tips/data/words_freq.xlsx'
    stopwords = read_stopwords(stopword_path)
    counter = Counter()
    convert2fasttext_format(path)
    word_freq = counter.most_common(100)
    print(word_freq)
    import pandas as pd
    dfs = pd.DataFrame(word_freq, columns=['word', '词频'])
    dfs.to_excel(word_freq_path, index=False)
