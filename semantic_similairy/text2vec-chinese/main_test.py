from text2vec import SentenceModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


def load_model(model_type="base"):
    model_map = {
        "base": "/data5/data-lakes/text2vec-base-chinese",
        "large": "/data5/data-lakes/text2vec-large-chinese",
        "m3e_base": "/data5/data-lakes/m3e-base"
    }
    model_path = model_map.get(model_type)
    return SentenceModel(model_path)

def encode_text_emb(model, corpus):
    return model.encode(corpus)

def evaluate(model, corpus1, corpus2):
    emb1 = encode_text_emb(model, corpus1) 
    emb2 = encode_text_emb(model, corpus2) 
    res = cosine_similarity(emb1, emb2) 
    return res


    

if __name__ == "__main__":
    #model_type = "large"
    model_type = "m3e_base"
    model = load_model(model_type) 
    corpus1 = ['今天天气很好，我要出去吃饭']
    corpus2 = ['今天天气很好，我不要出去吃饭']
    res = evaluate(model, corpus1, corpus2)
    print(res)
