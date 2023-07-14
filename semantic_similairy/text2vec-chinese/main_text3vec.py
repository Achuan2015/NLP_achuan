from text2vec import SentenceModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


def load_model(model_type="base"):
    model_map = {
        "base": "/data5/data-lakes/text2vec-base-chinese",
        "large": "/data5/data-lakes/text2vec-large-chinese"
    }
    model_path = model_map.get(model_type)
    return SentenceModel(model_path)

def read_corpus(data_path):
    dfs = pd.read_excel(data_path)
    intent_ids = dfs['意图ID'].drop_duplicates().tolist()
    intent_dataframe = dfs.drop_duplicates(subset='意图ID', keep="first")
    intentid2name = dict(zip(intent_dataframe['意图ID'], intent_dataframe['意图名称']))
    intent2corpus = {}
    for intent_id in intent_ids:
        intent2corpus[intent_id] = dfs[dfs['意图ID'] == intent_id]['语料文本'].tolist()
    return intent_ids, intentid2name, intent2corpus

def encode_text_emb(model, corpus):
    return model.encode(corpus)
    

if __name__ == "__main__":
    model_type = "large"
    model = load_model(model_type) 
    data_path = "Result_5.xlsx"
    intent2embedding = {}
    intent_ids, intentid2name, intent2corpus = read_corpus(data_path)
    intent_similarites = {}
    print(f'意图长度: {len(intent_ids)}') 
    #print(intent_ids)
    #print(intent2corpus)
    for intent_id in intent2corpus:
        print(intent_id)
        intent2embedding[intent_id] = encode_text_emb(model, intent2corpus[intent_id]) 
   
    threshold = 0.6
    for i in intent_ids:
        for j in intent_ids:
            print(i, j)
            cur_res = cosine_similarity(intent2embedding[i], intent2embedding[j])
            if i !=j:
                index = np.unravel_index(np.argmax(cur_res, axis=None), cur_res.shape)
                score = cur_res[index]
            else:
                index = np.unravel_index(np.argmin(cur_res, axis=None), cur_res.shape)
                score = cur_res[index] 
            if score > 0.6:
                intent_similarites[i] = intent_similarites.get(i, []) + [(j, score, 1)]
            else:
                intent_similarites[i] = intent_similarites.get(i, []) + [(j, score, 0)]
    output_data = []
    for ii in intent_similarites:
        name = intentid2name[ii]
        output_data.append((str(ii), name, intent_similarites[ii]))
    dfs_res = pd.DataFrame(output_data, columns=['intent_id', 'intent_name', 'cosine_result'])
    dfs_res.to_excel('./result_cosine_large_v1.xlsx', index=False)
        
