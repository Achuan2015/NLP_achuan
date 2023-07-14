import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sentence_tokenizer import sentence_tokenizer


def read_corpus(data_path):
    dfs = pd.read_excel(data_path)
    intent_ids = dfs['意图ID'].drop_duplicates().tolist()
    intent_dataframe = dfs.drop_duplicates(subset='意图ID', keep="first")
    intentid2name = dict(zip(intent_dataframe['意图ID'], intent_dataframe['意图名称']))
    intent2corpus = {}
    for intent_id in intent_ids:
        intent2corpus[intent_id] = dfs[dfs['意图ID'] == intent_id]['语料文本'].tolist()
    return intent_ids, intentid2name, intent2corpus

def encode_text_emb(model, corpus, url):
    return model.encode(corpus, url)
 

if __name__ == "__main__":
    model_name, model_version= "TinyBert-cosent", 5
    url='http://service-nhcg0hv1-1308945662.sh.apigw.tencentcs.com:80/tione/v1/models/m:predict'
    sentence_tokenizer.init_with_cos(model_name, model_version)

    data_path = "Result_8.xlsx"
    intent2embedding = {}
    intent_ids, intentid2name, intent2corpus = read_corpus(data_path)

    data = {'corpus':[], 'intent_id':[], 'intent_name':[]}
    intentnames = list(intentid2name.values())
    for iname in intentnames:
        data[iname] = []

    intent_similarites = {}
    print(f'意图长度: {len(intent_ids)}') 
    for intent_id in intent2corpus:
       intent2embedding[intent_id] = encode_text_emb(sentence_tokenizer, intent2corpus[intent_id], url) 

    threshold = 0.80
    for i in intent_ids:
        i_corpus = intent2corpus[i]
        i_embedding = intent2embedding[i]
        for ic, ie in zip(i_corpus, i_embedding):
            data['corpus'].append(ic)
            data['intent_id'].append(str(i))
            data['intent_name'].append(intentid2name[i])

            for j in intent_ids:
                j_corpus = intent2corpus[j]
                j_embedding = intent2embedding[j]
                j_intentname = intentid2name[j]

                #cur_res = cosine_similarity(intent2embedding[i], intent2embedding[j])
                cur_res = cosine_similarity([ie], j_embedding)

                max_index = np.unravel_index(np.argmax(cur_res, axis=None), cur_res.shape)
                min_index = np.unravel_index(np.argmin(cur_res, axis=None), cur_res.shape)
                max_score = cur_res[max_index]
                max_text = j_corpus[max_index[1]]
                min_score = cur_res[min_index]
                min_text = j_corpus[min_index[1]]
        
                if max_score >= threshold:
                    #intent_similarites[i] = intent_similarites.get(i, []) + [(j, min_score, max_score, 1)]
                    data[j_intentname].append([{'min_score': min_score, 'min_text': min_text}, 
                                               {'max_score': max_score, 'max_text': max_text},{'是否通过':'pass'}])
                else:
                    #intent_similarites[i] = intent_similarites.get(i, []) + [(j, min_score, max_score, 0)]
                    data[j_intentname].append([{'min_score': min_score, 'min_text': min_text}, 
                                              {'max_score': max_score, 'max_text': max_text},{'是否通过':'false'}])

    dfs_res = pd.DataFrame(data)
    dfs_res.to_excel(f'./result_cosine_bot_base_v8_{threshold}.xlsx', index=False)
