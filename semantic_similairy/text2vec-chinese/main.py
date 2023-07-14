from transformers import BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import pandas as pd
from model import BertForCoSentNetwork
from transformers import BertConfig
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class newDataset(Dataset):
    """
    reference:https://huggingface.co/transformers/custom_datasets.html?highlight=datasets
    """

    def __init__(self, encoded_corpus):
        self.encoded_corpus = encoded_corpus

    def __getitem__(self, idx):
        return {
            'encoded_corpus': {key:val[idx] for key, val in self.encoded_corpus.items()}
        }

    def __len__(self):
        return len(self.encoded_corpus['input_ids'])


def load_model(model_path):
    torch.cuda.set_device(2)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(model_path)
    model = BertForCoSentNetwork(model_path, config)
    model = model.to(device)
    return model, tokenizer, device

def encode_text_emb(model, corpus):
    return model.encode(corpus)

def read_corpus(data_path):
    dfs = pd.read_excel(data_path)
    intent_ids = dfs['意图ID'].drop_duplicates().tolist()
    intent_dataframe = dfs.drop_duplicates(subset='意图ID', keep="first")
    intentid2name = dict(zip(intent_dataframe['意图ID'], intent_dataframe['意图名称']))
    intent2corpus = {}
    for intent_id in intent_ids:
        intent2corpus[intent_id] = dfs[dfs['意图ID'] == intent_id]['语料文本'].tolist()
    return intent_ids, intentid2name, intent2corpus

def get_vector(model, tokenizer, corpus, device):
    data_array = []
    tokenizer_corpus = tokenizer(corpus, padding=True, truncation=True, max_length=128, return_tensors="pt")
    corpus_dataset = newDataset(tokenizer_corpus)
    dataloader = DataLoader(corpus_dataset, batch_size=32, shuffle=False)
    for d in dataloader:
        # print(d['encoded_corpus'])
        encoded_input = {k:v.to(device) for k,v in d['encoded_corpus'].items()}
        embeds = model.encode(encoded_input)
        embeds_array = embeds.detach().cpu().numpy()
        data_array.append(embeds_array)
    data_arrary_cat = np.concatenate(data_array, axis=0)
    return data_arrary_cat


if __name__ == "__main__":
    model_path = "/data/projects/fine-tuning-chinese-bert-with-transformers/outputs/chinese_wwm_pytorch-50-5-cosent-v1"
    data_path = "Result_8.xlsx"
    intent2embedding = {}
    intent_ids, intentid2name, intent2corpus = read_corpus(data_path)

    model, tokenizer, device = load_model(model_path)

    data = {'corpus':[], 'intent_id':[], 'intent_name':[]}
    intentnames = list(intentid2name.values())
    for iname in intentnames:
        data[iname] = []

    intent_similarites = {}
    print(f'意图长度: {len(intent_ids)}') 
    for intent_id in intent2corpus:
        intentcorpus = intent2corpus[intent_id]
        intent2embedding[intent_id] = get_vector(model, tokenizer, intentcorpus, device)

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
    dfs_res.to_excel(f'./result_cosine_newbot_base_v8_{threshold}.xlsx', index=False)