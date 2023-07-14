from loguru import logger
import sys
import time

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

from transformers import BertTokenizer
from handler_util import get_data_manager
from hrtps_file_manager import EnvirEnum
from http_client import http_client


class Sentence_Tokenizer:

    MAX_LENGTH=256

    def __init__(self):
        self.logger = logger
        self.tokenizer = None
    
    def init_with_cos(self, model_name, model_version):
        data_manager = get_data_manager()
        model_parcel = data_manager.load_model(model_name=model_name, model_version=model_version, environment=EnvirEnum.MASTER)
        model_param = model_parcel.model_obj_bin
        self.tokenizer = model_param.get('tokenizer')

    def encode_query(self, query):
        encoded_query = self.tokenizer(query, padding=True, truncation=True, max_length=self.MAX_LENGTH)
        encoded_query = [v for _, v in encoded_query.items()]
        return encoded_query

    def encode(self, query, url, batch_size=32):
        if not query:
            return []
        all_embeddings = []
        encoded_query = {}
        t1 = time.time()
        for start_index in range(0, len(query), batch_size):
            query_batch = query[start_index: start_index + batch_size]
            encoded_query_batch = self.encode_query(query_batch)
            encoded_query[start_index] = encoded_query_batch
        
        t2 = time.time()
        self.logger.info(f'query:{query} 分词耗时: {t2 - t1}')
        
        sentence_inference_executor = OrderedDict()
        with ThreadPoolExecutor() as executor:
            for index, eq in encoded_query.items():
                param = {
                    'query':eq
                }
                sentence_inference_executor[index] = executor.submit(http_client.query_external_service, url, param)
        t3 = time.time()
        if sentence_inference_executor:
            for index, resp in sentence_inference_executor.items():
                try:
                    result = resp.result()
                    query_vector = result.get('query_vector', [])
                    all_embeddings.extend(query_vector)
                except Exception as e:
                    self.logger.exception(e, "query handler: get external service result occur exception")
        t4 = time.time()
        self.logger.info(f'分词 + t1平台服务请求耗时: {t4 - t1}')
        return all_embeddings


sentence_tokenizer = Sentence_Tokenizer()

#if __name__ == '__main__':
#    model_name, model_version= "TinyBert-cosent", 5
#    url='http://service-nhcg0hv1-1308945662.sh.apigw.tencentcs.com:80/tione/v1/models/m:predict'
#    query = ['你知道公司怎么去吗？', '道公司怎么去吗？']
#    sentence_tokenizer = Sentence_Tokenizer()
#    sentence_tokenizer.init_with_cos(model_name, model_version)
#    query_emebdding = sentence_tokenizer.encode(query, url)
#    print(len(query), len(query_emebdding))
#    assert len(query) == len(query_emebdding), '注意：query 与 query_embedding 长度不相同'
