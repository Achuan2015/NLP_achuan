from loguru import logger
import requests


class Http_Client(object):

    def __init__(self, **kw):
        self.logger = logger
        self.url = None
        self.headers = None

    def init(self, **http_params):
        pipeline_host = http_params.get('PIPELINE_HOST', "81.69.154.243")
        pipeline_port = http_params.get('PIPELINE_PORT', 80)
        auth = http_params.get('AUTH', 'dc5976de')
        self.url = "http://{}:{}/models/pipeline/transfer".format(pipeline_host, pipeline_port)
        self.headers = {'Auth': auth, 'Content-Type': 'application/json'}

    def query(self, pipeline_name, params, timeout=10):
        response = {}
        data = {
            "params": params,
            "pipeline_name": pipeline_name
        }
        response = requests.post(self.url, headers=self.headers, json=data, timeout=timeout)
        response = response.json()
        return response

    def query_without_pipeline(self, url, data, header, timeout=60):
        response = requests.post(url, headers=header, json=data, timeout=timeout)
        response = response.json()
        return response
    
    def query_external_service(self, url, data, timeout=5):
        response = requests.post(url, json=data, timeout=timeout)
        response = response.json()
        return response


http_client = Http_Client()
