import torch.nn as nn
from transformers import BertPreTrainedModel


class SimCSEBert(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)