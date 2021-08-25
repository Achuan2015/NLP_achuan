"""
模型直接使用Transformer自带的 BertForSequenceClassification 模型。 BertForSequenceClassification里面内嵌了，BertModel 在BertModel的output的基础上，拿到经过pooler的
[CLS] token的表征，经过一层 dropout 之后进行线性分类进行预测，并且内嵌的loss function 会根据label的输出直接输出 loss。

BertModel 的output 
    -> sequence_output (原生[CLS] token 表征)，
    pooled_output(经过BertPooler 层映射过的[CLS]token 表征)，
    （hidden_states) 各层的hidden_states, 
    (attentions) 各层的attention value

BertForSequenceClassification 输出:
    -> loss： 损失函数的值
    -> logits (线性分类的结果) B * num_classes
       hidden_states： bertmodel 的输出
       attention: bertmodel 的输出
"""

import torch.nn as nn
from transformers import BertModel


class BertForSequenceClassification(nn.Module):

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.model_path)
        self.dropout = config.hidden_dropout_prob
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooler_output = self.dropout(outputs.pooler_output)
        output = self.classifier(pooler_output)
        return output