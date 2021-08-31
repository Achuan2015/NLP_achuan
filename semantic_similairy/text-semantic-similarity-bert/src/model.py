from transformers import  BertPreTrainedModel
from transformers import BertModel
import torch
import torch.nn as nn


class BertForSiameseNetwork(BertPreTrainedModel):
    """
    reference: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py
    
    继承BertPreTrainedModel 是为了方便使用save_pretrained 这个函数
    """

    def __init__(self, model_config, model_path):
        super().__init__(model_config)
        self.bert = BertModel.from_pretrained(model_path)

    def encode(self, encoded_input):
        model_output = self.bert(**encoded_input)
        input_embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return input_embedding

    def mean_pooling(self, model_output, attention_mask):
        token_embedding = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embedding.size()).float()
        # sum columns
        sum_embddings = torch.sum(token_embedding * input_mask_expanded, 1)
        # sum_mask
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embddings / sum_mask

    def forward(self, encoded_sent1, encoded_sent2):
        sent1_embedding = self.encode(encoded_sent1)
        sent2_embedding = self.encode(encoded_sent2)
        return sent1_embedding, sent2_embedding


class BertForPoolingNetwork(BertPreTrainedModel):
    """
    reference: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py
    
    继承BertPreTrainedModel 是为了方便使用save_pretrained 这个函数
    """

    def __init__(self, model_config, model_path):
        super().__init__(model_config)
        self.bert = BertModel.from_pretrained(model_path)

    def encode(self, encoded_input):
        model_output = self.bert(**encoded_input)
        input_embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return input_embedding

    def mean_pooling(self, model_output, attention_mask):
        token_embedding = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embedding.size()).float()
        # sum columns
        sum_embddings = torch.sum(token_embedding * input_mask_expanded, 1)
        # sum_mask
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embddings / sum_mask

    def forward(self, input_encoded):
        pooler_embedding = self.encode(input_encoded)
        return pooler_embedding


class BertCNNForClassification(BertPreTrainedModel):

    def __init__(self, bert_config, config):
        super().__init__(bert_config)
        self.bert = BertModel.from_pretrained(config.model_path)
        self.convs = nn.ModuleList(
            [
               nn.Sequential(nn.Conv1d(config.hidden_size, config.feature_size, kernel_size=h),
               nn.ReLU(), nn.MaxPool1d(config.max_seq_len - h + 1)) for h in config.window_sizes
            ]
        )
        self.classifier = nn.Linear(config.feature_size * len(config.window_sizes), config.num_labels)
    
    def cell_textcnn(self, x_emb):
        x_emb = x_emb.transpose(1, 2) # batch_size * hidden_size * max_seq
        out = [conv1d(x_emb) for conv1d in self.convs] # batch_size * feature_size * 1
        out = torch.cat(out, dim=1) # batch_size * feature_size * 1
        # flatten the out
        out = out.view(-1, out.size(1))
        return out

    def forward(self, input_encoded):
        outputs_bert = self.bert(**input_encoded)
        output_sequence = outputs_bert[0]
        cnn_output = self.cell_textcnn(output_sequence)
        output = self.classifier(cnn_output)
        return output

class BertCNNForClassificationNew(BertPreTrainedModel):
    """
    考虑将bert作为base模型，作为特征提取器，并且不改动参数，仅仅改动分类器的的参数。
    """

    def __init__(self, bert_config, config):
        super(BertCNNForClassificationNew, self).__init__(bert_config)
        self.bert = BertModel.from_pretrained(config.model_path)
        for p in self.parameters():
            p.requires_grad = False
        self.convs = nn.ModuleList(
            [
               nn.Sequential(nn.Conv1d(config.hidden_size, config.feature_size, kernel_size=h),
               nn.ReLU(), nn.MaxPool1d(config.max_seq_len - h + 1)) for h in config.window_sizes
            ]
        )
        self.classifier = nn.Linear(config.feature_size * len(config.window_sizes), config.num_labels)
    
    def cell_textcnn(self, x_emb):
        x_emb = x_emb.transpose(1, 2) # batch_size * hidden_size * max_seq
        out = [conv1d(x_emb) for conv1d in self.convs] # batch_size * feature_size * 1
        out = torch.cat(out, dim=1) # batch_size * feature_size * 1
        # flatten the out
        out = out.view(-1, out.size(1))
        return out

    def forward(self, input_encoded):
        outputs_bert = self.bert(**input_encoded)
        output_sequence = outputs_bert[0]
        cnn_output = self.cell_textcnn(output_sequence)
        output = self.classifier(cnn_output)
        return output