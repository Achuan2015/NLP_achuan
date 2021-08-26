from transformers import  BertPreTrainedModel
from transformers import BertModel
import torch

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


if __name__ == "__main__":
    from transformers import BertConfig
    import config

    bert_config = BertConfig.from_pretrained("input/bert-base-chinese")
    model_path = config.model_path
    output_path = config.output_path
    model = BertForSiameseNetwork(bert_config, model_path)
    # model.save_pretrained(output_path)
    query = "明天天气很好"
    output = config.tokenizer(query, padding=True, truncation=True, max_length=128, return_tensors="pt")
    print(output)