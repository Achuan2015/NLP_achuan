from transformers import BertTokenizer

max_seq_len=40
epochs=3
batch_size=384
weight_decay=1e-3
hidden_dropout_prob=0.3
hidden_size=312
feature_size=100
window_sizes=[2,3,4]
num_labels=1
learning_rate=2e-5
model_path="input/TinyBERT_4L_zh/"
tokenizer = BertTokenizer.from_pretrained(model_path)
output_path="output/TinyBERT_4L_zh"