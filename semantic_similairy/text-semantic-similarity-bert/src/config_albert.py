from transformers import BertTokenizer

max_seq_len=40
epochs=7
batch_size=1024
weight_decay=1e-3
hidden_dropout_prob=0.3
hidden_size=312
feature_size=100
window_sizes=[2,3,4,5]
num_labels=1
learning_rate=2e-5
model_path="input/albert_chinese_tiny//"
tokenizer = BertTokenizer.from_pretrained(model_path)
output_path="output/albert_chinese_tiny/"
