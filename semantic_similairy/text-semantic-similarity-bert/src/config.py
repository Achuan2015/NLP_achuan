from transformers import BertTokenizer

max_seq_len=40
epochs=7
batch_size=512
weight_decay=1e-3
hidden_dropout_prob=0.3
hidden_size=768
feature_size=100
window_sizes=[2,3,4]
num_labels=1
learning_rate=2e-5
model_path="input/bert-base-chinese/"
tokenizer = BertTokenizer.from_pretrained(model_path)
output_path="output/bert-base-chinese"
