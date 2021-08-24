from transformers import BertTokenizer

model_path="input/roberta_zh_large"
max_seq_len=30
batch_size=64
hidden_dropout_prob=0.3
weight_decay=1e-3
epochs=3
learning_rate=1e-3
num_labels=3
tokenizer = BertTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(model_path)
model_output="output/roberta_zh_large_finetuing"