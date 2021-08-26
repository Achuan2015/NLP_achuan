from transformers import BertTokenizer

max_seq_len=40
epochs=3
batch_size=128
weight_decay=1e-3
epochs=3
learning_rate=2e-5
model_path="input/bert-base-chinese/"
tokenizer = BertTokenizer.from_pretrained(model_path)
output_path="output/bert-base-chinese"